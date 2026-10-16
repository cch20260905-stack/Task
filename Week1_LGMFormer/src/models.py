# -*- coding: utf-8 -*-
"""基线模型：U-Net + ResNet18 编码器，输出单通道密度图。

论文对基线只说 "We use U-Net with a ResNet18 backbone as the baseline"，
解码器通道数、上采样方式均未公布（属于已知的超参缺口），此处采用 crowd-counting
社区通行配置：解码器 256/128/64/32，双线性上采样 + 双卷积，密度头 conv3x3 + conv1x1。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """双线性上采样 -> 与跳连接拼接 -> 双卷积。"""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HLFG(nn.Module):
    """High-Level Feature Guidance（论文 §III-B / Eq.1-3 / Fig.3）。

    用高层特征 Xlg（解码器更深一级的上采样特征）生成通道注意力，
    引导低层特征 Xres（编码器 ResLayer 输出），滤掉低层特征里的无关信息：
        W'  = Sigmoid(Linear(GAP(DWC(Xlg))))            (Eq.1 + reduce&expand÷4)
        Xhf = ReLU(BN(Conv3x3(W' ⊗ Xres)))              (Eq.2)
        X   = α·Xlg + β·Xhf + γ·Xres,  α+β+γ=1          (Eq.3, softmax 保证和为1)
    """

    def __init__(self, c_lg, c_res, c_out, reduction=4):
        super().__init__()
        self.align_lg = nn.Conv2d(c_lg, c_out, 1, bias=False)
        self.align_res = nn.Conv2d(c_res, c_out, 1, bias=False)
        # Eq.1: depthwise 3x3 + GAP -> 通道注意力
        self.dwc = nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out, bias=False)
        c_red = max(c_out // reduction, 4)               # reduce & expand（÷4 再还原）
        self.fc = nn.Sequential(
            nn.Linear(c_out, c_red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_red, c_out, bias=False),
            nn.Sigmoid(),
        )
        # Eq.2
        self.proj = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        # Eq.3 的加权聚合系数；零初始化 -> α=β=γ=1/3
        self.w = nn.Parameter(torch.zeros(3))

    def forward(self, xlg, xres):
        a_lg = self.align_lg(xlg)
        a_res = self.align_res(xres)
        w = self.dwc(a_lg).mean(dim=(2, 3))              # GAP -> (B,C)
        w = self.fc(w)[..., None, None]                  # W' -> (B,C,1,1)
        xhf = self.proj(w * a_res)
        alpha, beta, gamma = torch.softmax(self.w, dim=0)
        return alpha * a_lg + beta * xhf + gamma * a_res


class HLFGUpBlock(nn.Module):
    """上采样后的融合：HLFG 引导融合（对应论文 Fig.2 的 WA + High-Level Feature Guidance）。

    实测教训（不要再用前两种模式，留作消融对照）：
    - mode="replace"：严格按 Eq.3 用加权聚合替代 concat → 测试 MAE 8.23，比基线差 21%。
      原因：raw skip 只能经随机 1×1 投影进入融合，预训练 skip 特征被破坏，且 α/β/γ 学不动。
    - mode="add"：cat[x_up, hf] → 同样问题（raw skip 仍被投影），60 轮即劣化，已停跑。
    - mode="cat3"（默认）：cat[x_up, skip, hf] 三路拼接——基线的信息通路原样保留，
      HLFG 引导分支作为第三路并联，只增不减。
    """

    def __init__(self, cin, cskip, cout, mode="cat3"):
        super().__init__()
        self.mode = mode
        self.hlfg = HLFG(cin, cskip, cout)
        if mode == "replace":
            self.conv = DoubleConv(cout, cout)
        elif mode == "add":
            self.conv = DoubleConv(cin + cout, cout)
        else:                                   # cat3
            self.conv = DoubleConv(cin + cskip + cout, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        hf = self.hlfg(x, skip)
        if self.mode == "replace":
            return self.conv(hf)
        if self.mode == "add":
            return self.conv(torch.cat([x, hf], dim=1))
        return self.conv(torch.cat([x, skip, hf], dim=1))


class WMSA(nn.Module):
    """Window-based Multi-head Self-Attention（论文 §III-C-1，Swin 式窗内注意力）。

    与 Swin 的区别：不做 shifted window（两个连续 block 太贵），窗间交互交给
    十字卷积（cross-shaped convolution, Eq.4）。窗口大小 w 论文未公布，取 8。
    """

    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        self.dim, self.heads, self.w = dim, heads, window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        w, nh = self.w, self.heads
        c = C // nh
        ph, pw = (w - H % w) % w, (w - W % w) % w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Hp, Wp = x.shape[-2:]

        qkv = self.qkv(x)                      # B,3C,Hp,Wp
        # (B,3,nh,c,hw,w,ww,w) -> (3,B,hw,ww,nh,w,w,c) -> (3,B*nw,nh,w*w,c)
        q, k, v = qkv.view(B, 3, nh, c, Hp // w, w, Wp // w, w).permute(
            1, 0, 4, 6, 2, 5, 7, 3).reshape(3, -1, nh, w * w, c).unbind(0)
        att = (q @ k.transpose(-2, -1)) * self.scale        # 窗内注意力图
        out = att.softmax(dim=-1) @ v                       # Xwc: (B*nw,nh,ww,c)

        # 逆重排回 (B,C,Hp,Wp)
        out = out.reshape(B, Hp // w, Wp // w, nh, w, w, c).permute(
            0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out = self.proj(out)
        return out[:, :, :H, :W]


class LGFI(nn.Module):
    """Local–Global Feature Interaction（论文 §III-C / Eq.4-6 / Fig.4）。

    全局分支: BN -> WMSA(窗内自注意力) -> 十字卷积 Conv1×w + Convw×1 -> Xgc
    局部分支: 4 个膨胀率 1/2/3/4 的深度可分膨胀卷积 -> concat -> 3×3 融合卷积 -> +X -> Xlc
    输出:     Xlg = (X + Xgc + Xlc) -> BN -> MLP -> +残差   （按 Fig.4(a) 布线）

    未公布超参的处理：窗口/卷积长度 w=8、头数 4、MLP 扩展比 2、MDC 融合用普通
    3×3 卷积把 4C 压回 C（论文只写 DWC，但 DWC 不改通道数，无法与残差相加，
    必然还有一步降维，此处合并实现）。
    """

    def __init__(self, dim, heads=4, window=8, mlp_ratio=2):
        super().__init__()
        self.dim = dim
        self.bn_in = nn.BatchNorm2d(dim)
        self.wmsa = WMSA(dim, heads=heads, window=window)
        pw = window // 2
        self.cross_h = nn.Conv2d(dim, dim, (1, window), padding=0, groups=dim, bias=False)
        self.cross_v = nn.Conv2d(dim, dim, (window, 1), padding=0, groups=dim, bias=False)
        self.bn_mid = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1, bias=False),
        )
        self.dw_dil = nn.ModuleList(
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False)
            for d in (1, 2, 3, 4))
        # 论文只写 concat 后接 DWC，但 DWC 不改通道数，无法回到 C 维与残差相加，
        # 必然还有降维一步；按"深度可分"惯例实现为 DWC(3×3, groups=4C) + 1×1 压回 C。
        self.mdc_dw = nn.Conv2d(4 * dim, 4 * dim, 3, padding=1, groups=4 * dim, bias=False)
        self.mdc_pw = nn.Conv2d(4 * dim, dim, 1, bias=False)
        # 偶数核的十字卷积需要不对称 padding（左/上 (w-1)//2，右/下补余量）才能保持尺寸
        pl = (window - 1) // 2
        self.pad_h = (pl, window - 1 - pl, 0, 0)      # 水平 1×w 卷积: 补宽度
        self.pad_v = (0, 0, pl, window - 1 - pl)      # 垂直 w×1 卷积: 补高度

    def forward(self, x):
        h = self.bn_in(x)
        xwc = self.wmsa(h)                                   # 窗口上下文
        xgc = self.cross_h(F.pad(xwc, self.pad_h)) + \
              self.cross_v(F.pad(xwc, self.pad_v))           # Eq.4 全局上下文
        xlc = self.mdc_pw(self.mdc_dw(
            torch.cat([d(x) for d in self.dw_dil], dim=1))) + x        # Eq.5
        y = self.bn_mid(h + xgc + xlc)
        return y + self.mlp(y)                               # Eq.6 + 变换器残差


class SegBranch(nn.Module):
    """多尺度分割分支（论文式(8)：Xs = LGFI(Σ 上采样对齐的三级特征)）。

    输入三个解码级特征（/16、/8、/4），对齐到最细级后求和，过 LGFI 融合多尺度
    上下文，输出 3 类 logits（/4 分辨率——与密度头特征 Xd 同分辨率，供 SEDH 使用；
    训练时再上采样到原图算 CE/Dice）。
    """

    def __init__(self, c16, c8, c4, c=64):
        super().__init__()
        self.p16 = nn.Conv2d(c16, c, 1, bias=False)
        self.p8 = nn.Conv2d(c8, c, 1, bias=False)
        self.p4 = nn.Conv2d(c4, c, 1, bias=False)
        self.lgfi = LGFI(c)
        self.cls = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, 1),
        )

    def forward(self, f16, f8, f4):
        t = self.p4(f4)
        t = t + self.p8(F.interpolate(f8, size=t.shape[-2:], mode="bilinear",
                                      align_corners=False))
        t = t + self.p16(F.interpolate(f16, size=t.shape[-2:], mode="bilinear",
                                       align_corners=False))
        return self.cls(self.lgfi(t))          # (B,3,H/4,W/4)


def resnet18_encoder(pretrained=True):
    """返回 ResNet18 的各阶段特征提取器。预训练权重缺失时自动退化为随机初始化。"""
    import torchvision
    try:
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        got = "ImageNet 预训练" if pretrained else "随机初始化"
    except Exception as e:                                    # 离线环境兜底
        print("[warn] 预训练权重加载失败(%s)，改用随机初始化" % type(e).__name__)
        r = torchvision.models.resnet18(weights=None)
        got = "随机初始化"
    return r, got


class UNetResNet18(nn.Module):
    def __init__(self, pretrained=True, dec=(256, 128, 64, 32),
                 head_act="softplus", head_bias_init=-4.13,
                 use_hlfg=False, hlfg_mode="cat3", use_lgfi=False,
                 use_seg=False, use_sedh=False):
        """head_bias_init 默认值 -4.13 对应 softplus(b)=0.0161，
        即 ARIS 密度均值 1.6e-4 × 默认缩放系数 100。
        use_hlfg=True 时按论文在 3 个深层融合点(up4/up3/up2)换成 HLFG 引导融合，
        对应消融表的 Baseline + HLFG 配置（最细的 up1/stem 层保持普通拼接）。
        use_lgfi=True 时加 3 个 LGFI 模块（瓶颈 1/32 + 解码 1/16、1/8），
        对应消融表的 Baseline + LGFI / Baseline + HLFG + LGFI；
        此时 LGFI 输出即论文里的 Xlg，作为对应 HLFG 的高层引导输入。
        use_seg=True 加多尺度分割分支（式8），对应消融表 +分割头(MTL) 行；
        use_sedh=True 再加 SEDH（式9-11，用 P_f/P_n 概率图精修密度特征），
        两者合起来即完整 LGMFormer 的密度路径。"""
        super().__init__()
        r, self.encoder_init = resnet18_encoder(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # /4, 64
        self.layer1, self.layer2 = r.layer1, r.layer2                  # /4 64, /8 128
        self.layer3, self.layer4 = r.layer3, r.layer4                  # /16 256, /32 512

        if use_hlfg:
            self.up4 = HLFGUpBlock(512, 256, dec[0], mode=hlfg_mode)
            self.up3 = HLFGUpBlock(dec[0], 128, dec[1], mode=hlfg_mode)
            self.up2 = HLFGUpBlock(dec[1], 64, dec[2], mode=hlfg_mode)
        else:
            self.up4 = UpBlock(512, 256, dec[0])
            self.up3 = UpBlock(dec[0], 128, dec[1])
            self.up2 = UpBlock(dec[1], 64, dec[2])
        self.up1 = UpBlock(dec[2], 64, dec[3])
        if use_lgfi:
            self.lgfi0 = LGFI(512)          # 瓶颈: 最终 ResLayer 输出 (1/32)
            self.lgfi1 = LGFI(dec[0])       # 解码 1/16（WA 模块输出）
            self.lgfi2 = LGFI(dec[1])       # 解码 1/8
        self.use_seg = use_seg
        self.use_sedh = use_sedh
        if use_seg:
            self.seg_branch = SegBranch(dec[0], dec[1], dec[2])
        if use_sedh:
            assert use_seg, "SEDH 依赖分割分支的 P_f/P_n，需同时开 use_seg"
        self.head = nn.Sequential(
            nn.Conv2d(dec[3], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.act_name = head_act
        if head_act == "softplus":
            # 稀疏密度回归的三个数值坑，这里一并处理：
            # (1) ARIS 密度图均值仅 ~1.6e-4，随机初始化的密度头却输出 O(0.1)，
            #     初始预测鱼数是真值的 700+ 倍，白耗训练轮次；
            # (2) 若用 ReLU 且把末层零初始化（让初始输出≈0），预激活恰好等于 0，
            #     而 relu'(0)=0，梯度会被完全掐断——模型永远停在"输出恒为0"；
            # (3) 末层权重若严格为 0，上游梯度也拿不到信号。故用极小非零初始化。
            # softplus 处处可导且非负；偏置按 softplus(b) = 密度均值×缩放系数 反解，
            # 使初始预测恰好落在真实密度量级（见 train.py 的 --density-scale）。
            nn.init.normal_(self.head[-1].weight, std=1e-3)
            nn.init.constant_(self.head[-1].bias, head_bias_init)
        elif head_act == "relu":
            nn.init.constant_(self.head[-1].bias, 1e-3)   # 避开 relu 的 0 点
        self.act = {"softplus": nn.Softplus(), "relu": nn.ReLU(inplace=True),
                    "none": nn.Identity()}[head_act]

    def forward(self, x):
        e0 = self.stem(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        b = self.lgfi0(e4) if hasattr(self, "lgfi0") else e4
        d4 = self.up4(b, e3)
        if hasattr(self, "lgfi1"):
            d4 = self.lgfi1(d4)
        d3 = self.up3(d4, e2)
        if hasattr(self, "lgfi2"):
            d3 = self.lgfi2(d3)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)

        seg_logits = None
        if self.use_seg:
            seg4 = self.seg_branch(d4, d3, d2)             # /4 分辨率 logits
            seg_logits = F.interpolate(seg4, size=x.shape[-2:], mode="bilinear",
                                       align_corners=False)
            if self.use_sedh:
                prob = seg4.softmax(dim=1)
                # data.py 约定: 0=背景 1=鱼 2=噪声（见 generate_seg_labels.py）
                pf, pn = prob[:, 1:2], prob[:, 2:3]        # 鱼(1)/噪声(2) 概率图
                xe = d1 * torch.sigmoid(pf) + d1           # 式(9): 增强鱼区
                xr = self.head[:-1](xe)                    # 式(10) Convseq
                xf = xr * (1 - torch.sigmoid(pn)) + xr     # 式(11): 抑制噪声区
                d = F.interpolate(xf, size=x.shape[-2:], mode="bilinear",
                                  align_corners=False)
                return self.act(self.head[-1](d)), seg_logits

        d = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.act(self.head(d)), seg_logits          # 密度图非负

    def param_groups(self, lr_base, lr_encoder):
        """论文的 layerwise learning rate：编码器用更小的学习率。"""
        enc, dec = [], []
        for n, p in self.named_parameters():
            (enc if (n.startswith(("stem", "layer1", "layer2", "layer3", "layer4"))) else dec).append(p)
        return [{"params": enc, "lr": lr_encoder}, {"params": dec, "lr": lr_base}]


class LGMFormer(nn.Module):
    """论文完整架构（Fig.2 / §III-A）：ResNet18 编码器 + transformer 解码器。

    解码器逐级：上采样 -> HLFG（高层引导低层融合，Eq.1-3）-> LGFI（transformer
    block，Eq.4-6）。三个 LGFI 输出（/32、/16、/8）上采样到 /4 逐元素相加，再过
    第四个 LGFI 得分割 logits——论文 §III-A 原文 "Outputs from the three LGFI
    modules are summed and then passed into another LGFI module for multiclass
    semantic segmentation"。密度路径：/4 级 HLFG 即论文的 Weighted Aggregation，
    其输出经 SEDH（式9-11，用分割的 P_f/P_n 概率精修）后上采样出密度图。

    未公布超参的自定处理：解码通道 dec=(256,128,64,32)（与 U-Net 版一致）、
    瓶颈 512->dec[0] 用 1×1 投影、上采样用双线性（论文只说 upsampling decoder，
    未公布方式）、seg 投影统一到 64 通道。
    """

    def __init__(self, pretrained=True, dec=(256, 128, 64, 32),
                 head_act="softplus", head_bias_init=-4.13,
                 use_seg=True, use_sedh=True, fusion="paper"):
        """fusion: paper=严格按论文 Eq.3，HLFG 加权聚合输出直接进 LGFI；
        cat=每级融合改为 cat[up, skip, hlfg] 后 1×1 投影再进 LGFI——
        保留基线的 raw skip 通路（U-Net 消融已证明 Eq.3 独占融合会削弱 skip，
        cat3 模式把 MAE 从 8.2 拉回 6.94）。"""
        super().__init__()
        r, self.encoder_init = resnet18_encoder(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # /4 64
        self.layer1, self.layer2 = r.layer1, r.layer2                  # /4 64, /8 128
        self.layer3, self.layer4 = r.layer3, r.layer4                  # /16 256, /32 512

        self.use_seg, self.use_sedh = use_seg, use_sedh
        self.fusion = fusion
        if use_sedh:
            assert use_seg, "SEDH 依赖分割分支的 P_f/P_n，需同时开 use_seg"

        # ---- transformer 解码器主线（Fig.2）----
        self.in_proj = nn.Conv2d(512, dec[0], 1, bias=False)
        self.lgfi_b = LGFI(dec[0])                    # /32 瓶颈 LGFI
        self.hlfg_16 = HLFG(dec[0], 256, dec[1])      # 融合 e3
        self.lgfi_16 = LGFI(dec[1])                   # /16
        self.hlfg_8 = HLFG(dec[1], 128, dec[2])       # 融合 e2
        self.lgfi_8 = LGFI(dec[2])                    # /8
        self.hlfg_4 = HLFG(dec[2], 64, dec[3])        # 融合 e1 = Weighted Aggregation (/4)
        if fusion == "cat":
            # cat 模式：每级 cat[up, skip, hlfg] 后 1×1 投影回该级通道
            cs = (256, 128, 64)                       # e3/e2/e1 的通道
            self.fuse = nn.ModuleList(
                nn.Conv2d(dec[i] + cs[i] + dec[i + 1], dec[i + 1], 1, bias=False)
                for i in range(3))

        # ---- 分割分支：三个 LGFI 输出对齐 /4 求和 -> 第四个 LGFI -> seg head ----
        if use_seg:
            self.seg_proj = nn.ModuleList(
                nn.Conv2d(dec[i], 64, 1, bias=False) for i in range(3))
            self.seg_lgfi = LGFI(64)
            self.seg_head = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),
            )

        # ---- 密度头（SEDH 的 Eq.10 Convseq 复用 head[:-1]）----
        self.head = nn.Sequential(
            nn.Conv2d(dec[3], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.act_name = head_act
        if head_act == "softplus":
            # 稀疏密度回归初始化（与 U-Net 版同理，见 UNetResNet18 注释）
            nn.init.normal_(self.head[-1].weight, std=1e-3)
            nn.init.constant_(self.head[-1].bias, head_bias_init)
        elif head_act == "relu":
            nn.init.constant_(self.head[-1].bias, 1e-3)
        self.act = {"softplus": nn.Softplus(), "relu": nn.ReLU(inplace=True),
                    "none": nn.Identity()}[head_act]

    def forward(self, x):
        e0 = self.stem(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        def up(t, sz):
            return F.interpolate(t, size=sz, mode="bilinear", align_corners=False)

        b = self.lgfi_b(self.in_proj(e4))                              # /32 LGFI
        u3, u2, u1 = up(b, e3.shape[-2:]), None, None
        if self.fusion == "cat":
            d3 = self.lgfi_16(self.fuse[0](torch.cat([u3, e3,
                                                      self.hlfg_16(u3, e3)], dim=1)))
        else:
            d3 = self.lgfi_16(self.hlfg_16(u3, e3))                    # /16
        u2 = up(d3, e2.shape[-2:])
        if self.fusion == "cat":
            d2 = self.lgfi_8(self.fuse[1](torch.cat([u2, e2,
                                                      self.hlfg_8(u2, e2)], dim=1)))
        else:
            d2 = self.lgfi_8(self.hlfg_8(u2, e2))                      # /8
        u1 = up(d2, e1.shape[-2:])
        if self.fusion == "cat":
            d1 = self.fuse[2](torch.cat([u1, e1,
                                         self.hlfg_4(u1, e1)], dim=1))  # /4 WA + raw 通路
        else:
            d1 = self.hlfg_4(u1, e1)                                   # /4 Weighted Aggregation

        seg4 = None
        if self.use_seg:
            sz4 = e1.shape[-2:]
            t = up(self.seg_proj[2](d2), sz4)                          # /8 -> /4
            t = t + self.seg_proj[1](up(d3, sz4))                      # /16 -> /4
            t = t + self.seg_proj[0](up(b, sz4))                       # /32 -> /4
            seg4 = self.seg_head(self.seg_lgfi(t))                     # /4 logits(3类)

        seg_logits = None
        if self.use_seg:
            seg_logits = up(seg4, x.shape[-2:])
            if self.use_sedh:
                prob = seg4.softmax(dim=1)
                # data.py 约定: 0=背景 1=鱼 2=噪声（见 generate_seg_labels.py）
                pf, pn = prob[:, 1:2], prob[:, 2:3]
                xe = d1 * torch.sigmoid(pf) + d1                       # 式(9) 增强鱼区
                xr = self.head[:-1](xe)                                # 式(10) Convseq
                xf = xr * (1 - torch.sigmoid(pn)) + xr                 # 式(11) 抑制噪声区
                d = up(xf, x.shape[-2:])
                return self.act(self.head[-1](d)), seg_logits

        d = up(d1, x.shape[-2:])
        return self.act(self.head(d)), seg_logits

    def param_groups(self, lr_base, lr_encoder):
        """论文的 layerwise learning rate：编码器用更小的学习率。"""
        enc, dec = [], []
        for n, p in self.named_parameters():
            (enc if (n.startswith(("stem", "layer1", "layer2", "layer3", "layer4"))) else dec).append(p)
        return [{"params": enc, "lr": lr_encoder}, {"params": dec, "lr": lr_base}]
