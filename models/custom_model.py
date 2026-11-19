import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel


def _inv_softplus(x):
    """返回满足 softplus(y) = x 的 y (x > 0)，用于初始化恒正的可学习参数。"""
    x = float(x)
    if x > 20.0:
        return x  # x 较大时 softplus(x) ≈ x
    return math.log(math.expm1(x))


def _inv_sigmoid(p):
    """返回满足 sigmoid(y) = p 的 y (0 < p < 1)，用于初始化残差系数 β。"""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class Model(nn.Module):
    """两层 MLP 基准模型：均值池化 + Linear-ReLU-Dropout-Linear。

    分类头支持两种可选配置 (use_ancestor_weighting):
    - False: 经典 MLP 分类头 (nn.Linear)，作为消融实验基准，与原始实现完全一致。
    - True:  "凸组合门控 + 祖先加权构造" 的层级感知分类头 (true-path 规则软约束)。
             分类头参数为自由项与祖先构造项的每类凸组合:
                 w_i = (1-γ_i) · w_free_i + γ_i · Σ_{j∈Anc(i)} α̂_ij · w_free_j
                 b_i = (1-γ_i) · b_free_i + γ_i · Σ_{j∈Anc(i)} α̂_ij · b_free_j
             其中祖先权重与 prototype_model 的 proto_w 计算逻辑一致
             (proto_w 乘祖先的均值原型特征，此处 α 乘祖先的自由分类头参数):
                 α_ij = exp(-γ · max(IC_i - IC_j, 0)) · n_j / (n_j + λ)
             α̂ 为 α 的行归一化版本 (对祖先加权平均并归一化)。
             凸组合门控 γ_i = β_i · τ / (n_i + τ) 对齐原型的样本量自适应旋钮
             (原型自分量权重 w = n/(n+τ)，祖先权重即 1-w = τ/(n+τ)):
             常见类 (n_i 大) γ_i→0，保留可靠的自有自由分量；罕见类 (n_i→0) γ_i→β_i，
             分类方向趋向祖先加权平均。β_i = sigmoid(raw_beta_i) ∈ (0,1) 为每类可学习
             耦合系数；τ = tau_min + softplus(raw_tau) 为可学习半饱和常数 (数据驱动初始化)。
             根节点 (无祖先) 强制 γ_i = 0，完全回退为自由参数。
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4,
                 use_ancestor_weighting=True, ancestor_mask=None, ic=None,
                 class_counts=None, alpha_lambda_init=10.0, ic_gamma_init=2.0,
                 beta_init=0.9, smooth_tau_init=30.0, tau_min=0.0):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_ancestor_weighting = use_ancestor_weighting
        # 有效分类头缓存（仅 eval 态使用，不进 state_dict）：推理期权重不变时，
        # 凸组合与祖先混合矩阵乘（C²·D）一次算好复用，消除稳态推理的主要开销；
        # 进入训练态或权重版本变化时自动失效重算
        self._eff_cache = None
        self._eff_cache_key = None

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        if use_ancestor_weighting:
            assert ancestor_mask is not None and ic is not None and class_counts is not None, \
                '祖先加权分类头需要传入 ancestor_mask / ic / class_counts'
            # 层级先验缓冲区 (由训练集 DAG 与标注统计得到，不参与训练)
            self.register_buffer('ancestor_mask', torch.as_tensor(ancestor_mask, dtype=torch.bool))
            self.register_buffer('ic', torch.as_tensor(ic, dtype=torch.float32))
            self.register_buffer('class_counts', torch.as_tensor(class_counts, dtype=torch.float32))
            # α 的可学习超参: softplus 重参数化保证恒正 (与原型网络一致)，配置值仅作为初始值
            # λ (alpha_lambda): 样本量敏感度;  γ (ic_gamma): IC 差衰减速率
            self.raw_alpha_lambda = nn.Parameter(torch.tensor(_inv_softplus(alpha_lambda_init), dtype=torch.float32))
            self.raw_ic_gamma = nn.Parameter(torch.tensor(_inv_softplus(ic_gamma_init), dtype=torch.float32))
            # 每类可学习耦合系数 β_i ∈ (0,1): sigmoid 重参数化保证数值稳定且有界，
            # 逐类控制凸组合门控的耦合上限 (以 beta_init 统一初始化)
            self.raw_beta = nn.Parameter(torch.full((num_classes,), _inv_sigmoid(beta_init), dtype=torch.float32))
            # 门控半饱和常数 τ ≥ tau_min: softplus 重参数化 (与原型网络的 smooth_tau 一致)，
            # 数据驱动初始值 smooth_tau_init 取类别正样本数中位数
            self.register_buffer('tau_min', torch.tensor(float(tau_min), dtype=torch.float32))
            self.raw_tau = nn.Parameter(torch.tensor(_inv_softplus(max(smooth_tau_init - float(tau_min), 1e-4)), dtype=torch.float32))
            # 自由参数 w_free / b_free，形状与原 nn.Linear 的 weight / bias 一致，
            # 初始化方式也与 nn.Linear 默认初始化保持一致
            self.w_free = nn.Parameter(torch.empty(num_classes, hidden_dim))
            self.b_free = nn.Parameter(torch.empty(num_classes))
            nn.init.kaiming_uniform_(self.w_free, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(hidden_dim)
            nn.init.uniform_(self.b_free, -bound, bound)
        else:
            # 经典 MLP 分类头 (消融基准)，保持原始实现
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
            )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.0,
            eps=1e-8
        )

    def _compute_ancestor_alpha(self):
        """计算祖先权重矩阵 α，逻辑与 prototype_model._compute_proto_w 一致。

        α[i][j] = exp(-γ · max(IC_i - IC_j, 0)) · n_j / (n_j + λ)，j ∈ Anc(i):
        - ΔIC = IC_i - IC_j: 惩罚过于泛化的祖先，可学习 γ 控制衰减速率;
        - n_j / (n_j + λ): 抑制样本量过小的罕见祖先，可学习 λ 调节样本量敏感度;
        - 非祖先位置为 0。

        Returns:
            alpha: (num_classes, num_classes) 祖先权重矩阵，alpha[i][j] 为祖先 j 相对功能 i 的权重
        """
        lam = F.softplus(self.raw_alpha_lambda)   # 可学习 λ > 0
        gamma = F.softplus(self.raw_ic_gamma)     # 可学习 γ > 0

        # ΔIC[i, j] = IC_i - IC_j
        ic_diff = self.ic.unsqueeze(1) - self.ic.unsqueeze(0)        # (C, C)
        ic_factor = torch.exp(-gamma * ic_diff.clamp(min=0.0))

        n = self.class_counts                                        # (C,)
        sample_factor = n / (n + lam)                                # 样本量缩放因子（按列广播，对应祖先 j）

        alpha = ic_factor * sample_factor.unsqueeze(0)
        alpha = alpha * self.ancestor_mask.float()                   # 非祖先位置置 0
        return alpha

    def _compute_class_gate(self):
        """计算每类凸组合门控 γ_i = β_i · τ / (n_i + τ)，对齐原型网络的样本量自适应旋钮
        （原型自分量权重 w = n/(n+τ)，祖先混合权重即 1-w = τ/(n+τ)，此处再乘每类 β_i）:
        常见类 (n_i 大) γ_i→0，保留可靠的自有自由分量；罕见类 (n_i→0) γ_i→β_i，
        趋向祖先加权平均。无祖先的根节点强制 γ_i = 0（完全回退自由参数）。

        Returns:
            gate: (num_classes,) 每类祖先混合权重，取值范围 [0, max_i β_i] ⊂ [0, 1)
        """
        beta = torch.sigmoid(self.raw_beta)                          # (C,) 每类可学习 β_i ∈ (0,1)
        tau = self.tau_min + F.softplus(self.raw_tau)                # 可学习 τ ≥ tau_min
        gate = beta * tau / (self.class_counts + tau)                # (C,)
        gate = torch.where(self.ancestor_mask.any(dim=1), gate, torch.zeros_like(gate))
        return gate

    def _cache_key(self):
        """缓存有效性键：相关可学习张量的版本号元组（任何 in-place 更新都会使其递增）。"""
        return (self.w_free._version, self.b_free._version,
                self.raw_beta._version, self.raw_tau._version,
                self.raw_alpha_lambda._version, self.raw_ic_gamma._version)

    def train(self, mode=True):
        self._eff_cache = None   # 进入训练态即失效，确保梯度路径不受缓存影响
        return super().train(mode)

    def _effective_classifier(self):
        """按"凸组合门控 + 祖先加权构造"计算有效分类头参数。

        γ_i = β_i · τ / (n_i + τ) (见 _compute_class_gate)，
        w_i = (1-γ_i) · w_free_i + γ_i · A_weight_i,  b_i 同理;
        A_weight_i = Σ_{j∈Anc(i)} α̂_ij · w_free_j (行归一化加权平均，与原型网络的 proto_w 用法一致)。
        凸组合保证 w_i 始终落在自身与祖先自由参数之间，层级耦合强度不受自由分量范数增长稀释
        （这是相对加性残差 w_free + β·A 的关键改进）。

        缓存：eval 态下权重不变，结果一次算好复用（以参数版本号为失效依据），
        稳态推理不再重复祖先混合矩阵乘；训练态始终现算以保证梯度正确。

        Returns:
            (weight, bias): 有效分类头参数，形状 (num_classes, hidden_dim) 与 (num_classes,)
        """
        if (not self.training and self._eff_cache is not None
                and self._eff_cache_key == self._cache_key()):
            return self._eff_cache

        alpha = self._compute_ancestor_alpha()                       # (C, C)
        alpha_sum = alpha.sum(dim=1, keepdim=True)                   # (C, 1)，根节点为 0
        alpha_norm = alpha / alpha_sum.clamp(min=1e-8)               # 行归一化 → 祖先加权平均
        gate = self._compute_class_gate()                            # (C,) 每类门控

        a_weight = alpha_norm @ self.w_free                          # (C, D)
        a_bias = alpha_norm @ self.b_free                            # (C,) 矩阵-向量积
        # weight 用 (C,1) 列向量广播、bias 用 (C,) 一维向量，避免 (C,1)+(C,) 混合广播
        weight = (1.0 - gate.unsqueeze(1)) * self.w_free + gate.unsqueeze(1) * a_weight  # (C, D)
        bias = (1.0 - gate) * self.b_free + gate * a_bias            # (C,)
        if not self.training:
            self._eff_cache = (weight, bias)
            self._eff_cache_key = self._cache_key()
        return weight, bias

    def get_ancestor_params(self):
        """返回 (lambda, gamma, tau, beta_mean) 的当前实际值，用于日志打印。"""
        lam = F.softplus(self.raw_alpha_lambda).item()
        gamma = F.softplus(self.raw_ic_gamma).item()
        tau = (self.tau_min + F.softplus(self.raw_tau)).item()
        beta_mean = torch.sigmoid(self.raw_beta).mean().item()
        return lam, gamma, tau, beta_mean

    def forward(self, input_feats, indices, labels):
        feats = self.proj(input_feats)
        if self.use_ancestor_weighting:
            weight, bias = self._effective_classifier()
            logits = F.linear(feats, weight, bias)
        else:
            logits = self.classifier(feats)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        return probs, logits, loss
