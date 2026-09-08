import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    auc, average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch_geometric.nn import GATv2Conv


class FocalLoss(nn.Module):
    """Focal Loss for binary classification: FL = -alpha * (1-p)^gamma * log(p)
    Reduces loss for well-classified examples, focusing on hard negatives/positives.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


class LogitAdjustedLoss(nn.Module):
    """Logit Adjustment Loss (Menon et al., ICLR 2021)

    针对类别不平衡，比 Focal Loss 更有理论保证：训练时把类别先验的
    log 值加到 logits 上，等价于在贝叶斯最优决策中扣除先验偏置。

        logit_adjusted = logits + tau * log(pi_pos / pi_neg)

    其中 pi_pos, pi_neg 是训练集正/负样本的先验频率。
    对二分类，只需调整正类 logit 相对负类的偏移：
        adjusted = logits + tau * log(pi_pos / pi_neg)

    优势（相比 Focal + 采样）：
      - 不丢数据、不重复数据：全量 600 正 + 6000 负每 epoch 都参与
      - 无需手调 alpha：偏移量由数据先验直接算出
      - 推理时用 0.5 阈值即为 balanced 决策，无需 threshold-moving
      - 理论上优化 balanced error（测试时平衡评估最优）

    Args:
        pos_prior: 正样本先验频率 (n_pos / N)
        tau:       调整强度，默认 1.0
    """
    def __init__(self, pos_prior, tau=1.0):
        super().__init__()
        pos_prior = max(min(pos_prior, 1 - 1e-6), 1e-6)
        neg_prior = 1.0 - pos_prior
        # 正类相对负类的 log 先验偏移（标量）
        self.register_buffer(
            'offset',
            torch.tensor(tau * np.log(pos_prior / neg_prior), dtype=torch.float32)
        )

    def forward(self, logits, targets):
        # 训练时将先验偏移加到 logits，再做标准 BCE
        adjusted = logits + self.offset
        return F.binary_cross_entropy_with_logits(adjusted, targets)


class ExpertLoadBalancingLoss(nn.Module):
    """专家负载均衡损失：L = N * Σ(f_i²)，其中 f_i 是专家 i 的平均使用频率

    当所有专家均匀使用时，L_min = 1（每个 f_i = 1/N → Σ = N*(1/N²) = 1/N → ×N = 1）
    当所有样本集中到一个专家时，L_max ≈ N
    """
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, gate_weights, valid_mask=None):
        # gate_weights: [B, L, num_experts]
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).to(gate_weights.dtype)
            denom = mask.sum(dim=[0, 1]).clamp(min=1.0)
            expert_usage = (gate_weights * mask).sum(dim=[0, 1]) / denom
        else:
            expert_usage = gate_weights.mean(dim=[0, 1])  # [num_experts]
        return self.num_experts * (expert_usage ** 2).sum()  # scalar


class ProteinClassifier(nn.Module):
    """双通道(语义 + 几何) + 残基级 MoE 融合 + 多代表点注意力池化

    设计动机：
    保护性抗原的抗原决定簇既依赖一级序列语义，也依赖三维空间几何。
    两类信息由两条机制不同的通道分别编码，再在残基级自适应融合：
      - 语义通道：SaProt(结构感知的序列语言模型) → Transformer → proj
                  提供富含进化/序列语义的残基表征
      - 几何通道：以残基三维坐标算出的 SE(3)-不变几何特征(残基对距离 + 相对
                  方向)作为边特征，用 GATv2 在残基接触图上做几何驱动的消息传递
                  提供三维空间结构表征

    残基级自适应融合：
      1. 门控标量插值：gate ∈ [0,1] 由 concat(语义, 几何) 经 sigmoid 网络预测
         gate ≈ 1 → 偏序列，gate ≈ 0 → 偏结构
         fused = gate × seq + (1-gate) × graph
      2. 残基级 MoE：以 concat(语义, 几何) 为输入做 top-k 稀疏路由，
         每个专家为 2H→H 的非线性 MLP，负载均衡损失防坍缩
      3. 门控与 MoE 并行，残差相加后经 LayerNorm

    多代表点注意力池化(replace 全局单 query pooling)：
      先用 learnable query 做残基级 MIL attention 得到 mil_weights；再取 top-K 高权重
      残基作为代表点，每个代表点作为 query 对全部残基做一次 attention pooling，最后按
      代表点权重加权平均。这是多代表点(multi-representative)注意力池化，不是聚类算法。

    输入:
      struc_input:  [B, L_s, 1280]  SaProt 逐残基嵌入
      graph_data:   torch_geometric Batch  (x=[N,1280], edge_index, coords=[N,3], batch)
      struc_mask:   [B, L_s]  1=有效, 0=padding
    输出:
      logits:               [B, 1]
      mil_weights:          [B, L]   MIL attention 权重 → 表位热图
      load_balancing_loss:  scalar   专家负载均衡损失
      gate:                 [B, L, 1] 门控值（gate ≈ 1 偏序列，gate ≈ 0 偏结构）
      aux (可选):           dict     {'gate': [B, L, 1],          # 门控值（可解释性）
                                      'router_weights': [B, L, num_experts],  # 专家路由权重
                                      'topk_indices':  [B, L, top_k],         # MoE 专家索引
                                      'rep_attention': [B, K, L],            # 每个代表点对残基的 attention
                                      'rep_weights':   [B, K],               # 每个代表点的聚合权重
                                      'rep_indices':   [B, K],               # 代表点残基索引
                                      'pooling_attention': [B, L],           # 最终池化注意力
                                      'residue_contributions': [B, L]}       # 阳性 logit 的有符号残基贡献
                                     仅当 return_aux=True 时返回
    """
    def __init__(self, input_size=1280, hidden_size=256, dropout_rate=0.50, num_heads=4,
                 num_experts=4, top_k=2, num_clusters=4):
        super(ProteinClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_clusters = num_clusters

        # ═══════════════════════════════════════════════════════
        # 语义通道：SaProt → Transformer → proj → [B, L, H]
        # ═══════════════════════════════════════════════════════
        self.seq_attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.seq_norm = nn.LayerNorm(input_size)
        self.seq_proj = nn.Linear(input_size, hidden_size)

        # ═══════════════════════════════════════════════════════
        # 几何通道：以三维坐标算出的几何边特征(距离 + 相对方向)驱动
        # 的 GATv2 消息传递 → [B, L, H]
        #   edge_attr 维度 = 4：[1/(1+d), 单位方向向量 dx,dy,dz]，SE(3)-不变(距离)
        #   + SO(3)-等变方向的标量投影（对全局平移不变；方向随整体旋转变化，
        #     但网络学到的是相对几何关系）
        #   注：几何图边数大(平均度~9，长蛋白单图上万条边)，GATv2 的 edge attention
        #   显存随 边数×heads 线性增长，故几何通道 heads 独立设小(默认 2)以控显存。
        # ═══════════════════════════════════════════════════════
        self.geom_edge_dim = 4
        geom_heads = min(2, num_heads)
        self.conv1 = GATv2Conv(input_size, hidden_size, heads=geom_heads,
                               concat=False, dropout=dropout_rate,
                               edge_dim=self.geom_edge_dim)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, heads=geom_heads,
                               concat=False, dropout=dropout_rate,
                               edge_dim=self.geom_edge_dim)
        self.gnn_dropout = nn.Dropout(dropout_rate)

        # ═══════════════════════════════════════════════════════
        # 门控网络：残基级标量插值融合
        # gate ∈ [0,1]，由 concat(语义, 几何) 经 sigmoid 网络得到。
        # gate 表征该残基对序列与结构的依赖倾向（gate≈1 偏序列，gate≈0 偏结构），
        # 融合结果为 gate·seq + (1-gate)·graph。
        # 门控与 MoE 并行、互补：门控轻量可解释，MoE 非线性表达力强。
        # ═══════════════════════════════════════════════════════
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # ═══════════════════════════════════════════════════════
        # 残基级 MoE 融合：在 concat(语义, 几何) [B, L, 2H] 上做 top-k 稀疏路由
        # 两来源在此仍可分，专家可学习不同融合策略：
        #   偏语义 / 偏几何 / 协同 / 低活性
        # 每个专家 2H → H，输出为残基级融合特征
        # ═══════════════════════════════════════════════════════
        self.fusion_gate = nn.Linear(hidden_size * 2, num_experts)

        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ) for _ in range(num_experts)
        ])

        self.fusion_norm = nn.LayerNorm(hidden_size)

        # ═══════════════════════════════════════════════════════
        # 多代表点注意力池化：learnable query → mil_weights →
        # top-K 代表点 → 每个代表点 attention pooling → 加权平均
        # ═══════════════════════════════════════════════════════
        self.mil_query = nn.Parameter(
            torch.randn(1, 1, hidden_size) * (hidden_size ** -0.5)
        )
        self.mil_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # ═══════════════════════════════════════════════════════
        # 分类头
        # ═══════════════════════════════════════════════════════
        self.classifier = nn.Linear(hidden_size, 1)

        # ═══════════════════════════════════════════════════════
        # 负载均衡损失
        # ═══════════════════════════════════════════════════════
        self.expert_load_balancing_loss = ExpertLoadBalancingLoss(num_experts=num_experts)

    def _graph_to_batched(self, x, batch):
        """将 GCN 逐节点输出 [N, H] 按样本拆分为 [B, L_g, H] + graph_mask [B, L_g]
        按 batch 内最大长度 padding，补齐 0 并用 mask 标记。
        """
        num_graphs = int(batch.max().item()) + 1
        residue_list = []
        len_list = []
        for g in range(num_graphs):
            mask = (batch == g)
            x_g = x[mask]  # [L_g, H]
            residue_list.append(x_g)
            len_list.append(x_g.size(0))

        max_len = max(len_list)
        B = num_graphs
        H = x.size(1)
        padded = torch.zeros(B, max_len, H, device=x.device, dtype=x.dtype)
        graph_mask = torch.zeros(B, max_len, device=x.device, dtype=torch.bool)
        for i, (res, ln) in enumerate(zip(residue_list, len_list)):
            padded[i, :ln, :] = res
            graph_mask[i, :ln] = True

        return padded, graph_mask

    @staticmethod
    def _build_geom_edge_attr(coords, edge_index):
        """从残基三维坐标构造几何边特征

        edge_attr = [1/(1+d), dx/d, dy/d, dz/d]，维度 4：
          - 1/(1+d)：残基对 CA 距离的平滑倒数(距离对全局平移/旋转不变)
          - (dx,dy,dz)/d：源→目标残基的单位方向向量(对平移不变)
        提供裸接触图(仅 0/1 邻接)所缺失的连续几何信号。

        Args:
            coords:     [N, 3]  批内所有残基的 CA 坐标(已按 graph 拼接)
            edge_index: [2, E]
        Returns:
            edge_attr:  [E, 4]
        """
        src, dst = edge_index[0], edge_index[1]
        diff = coords[dst] - coords[src]                 # [E, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)    # [E, 1]
        inv_dist = 1.0 / (1.0 + dist)                    # [E, 1]
        direction = diff / (dist + 1e-8)                 # [E, 3]
        return torch.cat([inv_dist, direction], dim=-1)  # [E, 4]

    def _fusion_moe(self, concat_feat):
        """残基级 MoE 融合：在 concat(语义, 几何) [B, L, 2H] 上做 top-k 稀疏路由。

        路由发生在两通道尚未混合之前，专家可学习不同融合策略(偏语义/偏几何/协同)，
        每个专家把 2H 融合投影到 H。

        Args:
            concat_feat: [B, L, 2H]  语义与几何通道的拼接特征
        Returns:
            out:            [B, L, H]        MoE 融合输出
            router_weights: [B, L, num_experts]  路由权重(负载均衡 + 可解释性)
            topk_indices:   [B, L, top_k]        每残基选中的专家索引
        """
        B, L = concat_feat.shape[:2]
        device = concat_feat.device

        gate_logits = self.fusion_gate(concat_feat)  # [B, L, num_experts]

        # Top-k 稀疏路由
        topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        mask = torch.full_like(gate_logits, float('-inf'))
        mask.scatter_(2, topk_indices, topk_vals)
        router_weights = F.softmax(mask, dim=-1)  # [B, L, num_experts]

        # 展平以便按残基索引
        flat_feat = concat_feat.view(-1, concat_feat.size(-1))      # [B*L, 2H]
        flat_rw = router_weights.view(-1, router_weights.size(-1))  # [B*L, num_experts]

        # 只计算被选中的专家（真正条件计算）
        out_flat = torch.zeros(B * L, self.hidden_size, device=device)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                residue_mask = (topk_indices[:, :, k] == e)  # [B, L]
                if not residue_mask.any():
                    continue
                flat_mask = residue_mask.view(-1)             # [B*L]
                expert_out = self.fusion_experts[e](flat_feat[flat_mask])  # [sum, H]
                out_flat[flat_mask] += flat_rw[flat_mask, e:e+1] * expert_out

        return out_flat.view(B, L, -1), router_weights, topk_indices

    def _multi_representative_pooling(self, fused, mil_weights, aligned_mask):
        """多代表点注意力池化(multi-representative attention pooling)。

        注意：这不是聚类算法，K 个代表点的 attention 可以关注重叠残基。
        流程：取 top-K 高 MIL 权重残基作为代表点 → 每个代表点作为 query 对全部残基
        做一次 attention pooling → 按代表点权重加权平均。相比单 query 全局池化，允许
        模型从多个高显著性锚点聚合信息。

        Args:
            fused:        [B, L, H]  融合特征
            mil_weights:  [B, L]     原始 MIL attention 权重
            aligned_mask: [B, L]     bool 有效残基掩码
        Returns:
            pooled:            [B, H]     池化输出
            rep_attention:     [B, K, L]  每个代表点的 attention 权重
            representatives:   [B, K, H]  代表点特征
        """
        B, L, H = fused.shape
        K = min(self.num_clusters, L)  # 最多 num_clusters 个代表点

        # Step 1: 取 top-K 高权重残基作为代表点
        if aligned_mask is not None:
            masked_weights = mil_weights.masked_fill(~aligned_mask, 0.0)
        else:
            masked_weights = mil_weights

        topk_vals, rep_indices = torch.topk(masked_weights, K, dim=-1)  # [B, K]
        representatives = torch.gather(fused, 1,
                                       rep_indices.unsqueeze(-1).expand(-1, -1, H))  # [B, K, H]

        # Step 2: 每个代表点作为 query，对所有残基做 attention
        query = representatives  # [B, K, H]
        fused_proj = self.mil_proj(fused)  # [B, L, H]
        raw_scores = torch.bmm(query, fused_proj.transpose(1, 2))  # [B, K, L]
        raw_scores = raw_scores / (H ** 0.5)

        if aligned_mask is not None:
            raw_scores = raw_scores.masked_fill(~aligned_mask.unsqueeze(1), float('-inf'))

        rep_attention = F.softmax(raw_scores, dim=-1)  # [B, K, L]

        # Step 3: 每个代表点做 weighted pooling
        pooled_per_rep = torch.bmm(rep_attention, fused)  # [B, K, H]

        # Step 4: 对代表点做加权平均，权重是代表点的显著性
        rep_weights = F.softmax(topk_vals, dim=-1)  # [B, K]
        pooled = torch.bmm(rep_weights.unsqueeze(1), pooled_per_rep).squeeze(1)  # [B, H]

        return pooled, rep_attention, representatives, rep_weights, rep_indices

    def forward(self, struc_input, graph_data, struc_mask=None, return_aux=False):
        B = struc_input.size(0)
        device = struc_input.device

        # ═══════════════════════════════════════════════════════
        # 1. 语义通道: [B, L_s, 1280] → [B, L_s, H]
        # ═══════════════════════════════════════════════════════
        key_padding_seq = None
        if struc_mask is not None:
            key_padding_seq = (struc_mask == 0)

        seq_attended, _ = self.seq_attention(
            struc_input, struc_input, struc_input,
            key_padding_mask=key_padding_seq
        )
        seq_attended = self.seq_norm(seq_attended)
        seq_residue = self.seq_proj(seq_attended)  # [B, L_s, H]
        L_s = seq_residue.size(1)

        # ═══════════════════════════════════════════════════════
        # 2. 几何通道: 坐标 → 几何边特征(距离+方向) → GATv2×2 → [B, L_g, H]
        # ═══════════════════════════════════════════════════════
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        coords = graph_data.coords  # [N, 3]
        edge_attr = self._build_geom_edge_attr(coords, edge_index)  # [E, 4]
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.gnn_dropout(F.relu(self.conv2(x, edge_index, edge_attr=edge_attr)))  # [N, H]

        graph_residue, graph_mask = self._graph_to_batched(x, batch)
        L_g = graph_residue.size(1)

        # ═══════════════════════════════════════════════════════
        # 3. 对齐: L = min(L_s, L_g)
        # ═══════════════════════════════════════════════════════
        L = min(L_s, L_g)
        seq_aligned = seq_residue[:, :L, :]  # [B, L, H]
        graph_aligned = graph_residue[:, :L, :]  # [B, L, H]
        aligned_mask = graph_mask[:, :L]  # [B, L] bool

        if struc_mask is not None:
            seq_aligned_mask = struc_mask[:, :L].bool()
            aligned_mask = aligned_mask & seq_aligned_mask

        # ═══════════════════════════════════════════════════════
        # 4. 残基级自适应融合：门控标量插值 + MoE 非线性变换（并行）
        # 两者都以 concat(语义, 几何) 为输入，互补而非冗余。
        # ═══════════════════════════════════════════════════════
        concat_feat = torch.cat([seq_aligned, graph_aligned], dim=-1)  # [B, L, 2H]
        gate = self.gate_net(concat_feat)  # [B, L, 1]
        gate_fused = gate * seq_aligned + (1 - gate) * graph_aligned  # [B, L, H]
        moe_out, router_weights, topk_indices = self._fusion_moe(concat_feat)  # [B,L,H]
        fused = self.fusion_norm(gate_fused + moe_out)  # [B, L, H]

        # ═══════════════════════════════════════════════════════
        # 5. 多代表点注意力池化: learnable query → mil_weights → top-K 代表点池化
        # ═══════════════════════════════════════════════════════
        query = self.mil_query.expand(B, -1, -1)  # [B, 1, H]
        fused_proj = self.mil_proj(fused)  # [B, L, H]
        raw_scores = (query * fused_proj).sum(dim=-1)  # [B, L]

        if aligned_mask is not None:
            raw_scores = raw_scores.masked_fill(~aligned_mask, float('-inf'))

        mil_weights = F.softmax(raw_scores, dim=-1)  # [B, L]

        pooled, rep_attention, representatives, rep_weights, rep_indices = self._multi_representative_pooling(
            fused, mil_weights, aligned_mask
        )
        logits = self.classifier(pooled)  # [B, 1]

        # Exact residue-level decomposition of the classifier logit:
        #   pooling_attention[j] = sum_k rep_weights[k] * rep_attention[k, j]
        #   logit = classifier.bias + sum_j pooling_attention[j] * (w^T fused[j])
        # Unlike attention alone, residue_contributions is class-directional:
        # positive values support the protective-antigen class and negative values oppose it.
        pooling_attention = torch.bmm(
            rep_weights.unsqueeze(1), rep_attention
        ).squeeze(1)  # [B, L]
        residue_logits = F.linear(
            fused, self.classifier.weight, bias=None
        ).squeeze(-1)  # [B, L]
        residue_contributions = pooling_attention * residue_logits  # [B, L]

        # ═══════════════════════════════════════════════════════
        # 负载均衡损失
        # ═══════════════════════════════════════════════════════
        lb_loss = self.expert_load_balancing_loss(router_weights, aligned_mask)

        if return_aux:
            aux = {
                'gate': gate,                          # [B, L, 1] 门控值（可解释性）
                'router_weights': router_weights,       # [B, L, num_experts]
                'topk_indices': topk_indices,           # [B, L, top_k]  MoE 专家索引
                'rep_attention': rep_attention,         # [B, K, L]  每个代表点对残基的 attention
                'rep_weights': rep_weights,             # [B, K]     每个代表点的聚合权重
                'rep_indices': rep_indices,             # [B, K]     代表点残基索引
                'raw_scores': raw_scores,               # [B, L]     MIL softmax 前的逐残基分数
                'pooling_attention': pooling_attention, # [B, L]     最终池化注意力（非类别贡献）
                'residue_logits': residue_logits,       # [B, L]     不含池化权重的逐残基类别分数
                'residue_contributions': residue_contributions,  # [B, L] 对阳性 logit 的有符号贡献
            }
            return logits, mil_weights, lb_loss, aux

        return logits, mil_weights, lb_loss


def calculate_metrics(y_true, y_pred, y_prob):
    """计算各种评估指标"""

    def safe_mcc(cm):
        TN, FP, FN, TP = cm.ravel()
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        return 0.0 if denominator == 0 else numerator / denominator

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    ap = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    mcc = safe_mcc(cm)

    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold_idx = min(best_idx, len(pr_thresholds) - 1)
    optimal_threshold = pr_thresholds[threshold_idx]
    optimal_preds = (y_prob > optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(y_true, optimal_preds)
    optimal_cm = confusion_matrix(y_true, optimal_preds)
    optimal_mcc = safe_mcc(optimal_cm)
    optimal_metrics = {
        'threshold': f'{optimal_threshold:.4f}',
        'accuracy': f'{optimal_accuracy:.4f}',
        'precision': f'{precision_curve[best_idx]:.4f}',
        'recall': f'{recall_curve[best_idx]:.4f}',
        'f1': f'{f1_scores[best_idx]:.4f}',
        'mcc': f'{optimal_mcc:.4f}',
        'roc_auc': f'{roc_auc:.4f}',
        'pr_auc': f'{pr_auc:.4f}'
    }

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'ap': ap,
        'confusion_matrix': cm,
        'optimal_metrics': optimal_metrics
    }


def plot_metrics(metrics_history, save_path='metrics_plots'):
    """绘制训练过程中的指标变化"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    if 'val_loss' in metrics_history:
        plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'ap']:
        plt.plot(metrics_history[f'train_{metric}'], label=f'Train {metric.upper()}')
        if f'val_{metric}' in metrics_history:
            plt.plot(metrics_history[f'val_{metric}'], label=f'Val {metric.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.savefig(f'{save_path}/metrics_curve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_history['final_confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()


def create_model(device, input_size=1280, hidden_size=256, dropout_rate=0.50, num_heads=4,
                 num_experts=4, top_k=2, num_clusters=4):
    """创建并返回模型实例

    Args:
        device:       运行设备
        input_size:   输入特征维度 (ESM2/SaProt 均为 1280)
        hidden_size:  隐藏层维度
        dropout_rate: Dropout 比率
        num_heads:    Attention 头数
        num_experts:  MoE 专家数量（残基级融合模式路由）
        top_k:        每个残基选择的专家数
        num_clusters: 聚类池化的簇数量（线性/构象/复合表位）
    """
    model = ProteinClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        num_clusters=num_clusters,
    ).to(device)

    return model
