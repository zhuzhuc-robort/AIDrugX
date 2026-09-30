"""
predict.py - 通用蛋白-小分子结合预测脚本

================================================================================
快速开始
================================================================================

1. 准备 input.csv，至少包含两列（列名区分大小写）：
       smiles,sequence
       CC(=O)Oc1ccccc1C(=O)O,MKTAYSD...
       c1ccccc1,MEEPQSDF...

2. 修改下面的【★ 必须修改 ★】路径，指向你自己的权重文件

3. 运行:
       python predict.py --input input.csv --output predictions.csv

   默认会输出:
       predictions.csv                        主结果表
       predictions.residue_scores.csv         逐残基分数（一行一个残基）
       predictions.residue_scores.npz         逐残基分数（numpy 压缩格式）
       predictions_plots/                     每个 (SMILES, 序列) 对一张热图

   如果不想生成热图/残基分数：
       python predict.py --input input.csv --no-plot --no-save-residue-scores

================================================================================
【★ 必须修改 ★】权重与依赖路径
================================================================================
运行前请确认以下 4 个文件存在：

    (1) 模型权重 (v5 残基级)   weights/best_model.pth
    (2) MolFormer 词表         LM_Mol/bert_vocab.txt
    (3) MolFormer checkpoint   LM_Mol/pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt
    (4) ESM2-650M              首次运行自动从 HuggingFace 下载

================================================================================
输出说明
================================================================================
predictions.csv 保留 input.csv 的所有列，额外增加:
    - prot_len       : 蛋白序列长度
    - score          : 综合打分（默认 top-20 残基平均）
    - top1_score     : 最高残基分
    - top5_mean      : top-5 残基平均
    - top20_mean     : top-20 残基平均
    - top50_mean     : top-50 残基平均
    - n_pos_residues : 预测概率 > 0.5 的残基数

predictions.residue_scores.csv 每个 (SMILES, 序列) 对的逐残基分数:
    row_id, smiles, sequence_len, position, residue, score
    其中 residue 是序列中该位置的氨基酸字符（如 'M', 'K'）

predictions_plots/ 每个 pair 一张 PNG:
    上图: 残基分数曲线 + top-k 高亮 + 阈值线
    下方: 累积贡献曲线（可选）
"""

import os
import sys
import time
import argparse
import warnings
import hashlib
from pathlib import Path
from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★                                                                              ★
# ★                       【★ 必须修改 ★】 用 户 路 径 配 置                    ★
# ★                                                                              ★
# ★   请把下面的路径改成你自己机器上实际的路径！                                   ★
# ★                                                                              ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()

# ---- (1) v5 残基级模型权重 ----
MODEL_PATH = str(SCRIPT_DIR / "trained_model_v5_residue" / "best_model.pth")

# ---- (2) LM_Mol 目录（包含 tokenizer.py, rotate_builder.py 等） ----
LM_MOL_DIR = str(SCRIPT_DIR / "LM_Mol")

# ---- (3) MolFormer 词表与 checkpoint ----
VOCAB_PATH      = str(Path(LM_MOL_DIR) / "bert_vocab.txt")
CHECKPOINT_PATH = str(Path(LM_MOL_DIR) / "pretrained" / "checkpoints"
                      / "N-Step-Checkpoint_3_30000.ckpt")

# ---- (4) HuggingFace 缓存目录 ----
HF_CACHE = str(Path.home() / ".cache" / "huggingface")

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

# ============================================================================
# 推理超参（通常不需要改）
# ============================================================================
ESM_CHUNK     = 512
ESM_OVERLAP   = 128
ESM_MAX_LEN   = 2000

LIG_BATCH     = 32
INFER_BATCH   = 32
NUM_WORKERS   = 0

AGG_MODE      = "topk_mean"
TOPK          = 20

PROT_ESM_DIM   = 1280
LIG_MOL_DIM    = 768
FUSION_DIM     = 128
CNN_BASE_CHAN  = 16
CNN_KERNEL     = 7
GNN_HID_DIM    = 256
GNN_NUM_LAYERS = 3
GAT_NUM_HEADS  = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 依赖检查
# ============================================================================
def _check_files():
    missing = []
    if not Path(MODEL_PATH).exists():
        missing.append(f"  [模型权重]      {MODEL_PATH}")
    if not Path(VOCAB_PATH).exists():
        missing.append(f"  [MolFormer 词表] {VOCAB_PATH}")
    if not Path(CHECKPOINT_PATH).exists():
        missing.append(f"  [MolFormer 权重] {CHECKPOINT_PATH}")
    if not Path(LM_MOL_DIR).exists():
        missing.append(f"  [LM_Mol 目录]   {LM_MOL_DIR}")

    if missing:
        msg = "\n[ERROR] 以下路径不存在，请修改 predict.py 顶部的【★ 必须修改 ★】路径:\n"
        msg += "\n".join(missing) + "\n"
        print(msg)
        sys.exit(1)


# ============================================================================
# ESM2 编码器
# ============================================================================
class ESM2Encoder:
    def __init__(self, model_name=ESM2_MODEL, cache_dir=HF_CACHE, device=DEVICE):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoModel, AutoTokenizer
        print(f"[ESM2] 加载 {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _encode_one(self, seq):
        inputs = self.tokenizer(
            seq, return_tensors="pt", padding=False,
            truncation=False, add_special_tokens=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        hidden = out.last_hidden_state[0]
        L = len(seq)
        return hidden[1:L+1].half().cpu()

    @torch.no_grad()
    def encode_sequence(self, seq):
        L = len(seq)
        if L <= ESM_CHUNK:
            return self._encode_one(seq)

        stride = ESM_CHUNK - ESM_OVERLAP
        out = torch.zeros(L, PROT_ESM_DIM, dtype=torch.float16)
        w   = torch.zeros(L, dtype=torch.float16)

        for start in range(0, L, stride):
            end = min(start + ESM_CHUNK, L)
            emb = self._encode_one(seq[start:end])
            out[start:end] += emb
            w[start:end]   += 1.0
            if end == L:
                break

        w = w.clamp(min=1.0).unsqueeze(-1)
        return (out.float() / w).half()


# ============================================================================
# MolFormer 编码器
# ============================================================================
class MolFormerEncoder:
    def __init__(self, lm_mol_dir=LM_MOL_DIR, vocab_path=VOCAB_PATH,
                 ckpt_path=CHECKPOINT_PATH, device=DEVICE):
        if lm_mol_dir not in sys.path:
            sys.path.insert(0, lm_mol_dir)
        if str(Path(lm_mol_dir).parent) not in sys.path:
            sys.path.insert(0, str(Path(lm_mol_dir).parent))

        from LM_Mol.tokenizer import MolTranBertTokenizer
        from LM_Mol.rotate_builder import RotateEncoderBuilder as rotate_builder
        from fast_transformers.feature_maps import GeneralizedRandomFeatures
        from fast_transformers.masking import LengthMask as LM

        self.LM = LM
        self.device = device
        self.tokenizer = MolTranBertTokenizer(vocab_path)
        n_vocab = len(self.tokenizer.vocab)

        print(f"[MolFormer] 加载 tokenizer (vocab={n_vocab}) ...")
        self.tok_emb = nn.Embedding(n_vocab, 768)
        builder = rotate_builder.from_kwargs(
            n_layers=12, n_heads=12,
            query_dimensions=768 // 12, value_dimensions=768 // 12,
            feed_forward_dimensions=768, attention_type="linear",
            feature_map=partial(GeneralizedRandomFeatures, n_dims=32),
            activation="gelu",
        )
        self.blocks = builder.get()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        for name, _ in self.blocks.state_dict().items():
            if "blocks." + name in ckpt:
                self.blocks.state_dict()[name].copy_(ckpt["blocks." + name])
        for name, _ in self.tok_emb.state_dict().items():
            if "tok_emb." + name in ckpt:
                self.tok_emb.state_dict()[name].copy_(ckpt["tok_emb." + name])
        for p in list(self.tok_emb.parameters()) + list(self.blocks.parameters()):
            p.requires_grad = False
        self.tok_emb.to(device).eval()
        self.blocks.to(device).eval()
        print("[MolFormer] 加载完成")

    def _encode_one(self, smi, max_len=256):
        try:
            ids = self.tokenizer.encode(smi)
        except Exception:
            return []
        if not isinstance(ids, list):
            ids = list(ids)
        return ids[:max_len]

    @torch.no_grad()
    def encode(self, smiles_list, batch_size=512):
        results = {}
        n = (len(smiles_list) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(smiles_list), batch_size),
                      total=n, desc="[MolFormer]", leave=False):
            batch = smiles_list[i:i+batch_size]
            token_lists = [self._encode_one(s) for s in batch]
            valid_idx = [j for j, t in enumerate(token_lists) if len(t) > 0]
            if not valid_idx:
                for s in batch:
                    results[s] = torch.zeros(1, 768, dtype=torch.float16)
                continue

            max_L = max(len(token_lists[j]) for j in valid_idx)
            B = len(batch)
            padded = torch.zeros(B, max_L, dtype=torch.long)
            mask = torch.zeros(B, max_L, dtype=torch.long)
            for j in valid_idx:
                t = token_lists[j]
                padded[j, :len(t)] = torch.tensor(t, dtype=torch.long)
                mask[j, :len(t)] = 1

            padded = padded.to(self.device)
            mask = mask.to(self.device)
            emb = self.tok_emb(padded)
            emb = self.blocks(emb, length_mask=self.LM(mask.sum(-1)))
            emb = emb.half().cpu()

            for j, s in enumerate(batch):
                L = int(mask[j].sum().item())
                if L == 0:
                    results[s] = torch.zeros(1, 768, dtype=torch.float16)
                else:
                    results[s] = emb[j, :L].clone()

            del padded, mask, emb
        return results


# ============================================================================
# 分子图
# ============================================================================
def build_graphs(smiles_list):
    from dgllife.utils import (
        smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
    )
    atom_feat = CanonicalAtomFeaturizer()
    bond_feat = CanonicalBondFeaturizer(self_loop=True)
    graphs = {}
    n_fail = 0
    for smi in tqdm(smiles_list, desc="[Graphs]", leave=False):
        try:
            g = smiles_to_bigraph(
                smi, add_self_loop=True,
                node_featurizer=atom_feat, edge_featurizer=bond_feat,
            )
            if g is None or g.num_nodes() == 0:
                n_fail += 1
                continue
            graphs[smi] = g
        except Exception:
            n_fail += 1
    return graphs, n_fail


# ============================================================================
# 模型定义
# ============================================================================
from dgl import batch as dgl_batch
from dgllife.model.gnn.gat import GAT
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim, self.n_heads = hid_dim, n_heads
        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        orig_dim = query.dim()
        if orig_dim == 3:
            B, L, D = query.shape
            query = query.reshape(B * L, D)
            key   = key.reshape(B * L, D)
            value = value.reshape(B * L, D)
        else:
            B, D = query.shape
            L = 1
        scale = self.scale.to(query.device)
        Q = self.f_q(query).view(B*L, self.n_heads, D // self.n_heads).unsqueeze(3)
        K = self.f_k(key).view(B*L, self.n_heads, D // self.n_heads).unsqueeze(3)
        K_T = K.transpose(2, 3)
        V = self.f_v(value).view(B*L, self.n_heads, D // self.n_heads).unsqueeze(3)
        energy = torch.matmul(Q, K_T) / scale
        energy = energy.clamp(-50.0, 50.0)
        attn = self.do(F.softmax(energy, dim=-1))
        wm = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous()
        wm = wm.view(B*L, self.n_heads * (self.hid_dim // self.n_heads))
        out = self.do(self.fc(wm))
        if orig_dim == 3:
            out = out.view(B, L, D)
        return out


class ProteinESMBranch(nn.Module):
    def __init__(self, in_dim=PROT_ESM_DIM, out_dim=FUSION_DIM):
        super().__init__()
        self.mlp_l1 = nn.Linear(in_dim, 512)
        self.mlp_l2 = nn.Linear(512, 128)
        self.mlp_l3 = nn.Linear(128, 128)
        self.mlp_l4 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.dropout(self.mlp_l1(x), 0.5, training=self.training)
        x = F.dropout(self.mlp_l2(x), 0.5, training=self.training)
        x = F.dropout(self.mlp_l3(x), 0.5, training=self.training)
        x = F.dropout(self.mlp_l4(x), 0.5, training=self.training)
        return x


class ProteinSeqCNN(nn.Module):
    def __init__(self, n_features=20, base_channel=CNN_BASE_CHAN,
                 kernal=CNN_KERNEL, out_dim=FUSION_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, base_channel, kernal, padding=kernal // 2)
        self.bn1   = nn.BatchNorm1d(base_channel)
        self.conv2 = nn.Conv1d(base_channel, base_channel * 4, kernal, padding=kernal // 2)
        self.bn2   = nn.BatchNorm1d(base_channel * 4)
        self.conv3 = nn.Conv1d(base_channel * 4, base_channel * 8, kernal, padding=kernal // 2)
        self.bn3   = nn.BatchNorm1d(base_channel * 8)
        self.shortcut = nn.Conv1d(n_features, base_channel * 8, 1)
        self.fct   = nn.Linear(base_channel * 8, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        identity = self.shortcut(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        h = self.bn3(self.conv3(h))
        out = F.relu(h + identity)
        out = out.transpose(1, 2)
        out = self.fct(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class LigandMolFormerBranch(nn.Module):
    def __init__(self, in_dim=LIG_MOL_DIM, mid_dim=128,
                 pooled_len=128, out_dim=FUSION_DIM):
        super().__init__()
        self.mlp_d1 = nn.Linear(in_dim, mid_dim)
        self.pooled_len = pooled_len
        self.mlp_d2 = nn.Linear(pooled_len * mid_dim, out_dim)

    def forward(self, x, mask):
        x = F.dropout(self.mlp_d1(x), 0.5, training=self.training)
        mask_exp = mask.unsqueeze(-1)
        x = x * mask_exp + (1.0 - mask_exp) * (-1e4)
        x = x.transpose(1, 2)
        x = F.adaptive_max_pool1d(x, self.pooled_len)
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = F.dropout(self.mlp_d2(x), 0.5, training=self.training)
        return x


class LigandGATBranch(nn.Module):
    def __init__(self, in_feats=74, hid=GNN_HID_DIM, n_layers=GNN_NUM_LAYERS,
                 n_heads=GAT_NUM_HEADS, out_dim=FUSION_DIM):
        super().__init__()
        self.gnn = GAT(
            in_feats=in_feats, hidden_feats=[hid] * n_layers,
            num_heads=[n_heads] * n_layers,
            feat_drops=[0.5] * n_layers, attn_drops=[0.5] * n_layers,
            alphas=[0.2] * n_layers, residuals=[True] * n_layers,
            agg_modes=["flatten"] * n_layers, activations=[F.relu] * n_layers,
            biases=[None] * n_layers,
        )
        gnn_out = hid * n_heads
        self.readout = WeightedSumAndMax(gnn_out)
        self.transform = nn.Linear(gnn_out * 2, out_dim)

    def forward(self, g, feats):
        return self.transform(self.readout(g, self.gnn(g, feats)))


class ProteinLigandNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = FUSION_DIM
        self.prot_esm_branch = ProteinESMBranch(out_dim=d)
        self.prot_cnn_branch = ProteinSeqCNN(out_dim=d)
        self.lig_mol_branch  = LigandMolFormerBranch(out_dim=d)
        self.lig_gat_branch  = LigandGATBranch(out_dim=d)

        self.LN1 = nn.LayerNorm(d)
        self.LN2 = nn.LayerNorm(d)
        self.LN3 = nn.LayerNorm(d)
        self.LN4 = nn.LayerNorm(d)
        self.LN5 = nn.LayerNorm(d)
        self.LN6 = nn.LayerNorm(d)
        self.LN7 = nn.LayerNorm(d)
        self.LN8 = nn.LayerNorm(d)
        self.LN9  = nn.LayerNorm(d * 2)
        self.LN10 = nn.LayerNorm(d * 2)
        self.LN11 = nn.LayerNorm(d * 4)

        self.attentionBlock1 = AttentionBlock(d, 2, 0.5)
        self.attentionBlock2 = AttentionBlock(d, 2, 0.5)
        self.attentionBlock3 = AttentionBlock(d, 2, 0.5)
        self.attentionBlock4 = AttentionBlock(d, 2, 0.5)
        self.attentionBlock5 = AttentionBlock(d * 4, 8, 0.6)
        self.attentionBlock6 = AttentionBlock(d * 2, 2, 0.5)
        self.attentionBlock7 = AttentionBlock(d * 2, 2, 0.5)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.ModuleList([
            nn.Linear(d * 4, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        ])

    def forward(self, prot_esm, prot_onehot, prot_mask,
                lig_molformer, lig_mask, lig_graph, lig_node_feats):
        v_Pe = self.prot_esm_branch(prot_esm)
        v_P  = self.prot_cnn_branch(prot_onehot, mask=prot_mask)
        v_De = self.lig_mol_branch(lig_molformer, lig_mask)
        v_D  = self.lig_gat_branch(lig_graph, lig_node_feats)

        L = v_Pe.shape[1]
        v_De = v_De.unsqueeze(1).expand(-1, L, -1)
        v_D  = v_D.unsqueeze(1).expand(-1, L, -1)

        v_Pe, v_De, v_P, v_D = (
            self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
        )
        CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
        CAB_l1 = v_P  + self.attentionBlock2(v_P,  v_D,  v_D)
        CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
        CAB_l2 = v_D  + self.attentionBlock4(v_D,  v_P,  v_P)
        CAB_g1, CAB_g2 = self.LN5(CAB_g1), self.LN6(CAB_g2)
        CAB_l1, CAB_l2 = self.LN7(CAB_l1), self.LN8(CAB_l2)

        v_f1 = torch.cat((CAB_g1, CAB_g2), dim=-1)
        v_f2 = torch.cat((CAB_l1, CAB_l2), dim=-1)
        v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
        v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)
        v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)
        v_f = torch.cat((v_f1n, v_f2n), dim=-1)
        v_f = self.LN11(v_f)
        v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)

        for i, layer in enumerate(self.fc):
            if i == len(self.fc) - 1:
                v_f = layer(v_f)
            else:
                v_f = F.relu(self.dropout(layer(v_f)))
        return v_f.squeeze(-1)


# ============================================================================
# 工具
# ============================================================================
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def seq_to_onehot(seq, num_aa=20):
    L = len(seq)
    arr = np.zeros((L, num_aa), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            arr[i, AA_TO_IDX[aa]] = 1.0
    return arr


def aggregate_residue_scores(scores, mode=AGG_MODE, topk=TOPK):
    if len(scores) == 0:
        return 0.0
    if mode == "max":
        return float(scores.max())
    elif mode == "topk_mean":
        kk = min(topk, len(scores))
        top_idx = np.argpartition(-scores, kk - 1)[:kk]
        return float(scores[top_idx].mean())
    elif mode == "smrtnet":
        thr, min_run = 0.5, 3
        high = scores > thr
        runs = []
        i = 0
        while i < len(high):
            if high[i]:
                j = i
                while j < len(high) and high[j]:
                    j += 1
                if j - i >= min_run:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            return float(scores.min())
        return max(float(scores[s:e].mean()) for s, e in runs)
    else:
        raise ValueError(f"Unknown agg mode: {mode}")


# ============================================================================
# 逐残基分数保存
# ============================================================================
def save_residue_scores_to_csv(df, residue_scores_dict, out_csv,
                                smiles_col="smiles", seq_col="sequence"):
    """
    把逐残基分数保存为 CSV，每行一条 (pair, position, residue, score)。
    residue_scores_dict 的 key 是 row_id (int)。
    """
    rows = []
    for idx, row in df.iterrows():
        if idx not in residue_scores_dict:
            continue
        seq = row[seq_col]
        smi = row[smiles_col]
        scores = residue_scores_dict[idx]
        L = min(len(seq), len(scores))
        for pos in range(L):
            rows.append({
                "row_id":       int(idx),
                "smiles":       smi,
                "sequence_len": L,
                "position":     pos,
                "residue":      seq[pos] if pos < len(seq) else "",
                "score":        float(scores[pos]),
            })
    if not rows:
        print(f"  [WARN] 没有残基分数可保存")
        return 0
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return len(df_out)


# ============================================================================
# 热图绘制（可选）
# ============================================================================
def plot_residue_score_heatmap(seq, smiles, scores,
                                row_id, score_agg, save_path,
                                top_k=20, threshold=0.5,
                                smiles_display_len=50):
    """
    单页热图，包含：
      上图：残基分数曲线 + top-k 高亮 + 阈值线 + 累积贡献
      下图：累积贡献曲线
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(scores)
    positions = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # ---------- 上图：残基分数 ----------
    ax = axes[0]
    ax.fill_between(positions, scores, alpha=0.3, color="steelblue")
    ax.plot(positions, scores, color="steelblue", linewidth=1.0)

    # 阈值线
    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.8, label=f"threshold = {threshold}")

    # top-k 高亮
    if top_k > 0 and L > 0:
        kk = min(top_k, L)
        top_idx = np.argpartition(-scores, kk - 1)[:kk]
        ax.scatter(top_idx, scores[top_idx],
                   color="red", s=30, zorder=5, alpha=0.9,
                   label=f"top-{kk}")

    # 标注最高残基
    top1_idx = int(np.argmax(scores))
    ax.annotate(
        f"top1={scores[top1_idx]:.3f}\npos={top1_idx} ({seq[top1_idx] if top1_idx < len(seq) else '?'})",
        xy=(top1_idx, scores[top1_idx]),
        xytext=(top1_idx, min(1.0, scores[top1_idx] + 0.15)),
        fontsize=9, color="red", ha="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
    )

    smi_show = smiles if len(smiles) <= smiles_display_len \
        else smiles[:smiles_display_len] + "..."
    ax.set_title(
        f"row #{row_id} | SMILES: {smi_show}\n"
        f"len={L} | aggregate score={score_agg:.4f} | "
        f"top1={scores[top1_idx]:.4f} | "
        f"n_pos(>{threshold})={int((scores > threshold).sum())}",
        fontsize=10,
    )
    ax.set_ylabel("Predicted binding score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ---------- 下图：累积贡献 ----------
    ax2 = axes[1]
    sorted_scores = np.sort(scores)[::-1]
    cumsum = np.cumsum(sorted_scores) / (sorted_scores.sum() + 1e-8)
    ax2.plot(np.arange(1, len(cumsum) + 1), cumsum,
             color="darkorange", linewidth=1.2)
    ax2.axvline(min(top_k, L), color="red", linestyle="--",
                linewidth=1.0, alpha=0.7, label=f"top-{top_k}")
    ax2.set_xlabel("Residue rank (descending by score)", fontsize=11)
    ax2.set_ylabel("Cumulative\ncontribution", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_plot_filename(row_id, smiles, seq, max_hash_len=8):
    """生成安全的热图文件名。"""
    h = hashlib.md5(smiles.encode()).hexdigest()[:max_hash_len]
    return f"row{row_id:05d}_{h}.png"


# ============================================================================
# 配体 Dataset
# ============================================================================
class LigandOnlyDataset(Dataset):
    def __init__(self, lig_embs, lig_masks, lig_graphs):
        self.lig_embs = lig_embs
        self.lig_masks = lig_masks
        self.lig_graphs = lig_graphs

    def __len__(self):
        return len(self.lig_embs)

    def __getitem__(self, idx):
        return {
            "idx":           idx,
            "lig_molformer": self.lig_embs[idx],
            "lig_mask":      self.lig_masks[idx],
            "lig_graph":     self.lig_graphs[idx],
        }


def make_collate(prot_esm, prot_onehot, prot_mask):
    def collate_fn(batch):
        B = len(batch)
        max_L = max(b["lig_molformer"].shape[0] for b in batch)
        lig_molformer = torch.zeros(B, max_L, LIG_MOL_DIM)
        lig_mask = torch.zeros(B, max_L)
        for i, b in enumerate(batch):
            L = b["lig_molformer"].shape[0]
            lig_molformer[i, :L] = b["lig_molformer"]
            lig_mask[i, :L] = b["lig_mask"]

        graphs = [b["lig_graph"] for b in batch]
        batched = dgl_batch(graphs)
        idxs = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

        return {
            "idx": idxs,
            "prot_esm": prot_esm,
            "prot_onehot": prot_onehot,
            "prot_mask": prot_mask,
            "lig_molformer": lig_molformer,
            "lig_mask": lig_mask,
            "lig_graph": batched,
        }
    return collate_fn


# ============================================================================
# 主推理函数
# ============================================================================
@torch.no_grad()
def predict_pairs(df, model, esm_encoder, mol_encoder,
                  save_residue_scores=True):
    """
    输入: df 至少包含 smiles, sequence 两列
    输出: df_out (添加预测列), residue_scores_dict
          residue_scores_dict: {row_id: np.ndarray(L,)}
    """
    # ---------- 收集唯一蛋白 ----------
    unique_seqs = df["sequence"].dropna().unique().tolist()
    print(f"\n[准备] 唯一蛋白序列: {len(unique_seqs)}，"
          f"唯一 SMILES: {df['smiles'].nunique()}")

    seqs_to_encode = []
    for s in unique_seqs:
        if len(s) > ESM_MAX_LEN:
            print(f"  [WARN] 序列长度 {len(s)} > {ESM_MAX_LEN}，将被截断")
            seqs_to_encode.append(s[:ESM_MAX_LEN])
        else:
            seqs_to_encode.append(s)

    # ---------- ESM2 编码 ----------
    print(f"\n[Step 1/4] ESM2 编码 {len(unique_seqs)} 条蛋白序列...")
    t0 = time.time()
    prot_esm_cache = {}
    for i, (s_full, s_trunc) in enumerate(zip(unique_seqs, seqs_to_encode)):
        if s_full in prot_esm_cache:
            continue
        emb = esm_encoder.encode_sequence(s_trunc)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        prot_esm_cache[s_full] = emb
        if (i + 1) % max(1, len(unique_seqs) // 10) == 0:
            print(f"  [{i+1}/{len(unique_seqs)}]  已耗时 {time.time()-t0:.1f}s")
    print(f"  [Step 1/4] 完成，耗时 {time.time()-t0:.1f}s")

    # ---------- MolFormer 编码 ----------
    unique_smiles = df["smiles"].dropna().unique().tolist()
    print(f"\n[Step 2/4] MolFormer 编码 {len(unique_smiles)} 个 SMILES...")
    t0 = time.time()
    lig_emb_dict = mol_encoder.encode(unique_smiles)
    print(f"  [Step 2/4] 完成，耗时 {time.time()-t0:.1f}s")

    # ---------- 分子图 ----------
    print(f"\n[Step 3/4] 构建分子图...")
    t0 = time.time()
    smiles_to_graph, n_fail = build_graphs(unique_smiles)
    print(f"  [Step 3/4] 完成，成功 {len(smiles_to_graph)}，"
          f"失败 {n_fail}，耗时 {time.time()-t0:.1f}s")

    # ---------- 过滤无图 SMILES ----------
    valid_mask = df["smiles"].isin(smiles_to_graph)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"  [WARN] {n_dropped} 行 SMILES 无法构建图，将被丢弃")
    df_valid = df[valid_mask].reset_index(drop=True)

    # ---------- 逐个蛋白推理 ----------
    print(f"\n[Step 4/4] 逐个蛋白推理...")
    t0 = time.time()

    all_scores        = np.full(len(df), np.nan, dtype=np.float32)
    all_top1          = np.full(len(df), np.nan, dtype=np.float32)
    all_top5          = np.full(len(df), np.nan, dtype=np.float32)
    all_top20         = np.full(len(df), np.nan, dtype=np.float32)
    all_top50         = np.full(len(df), np.nan, dtype=np.float32)
    all_npos          = np.full(len(df), -1, dtype=np.int32)
    residue_scores_dict = {} if save_residue_scores else None

    for seq in tqdm(unique_seqs, desc="Proteins"):
        prot_esm = prot_esm_cache[seq]
        L = prot_esm.shape[0]
        prot_esm_t = prot_esm.float().unsqueeze(0)
        prot_onehot_t = torch.from_numpy(
            seq_to_onehot(seq[:L])
        ).unsqueeze(0)
        prot_mask_t = torch.ones(1, L, dtype=torch.float32)

        rows = df_valid[df_valid["sequence"] == seq]
        if len(rows) == 0:
            continue
        # 用 df_valid 的 index 保持和 all_scores 的对应
        local_indices = df_valid.index[df_valid["sequence"] == seq].tolist()

        smiles_local = rows["smiles"].tolist()

        lig_embs, lig_masks, lig_graphs = [], [], []
        for smi in smiles_local:
            le = lig_emb_dict.get(smi)
            if le is None:
                le = torch.zeros(1, LIG_MOL_DIM, dtype=torch.float16)
            le = le.float()
            lig_embs.append(le)
            lig_masks.append(torch.ones(le.shape[0]))
            lig_graphs.append(smiles_to_graph[smi])

        ds = LigandOnlyDataset(lig_embs, lig_masks, lig_graphs)
        dl = DataLoader(
            ds, batch_size=INFER_BATCH, shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=make_collate(prot_esm_t, prot_onehot_t, prot_mask_t),
            pin_memory=True,
            persistent_workers=False,
        )

        with torch.no_grad():
            for batch in dl:
                idxs = batch["idx"].numpy()
                B = len(idxs)

                prot_esm_b = batch["prot_esm"].expand(B, -1, -1).to(DEVICE, non_blocking=True)
                prot_onehot_b = batch["prot_onehot"].expand(B, -1, -1).to(DEVICE, non_blocking=True)
                prot_mask_b = batch["prot_mask"].expand(B, -1).to(DEVICE, non_blocking=True)
                lig_mol = batch["lig_molformer"].to(DEVICE, non_blocking=True)
                lig_mask = batch["lig_mask"].to(DEVICE, non_blocking=True)
                lig_graph = batch["lig_graph"].to(DEVICE)

                logits = model(
                    prot_esm_b, prot_onehot_b, prot_mask_b,
                    lig_mol, lig_mask,
                    lig_graph, lig_graph.ndata["h"],
                )
                probs = torch.sigmoid(logits.float()).cpu().numpy()

                for i, local_i in enumerate(idxs):
                    gidx = local_indices[local_i]
                    s = probs[i]
                    all_scores[gidx] = aggregate_residue_scores(s)
                    all_top1[gidx]   = float(s.max())
                    kk5 = min(5, len(s))
                    all_top5[gidx]   = float(np.sort(s)[-kk5:].mean())
                    kk20 = min(20, len(s))
                    all_top20[gidx]  = float(np.sort(s)[-kk20:].mean())
                    kk50 = min(50, len(s))
                    all_top50[gidx]  = float(np.sort(s)[-kk50:].mean())
                    all_npos[gidx]   = int((s > 0.5).sum())
                    if residue_scores_dict is not None:
                        residue_scores_dict[gidx] = s.astype(np.float32)

                del prot_esm_b, prot_onehot_b, prot_mask_b
                del lig_mol, lig_mask, lig_graph, logits, probs

        del ds, dl, lig_embs, lig_masks, lig_graphs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [Step 4/4] 完成，总耗时 {time.time()-t0:.1f}s")

    # ---------- 组装输出 ----------
    df_out = df.copy()
    df_out["score"]          = all_scores
    df_out["top1_score"]     = all_top1
    df_out["top5_mean"]      = all_top5
    df_out["top20_mean"]     = all_top20
    df_out["top50_mean"]     = all_top50
    df_out["n_pos_residues"] = all_npos
    df_out["prot_len"]       = df_out["sequence"].str.len()

    return df_out, residue_scores_dict


# ============================================================================
# main
# ============================================================================
def main():
    # 覆盖全局参数
    global INFER_BATCH, DEVICE
    parser = argparse.ArgumentParser(
        description="通用蛋白-小分子结合预测",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="输入 CSV 路径，至少含 smiles,sequence 两列")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="输出 CSV 路径 (默认: predictions.csv)")
    parser.add_argument("--smiles-col",   type=str, default="smiles",
                        help="SMILES 列名 (默认: smiles)")
    parser.add_argument("--sequence-col", type=str, default="sequence",
                        help="蛋白序列列名 (默认: sequence)")
    parser.add_argument("--batch-size",   type=int, default=None,
                        help=f"推理 batch size (默认: {INFER_BATCH})")
    parser.add_argument("--device",       type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="强制指定设备 (默认: 自动检测)")

    # ---- 新增参数 ----
    parser.add_argument("--save-residue-scores", dest="save_residue_scores",
                        action="store_true", default=True,
                        help="保存逐残基分数 (默认: 开启)")
    parser.add_argument("--no-save-residue-scores", dest="save_residue_scores",
                        action="store_false",
                        help="关闭逐残基分数保存")

    parser.add_argument("--plot", dest="plot",
                        action="store_true", default=True,
                        help="绘制热图 (默认: 开启)")
    parser.add_argument("--no-plot", dest="plot",
                        action="store_false",
                        help="关闭热图绘制")

    parser.add_argument("--plot-dir", type=str, default=None,
                        help="热图输出目录 (默认: <output>_plots/)")
    parser.add_argument("--plot-top-k", type=int, default=20,
                        help="热图中高亮的 top-k 残基 (默认: 20)")
    parser.add_argument("--plot-threshold", type=float, default=0.5,
                        help="热图中的阈值线 (默认: 0.5)")
    parser.add_argument("--plot-max", type=int, default=None,
                        help="最多画多少张热图 (默认: 全部)")
    args = parser.parse_args()


    if args.batch_size is not None:
        INFER_BATCH = args.batch_size
    if args.device is not None:
        DEVICE = torch.device(args.device)

    # ---------- 依赖检查 ----------
    print("=" * 70)
    print("  通用蛋白-小分子结合预测")
    print("=" * 70)
    print(f"[Config] MODEL_PATH          = {MODEL_PATH}")
    print(f"[Config] LM_MOL_DIR          = {LM_MOL_DIR}")
    print(f"[Config] HF_CACHE            = {HF_CACHE}")
    print(f"[Config] DEVICE              = {DEVICE}")
    print(f"[Config] AGG_MODE            = {AGG_MODE} (k={TOPK})")
    print(f"[Config] INFER_BATCH         = {INFER_BATCH}")
    print(f"[Config] save_residue_scores = {args.save_residue_scores}")
    print(f"[Config] plot                = {args.plot}")
    _check_files()

    # ---------- 加载 input ----------
    if not Path(args.input).exists():
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"\n[Load] 读取 {len(df):,} 行")

    if args.smiles_col not in df.columns:
        print(f"[ERROR] 缺少列 '{args.smiles_col}'。现有列: {list(df.columns)}")
        print(f"  如果列名不同，可用 --smiles-col 指定")
        sys.exit(1)
    if args.sequence_col not in df.columns:
        print(f"[ERROR] 缺少列 '{args.sequence_col}'。现有列: {list(df.columns)}")
        print(f"  如果列名不同，可用 --sequence-col 指定")
        sys.exit(1)

    if args.smiles_col != "smiles" or args.sequence_col != "sequence":
        df = df.rename(columns={
            args.smiles_col: "smiles",
            args.sequence_col: "sequence",
        })

    df = df.dropna(subset=["smiles", "sequence"]).reset_index(drop=True)
    print(f"[Load] 有效行: {len(df):,}")

    # ---------- 加载编码器与模型 ----------
    print("\n[Init] 加载 ESM2 ...")
    esm_encoder = ESM2Encoder()

    print("\n[Init] 加载 MolFormer ...")
    mol_encoder = MolFormerEncoder()

    print("\n[Init] 加载 v5 残基级模型 ...")
    model = ProteinLigandNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    model.eval()

    # ---------- 推理 ----------
    df_out, residue_scores = predict_pairs(
        df, model, esm_encoder, mol_encoder,
        save_residue_scores=args.save_residue_scores or args.plot,
    )

    # ---------- 保存主结果 ----------
    df_out.to_csv(args.output, index=False)
    print(f"\n[Save] 主结果已保存: {args.output}")

    out_path = Path(args.output)
    out_stem = out_path.with_suffix("")

    # ---------- 保存逐残基分数 ----------
    if args.save_residue_scores and residue_scores:
        # CSV
        csv_path = f"{out_stem}.residue_scores.csv"
        n_rows = save_residue_scores_to_csv(df_out, residue_scores, csv_path)
        print(f"[Save] 逐残基分数 CSV: {csv_path}  ({n_rows:,} 行)")

        # NPZ
        npz_path = f"{out_stem}.residue_scores.npz"
        np.savez_compressed(
            npz_path,
            **{f"row_{k}": v for k, v in residue_scores.items()}
        )
        print(f"[Save] 逐残基分数 NPZ: {npz_path}")

    # ---------- 绘制热图 ----------
    if args.plot and residue_scores:
        plot_dir = args.plot_dir if args.plot_dir else f"{out_stem}_plots"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n[Plot] 绘制热图到 {plot_dir}/ ...")

        n_plotted = 0
        for idx, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Plot"):
            if idx not in residue_scores:
                continue
            if args.plot_max is not None and n_plotted >= args.plot_max:
                break

            seq = row["sequence"]
            smi = row["smiles"]
            scores = residue_scores[idx]
            score_agg = float(row["score"])
            fname = make_plot_filename(idx, smi, seq)
            save_path = Path(plot_dir) / fname

            try:
                plot_residue_score_heatmap(
                    seq=seq, smiles=smi, scores=scores,
                    row_id=int(idx), score_agg=score_agg,
                    save_path=save_path,
                    top_k=args.plot_top_k,
                    threshold=args.plot_threshold,
                )
                n_plotted += 1
            except Exception as e:
                print(f"  [WARN] row {idx} 画图失败: {e}")

        print(f"[Plot] 完成，共 {n_plotted} 张 -> {plot_dir}/")

    # ---------- 简要汇总 ----------
    print("\n" + "=" * 70)
    print("  预测完成")
    print("=" * 70)
    print(f"  总对数: {len(df_out):,}")
    print(f"  score 范围: {df_out['score'].min():.4f} ~ {df_out['score'].max():.4f}")
    print(f"  score 均值: {df_out['score'].mean():.4f}")
    print(f"  score 中位数: {df_out['score'].median():.4f}")

    print(f"\n  输出文件:")
    print(f"    主结果        : {args.output}")
    if args.save_residue_scores and residue_scores:
        print(f"    残基分数 CSV  : {out_stem}.residue_scores.csv")
        print(f"    残基分数 NPZ  : {out_stem}.residue_scores.npz")
    if args.plot and residue_scores:
        plot_dir = args.plot_dir if args.plot_dir else f"{out_stem}_plots"
        print(f"    热图目录      : {plot_dir}/")

    print(f"\n  前 5 行预览:")
    print(df_out.head(5).to_string())


if __name__ == "__main__":
    main()