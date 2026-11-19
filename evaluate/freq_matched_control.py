"""频率匹配对照实验（经典线性分类头）：量化监督自发习得的谱系排序信号。

实验定义：
  对 GO DAG 每条祖先关系对 (u=父/祖先, v=子/后代)，要求子节点训练正样本数 n_v >= 1：
      true_viol(e) = mean_i 1[P_i(v) > P_i(u) + eps]                    (eps = 1e-4)
      ctrl_viol(e) = mean_{w in W(u,delta)} mean_i 1[P_i(v) > P_i(w) + eps]   (K=30 个对照类)
      保护增益 gain(e) = ctrl_viol(e) - true_viol(e)
  对照类池 W(u,δ) = { w : |log(n_w+1) - log(n_u+1)| <= δ,
                      w != v, hop[v,w]=0 (w 非祖先), hop[w,v]=0 (w 非后代), n_w >= 1 }
  注：n_w >= 1 为本实验附加限制——零计数类的概率纯由负梯度塑造（退化对照），
      纳入会机械性抬高 ctrl_viol、伪造保护增益，故排除。

口径：direct(hop=1) / closure(hop>0)；分桶按子节点计数 n_v ∈ {1-2, 3-10, 11-50, >50}。
校准：TALE/CC 上扫描 δ → 取"最窄且增益稳定(与所有更宽 δ 差 < 1pp)、覆盖率>=95%"的 δ*。
对比：同一对照设计下同时报告原型度量模块（PrototypeNet，最新 checkpoint）的保护增益。

用法：
  python freq_matched_control.py --device cuda                       # 全量
  python freq_matched_control.py --smoke                             # 仅 TALE/CC 单 δ 冒烟
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils as utils
from models.custom_model import Model
from models.prototype_model import PrototypeNet
from datasets.dataset import Dataset
from datasets.collator import collator
from torch.utils.data import DataLoader, SequentialSampler

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

T_START = time.time()
EPS = 1e-4
K_CTRL = 30
DEFAULT_DELTAS = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
BUCKETS = [('1-2', 1, 2), ('3-10', 3, 10), ('11-50', 11, 50), ('>50', 51, np.inf)]
PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJ, 'data_tale')
FEAT_ROOT = '/archive/hot5/fty'
CKPT_ROOT = '/archive/hot5/fty/checkpoints'
OUT_DIR = os.path.join(PROJ, 'tmp', 'freq_matched_control')

REF = ('TALE', 'CC')  # 校准参考组合
COMBOS = [
    ('TALE', 'CC', 'CC_TALE_checkpoint_45.pth'),
    ('CAFA3', 'CC', 'CC_CAFA3_checkpoint_34.pth'),
    ('TALE', 'MF', 'MF_TALE_checkpoint_48.pth'),
    ('CAFA3', 'MF', 'MF_CAFA3_checkpoint_40.pth'),
    ('TALE', 'BP', 'BP_TALE_checkpoint_46.pth'),
    ('CAFA3', 'BP', 'BP_CAFA3_checkpoint_42.pth'),
]
CKPT_OF = {(ds, ns): ck for ds, ns, ck in COMBOS}


def log(msg=''):
    print(f'[{time.time() - T_START:7.1f}s] {msg}', flush=True)


def floyd_sample(n, k, rng):
    """Floyd 算法：从 {0..n-1} 无放回均匀抽 k 个，O(k)。"""
    if k >= n:
        return np.arange(n)
    S = {}
    for j in range(n - k, n):
        t = int(rng.integers(0, j + 1))
        S[j] = S.get(t, t)
    return np.fromiter(S.values(), dtype=np.int64, count=k)


# --------------------------------------------------------------------------
# 数据与模型
# --------------------------------------------------------------------------

def load_combo(ds, ns, device):
    dp = os.path.join(DATA_ROOT, ds)
    C = int(np.load(os.path.join(dp, f'{ns.lower()}_label_matrix_1_sparse.npy')).shape[0])
    train_seq = utils.load_data_from_pkl(os.path.join(dp, f'train_seq_{ns.lower()}'))
    edges = np.load(os.path.join(dp, f'{ns.lower()}_label_regular_1.npy'))
    log(f'[{ds}/{ns}] 计算 hop 矩阵 (BFS, C={C}) ...')
    hop_t = utils.get_ancestor_hop_matrix(edges)
    assert hop_t.shape[0] == C, f'hop {hop_t.shape} vs C={C}'
    hop_int = hop_t.numpy()
    # 训练计数（标签已沿 DAG 传播，保证 count(父) >= count(子)）；空标注蛋白兜底根类
    counts = np.zeros(C, dtype=np.int64)
    n_empty = 0
    for it in train_seq:
        labs = it.get('label', [])
        if labs:
            for l in labs:
                counts[l] += 1
        else:
            n_empty += 1
    counts_t = torch.from_numpy(counts.astype(np.float32))
    feats_tr = torch.load(os.path.join(FEAT_ROOT, ds, 'protein_feats', f'train_{ns.lower()}_protein_feats.pt'),
                          weights_only=True, map_location='cpu')
    feats_te = torch.load(os.path.join(FEAT_ROOT, ds, 'protein_feats', f'test_{ns.lower()}_protein_feats.pt'),
                          weights_only=True, map_location='cpu')
    test_ds = Dataset(dp, ns, train_mode='test', dataset_mode='test', valid_mask=None)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, sampler=SequentialSampler(test_ds),
                        drop_last=False, num_workers=0, collate_fn=collator(C, feats_te))
    log(f'[{ds}/{ns}] 训练蛋白={len(train_seq)}, 测试蛋白={len(test_ds)}, 空标注(兜底根类)={n_empty}')
    return dict(ds=ds, ns=ns, dp=dp, C=C, train_seq=train_seq, hop_t=hop_t, hop_int=hop_int,
                counts=counts, counts_t=counts_t, feats_tr=feats_tr, loader=loader, n_test=len(test_ds))


def classic_head_probs(combo, ckpt_path, device):
    """经典线性分类头概率矩阵 (N_test, C)：sigmoid(proj(feats) @ W^T + b)。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = utils.extract_state_dict(ckpt)
    del ckpt
    C, hidden = sd['classifier.0.weight'].shape
    in_dim = sd['proj.1.weight'].shape[1]
    model = Model(input_dim=in_dim, hidden_dim=hidden, num_classes=C,
                  use_ancestor_weighting=False).to(device)
    model.load_state_dict(sd)
    del sd
    model.eval()
    out = []
    with torch.no_grad():
        for feats, labels, indices in combo['loader']:
            p, _, _ = model(feats.to(device), indices.to(device), labels.to(device))
            out.append(p.float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(out).numpy().astype(np.float32)


def proto_probs(combo, device):
    """原型度量模块概率矩阵 (N_test, C)。取该组合 prototype 目录下 mtime 最新的 checkpoint。"""
    ds, ns, dp, C = combo['ds'], combo['ns'], combo['dp'], combo['C']
    proto_dir = os.path.join(CKPT_ROOT, ns, ds, 'prototype')
    if not os.path.isdir(proto_dir):
        return None, None
    cks = [f for f in os.listdir(proto_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not cks:
        return None, None
    ck = max(cks, key=lambda f: os.path.getmtime(os.path.join(proto_dir, f)))
    try:
        state = utils.extract_state_dict(torch.load(os.path.join(proto_dir, ck), map_location='cpu', weights_only=False))
        hidden = state['mlp.1.weight'].shape[0]
        if state.get('temperature', torch.zeros(0)).shape[0] != C:
            log(f'[{ds}/{ns}] 原型 ckpt 类别数不匹配 ({state["temperature"].shape[0]} != {C})，跳过')
            return None, ck
        go2id = utils.load_data_from_pkl(os.path.join(dp, f'{ns.lower()}_go_1.pickle'))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        log(f'[{ds}/{ns}] 计算 IC (goatools) ...')
        ic = utils.compute_ic(go2id, combo['train_seq'], os.path.join(dp, 'go-basic.obo'))
        freq = combo['counts_t'] / max(len(combo['train_seq']), 1)
        lam, beta, tau = utils.compute_smooth_param_inits(combo['hop_t'], ic, combo['counts_t'])
        pm = PrototypeNet(input_dim=1280, hidden_dim=hidden, num_classes=C, frequency=freq,
                          parents_matrix=parents_matrix, class_counts=combo['counts_t'],
                          hop_counts=combo['hop_t'], ic=ic,
                          smooth_tau=tau, lambda_init=lam, beta_init=beta).to(device)
        if 'tau_min' not in state:
            state['tau_min'] = torch.tensor(0.0)
        pm.load_state_dict(state)
        pm.eval()
        # ---- 原型构造：index_add 分块复刻 _get_prototypes（均值 + _smooth_prototypes），避免 (N,C) 大矩阵 ----
        feats = combo['feats_tr']
        keys = sorted(feats.keys(), key=int)
        assert len(keys) == len(combo['train_seq'])
        all_feats = torch.stack([feats[k] for k in keys], dim=0).to(device)
        rows, cols = [], []
        for i, it in enumerate(combo['train_seq']):
            labs = it.get('label', [])
            if labs:
                rows.extend(labs)
                cols.extend([i] * len(labs))
            else:  # 空标注兜底根类，与 prototype.py 的 root-fix 一致
                rows.append(0)
                cols.append(i)
        cls_i = torch.tensor(rows, dtype=torch.long)
        prot_i = torch.tensor(cols, dtype=torch.long)
        order2 = torch.argsort(prot_i)
        cls_s = cls_i[order2].to(device)
        prot_s = prot_i[order2].to(device)
        with torch.no_grad():
            s_feats = pm.mlp(all_feats)
            D = s_feats.shape[1]
            proto_sum = torch.zeros(C, D, device=device)
            cnt = torch.zeros(C, device=device)
            one = torch.ones(1, device=device)
            CH = 100000
            for s in range(0, len(cls_s), CH):
                e = min(s + CH, len(cls_s))
                proto_sum.index_add_(0, cls_s[s:e], s_feats[prot_s[s:e]])
                cnt.index_add_(0, cls_s[s:e], one.expand(e - s))
            means = proto_sum / cnt.clamp(min=1).unsqueeze(1)
            prototypes = pm._smooth_prototypes(means)
            del s_feats, proto_sum, all_feats
            out = []
            for bf, labels, indices in combo['loader']:
                q = pm.mlp(bf.to(device))
                logits = pm.predict(q, prototypes)
                out.append(torch.sigmoid(logits).float().cpu())
        del pm, prototypes
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cat(out).numpy().astype(np.float32), ck
    except Exception:
        log(f'[{ds}/{ns}] 原型模块失败，跳过:\n{traceback.format_exc()}')
        gc.collect()
        torch.cuda.empty_cache()
        return None, ck


# --------------------------------------------------------------------------
# 边集 / 对照采样 / 统计
# --------------------------------------------------------------------------

def build_edges(hop_int, counts, scope):
    """scope=direct: hop==1；closure: hop>0。返回 (parent, child)，仅保留子节点 n_v>=1 的边，按子节点排序。"""
    if scope == 'direct':
        r, c = np.nonzero(hop_int == 1)
    else:
        r, c = np.nonzero(hop_int > 0)
    parent, child = c.astype(np.int64), r.astype(np.int64)
    keep = counts[child] >= 1
    parent, child = parent[keep], child[keep]
    o = np.argsort(child, kind='stable')
    return parent[o], child[o]


def sample_controls(parent, child, counts, hop_bool, desc_bool, delta, k_ctrl, rng):
    """逐边采样 K 个频率匹配非亲缘对照类。

    窗口: |log(n_w+1) - log(n_u+1)| <= δ (np.searchsorted 在 log 排序数组上定位)；
    排除: v 的祖先 (hop[v,w]>0)、v 的后代 (hop[w,v]>0)、w==v、n_w<1。
    按子节点分组处理：每个 v 只做一次亲缘过滤，组内各边在过滤后的候选上二次定位窗口。
    Returns:
        controls: (E, k_ctrl) int64，-1 为填充位；n_cand: (E,) 各边合格候选数
    """
    E = len(parent)
    logs = np.log(counts.astype(np.float64) + 1.0)
    order = np.argsort(logs, kind='stable')
    sorted_logs = logs[order]
    controls = np.full((E, k_ctrl), -1, dtype=np.int64)
    n_cand = np.zeros(E, dtype=np.int64)
    uv, vs = np.unique(child, return_index=True)
    for vi in range(len(uv)):
        v = int(uv[vi])
        s = int(vs[vi])
        e = int(vs[vi + 1]) if vi + 1 < len(uv) else E
        anc_v = hop_bool[v]        # v 的祖先行
        desc_v = desc_bool[v]      # v 的后代（转置连续存储后的行）
        us = parent[s:e]
        lo_all = np.searchsorted(sorted_logs, logs[us] - delta, 'left')
        hi_all = np.searchsorted(sorted_logs, logs[us] + delta, 'right')
        union = order[int(lo_all.min()):int(hi_all.max())]
        bad = anc_v[union] | desc_v[union] | (union == v) | (counts[union] < 1)
        ok = union[~bad]
        ok_logs = logs[ok]
        for ei in range(s, e):
            u = parent[ei]
            a = int(np.searchsorted(ok_logs, logs[u] - delta, 'left'))
            b = int(np.searchsorted(ok_logs, logs[u] + delta, 'right'))
            m = b - a
            n_cand[ei] = m
            if m <= 0:
                continue
            if m <= k_ctrl:
                controls[ei, :m] = ok[a:b]
            else:
                pos = floyd_sample(m, k_ctrl, rng)
                controls[ei, :k_ctrl] = ok[a + pos]
    return controls, n_cand


def edge_stats(probs, parent, child, controls, eps=EPS, chunk=384):
    """逐边真实/对照违反率。分块向量化：每块一次 (N, Ec, K) 比较。

    true_viol = mean_i 1[P_i(v) > P_i(u)+eps]；
    ctrl_viol = mean_{有效 w} mean_i 1[P_i(v) > P_i(w)+eps]（无对照的边记 NaN）。
    """
    E = len(parent)
    N = probs.shape[0]
    true_v = np.zeros(E, dtype=np.float64)
    ctrl_v = np.full(E, np.nan, dtype=np.float64)
    for s in range(0, E, chunk):
        e = slice(s, min(s + chunk, E))
        u = parent[e]
        v = child[e]
        w2 = controls[e]
        pv = probs[:, v]                       # (N, Ec)
        true_v[e] = (pv > probs[:, u] + eps).mean(axis=0)
        valid = w2 >= 0                        # (Ec, K)
        if not valid.any():
            continue
        wsafe = np.where(valid, w2, 0)
        pw = probs[:, wsafe]                   # (N, Ec, K)
        ind = pv[:, :, None] > (pw + eps)
        ind &= valid[None, :, :]
        denom = np.maximum(valid.sum(axis=1), 1) * N
        ctrl_v[e] = np.where(valid.any(axis=1), ind.sum(axis=(0, 2)) / denom, np.nan)
        del pw, ind
    return true_v, ctrl_v


def true_violation_all(probs, parent, child, chunk=2048):
    """全量边（不过滤子节点）的真实违反率均值——用于与论文 Table 5 的 all 列对账。"""
    E = len(parent)
    tv = np.zeros(E, dtype=np.float64)
    for s in range(0, E, chunk):
        e = slice(s, min(s + chunk, E))
        tv[e] = (probs[:, child[e]] > probs[:, parent[e]] + EPS).mean(axis=0)
    return float(tv.mean())


def bucket_stats(true_v, ctrl_v, child_counts):
    """按子节点计数 n_v 分桶：边数/覆盖数/true/ctrl/增益/逐边配对 Wilcoxon p（双侧）。

    Args:
        child_counts: (E,) 每条边子节点的训练正样本数（counts[child]，非节点 id）。
    """
    rows = []
    for name, lo, hi in BUCKETS:
        sel = (child_counts >= lo) & (child_counts <= hi)
        n_edges = int(sel.sum())
        selc = sel & np.isfinite(ctrl_v)
        n_cov = int(selc.sum())
        if n_cov == 0:
            rows.append({'bucket': name, 'edges': n_edges, 'covered': 0})
            continue
        t = true_v[selc]
        c = ctrl_v[selc]
        p = float('nan')
        if wilcoxon is not None and n_cov >= 3:
            try:
                p = float(wilcoxon(c, t, zero_method='wilcox').pvalue)
            except ValueError:
                p = 1.0  # 全部逐边差分为零：无任何差异证据
            except Exception:
                p = float('nan')
        rows.append({'bucket': name, 'edges': n_edges, 'covered': n_cov,
                     'true': float(t.mean()), 'ctrl': float(c.mean()),
                     'gain': float(c.mean() - t.mean()), 'p': p})
    return rows


def print_rows(scope, model_name, key, rows):
    for b in rows:
        if b['covered'] == 0:
            print(f'  [{scope:<7}|{model_name:<7}] {key:<10} {b["bucket"]:>6}  边数={b["edges"]:<7} 覆盖=0', flush=True)
            continue
        print(f'  [{scope:<7}|{model_name:<7}] {key:<10} {b["bucket"]:>6}  边数={b["edges"]:<7} 覆盖={b["covered"]:<7} '
              f'true={b["true"]:.4f}  ctrl={b["ctrl"]:.4f}  增益={b["gain"]:+.4f}  p={b["p"]:.2e}', flush=True)


def gains_of(entry, model='classic'):
    return {b['bucket']: b.get('gain', float('nan')) for b in entry['models'][model]['buckets']}


def pick_delta(cal, deltas):
    """选最窄 δ：满足 (a) coverage>=0.95；(b) 再收窄一档时各桶增益变化 < 1pp。

    动机：窗口越窄，对照类与参照父类的频率匹配越严格；宽窗口因长尾不对称
    （高频父类向下扩窗引入大量更低频对照）会机械性抬高 ctrl_viol，
    故取"窄端稳定平台"上最窄的可覆盖 δ 作保守估计。
    若无 δ 同时满足 (a)(b)，回退为 coverage>=0.9 中收窄步长最小者。
    """
    ds = sorted(deltas)
    info = {d: (cal['direct'][str(d)]['coverage'], gains_of(cal['direct'][str(d)])) for d in ds}
    chosen, fallback, fb_dev = None, None, np.inf
    for i, d in enumerate(ds):
        cov, g = info[d]
        if i == 0:
            step = 0.0  # 已是网格最窄，无更窄档位
        else:
            _, g_prev = info[ds[i - 1]]
            pairs = [(g[b], g_prev.get(b, float('nan'))) for b in g]
            step = max(abs(a - b2) for a, b2 in pairs
                       if np.isfinite(a) and np.isfinite(b2)) if pairs else 0.0
        if cov >= 0.95 and step < 0.01 and chosen is None:
            chosen = d
        if cov >= 0.9 and step < fb_dev:
            fallback, fb_dev = d, step
    if chosen is None:
        chosen = fallback
    return chosen, info


def main():
    global K_CTRL
    ap = argparse.ArgumentParser(description='经典分类头频率匹配对照实验')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--deltas', type=str, default=','.join(str(d) for d in DEFAULT_DELTAS))
    ap.add_argument('--k', type=int, default=K_CTRL)
    ap.add_argument('--skip-proto', action='store_true')
    ap.add_argument('--delta-fixed', type=float, default=None,
                    help='跳过校准扫描，直接用给定 δ 跑六组合最终表')
    ap.add_argument('--calib-all', action='store_true',
                    help='对全部六组合做 δ 校准扫描（仅经典头，双口径），不跑最终表')
    ap.add_argument('--smoke', action='store_true', help='仅 TALE/CC、单 δ，冒烟验证')
    args = ap.parse_args()
    K_CTRL = args.k
    deltas = [float(x) for x in args.deltas.split(',')]
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {'meta': {'eps': EPS, 'k': K_CTRL, 'deltas': deltas,
                        'control_rule': '|log(n_w+1)-log(n_u+1)|<=δ, 非亲缘, w!=v, n_w>=1',
                        'proto_ckpts': {}},
               'calibration': {}, 'final': {}, 'sanity': {}}

    # ============ 模式一：全部六组合 δ 校准扫描（仅经典头） ============
    if args.calib_all:
        out = {'meta': {'eps': EPS, 'k': K_CTRL, 'deltas': deltas,
                        'control_rule': '|log(n_w+1)-log(n_u+1)|<=δ, 非亲缘, w!=v, n_w>=1'},
               'calibration': {}}
        for ds, ns, ckf in COMBOS:
            key = f'{ds}/{ns}'
            try:
                cb = load_combo(ds, ns, device)
                pmat = classic_head_probs(cb, os.path.join(CKPT_ROOT, ckf), device)
                hb = cb['hop_int'] > 0
                db = np.ascontiguousarray(hb.T)
                out['calibration'][key] = {}
                for sc in ('direct', 'closure'):
                    parent, child = build_edges(cb['hop_int'], cb['counts'], sc)
                    out['calibration'][key][sc] = {}
                    for d in deltas:
                        rng = np.random.default_rng(12345)
                        controls, n_cand = sample_controls(parent, child, cb['counts'], hb, db,
                                                           d, K_CTRL, rng)
                        entry = {'coverage': float((n_cand > 0).mean()), 'models': {}}
                        t, c = edge_stats(pmat, parent, child, controls)
                        entry['models']['classic'] = {'buckets': bucket_stats(t, c, cb['counts'][child])}
                        out['calibration'][key][sc][str(d)] = entry
                        g = gains_of(entry)
                        pst = {b['bucket']: b.get('p', float('nan')) for b in entry['models']['classic']['buckets']}
                        log(f'[校准 {key} {sc} δ={d}] 覆盖率={entry["coverage"]:.4f} | ' +
                            ' '.join(f'{b}: gain={g[b]:+.4f} (p={pst[b]:.1e})' for b in g))
                        del controls
                del cb, pmat, hb, db
            except Exception:
                log(f'[{key}] 校准失败:\n{traceback.format_exc()}')
            gc.collect()
            torch.cuda.empty_cache()
        with open(os.path.join(OUT_DIR, 'calib_all.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        log(f'全部组合校准完成，已保存 {os.path.join(OUT_DIR, "calib_all.json")}')
        return

    # ============ 校准组合 TALE/CC ============
    combo = load_combo(*REF, device)
    key = f'{REF[0]}/{REF[1]}'
    probs = {'classic': classic_head_probs(combo, os.path.join(CKPT_ROOT, CKPT_OF[REF]), device)}
    log(f'[{key}] 经典头概率矩阵 {probs["classic"].shape}')
    if not args.skip_proto:
        pp, pck = proto_probs(combo, device)
        if pp is not None:
            probs['proto'] = pp
            results['meta']['proto_ckpts'][key] = pck
            log(f'[{key}] 原型模块概率矩阵 {pp.shape} (ckpt={pck})')

    hop_int = combo['hop_int']
    # sanity：全量边（含零样本子节点）违反率，应与论文 Table 5 的 all 列一致
    for sc in ('direct', 'closure'):
        r0, c0 = np.nonzero(hop_int == 1) if sc == 'direct' else np.nonzero(hop_int > 0)
        tv = true_violation_all(probs['classic'], c0, r0)
        results['sanity'][f'{key}_{sc}_classic_all'] = tv
        log(f'  [sanity] {sc} 全量边经典头违反率 = {tv:.4f}（对照 Table 5 标准头 all 列）')

    edges = {sc: build_edges(hop_int, combo['counts'], sc) for sc in ('direct', 'closure')}
    for sc in ('direct', 'closure'):
        log(f'  {sc}: 有监督边(n_v>=1) {len(edges[sc][0])}')
    hop_bool = hop_int > 0
    desc_bool = np.ascontiguousarray(hop_bool.T)

    # ============ δ 校准扫描 ============
    cal_scopes = ('direct',) if args.smoke else ('direct', 'closure')
    dlist = deltas[:1] if args.smoke else deltas
    for sc in cal_scopes:
        parent, child = edges[sc]
        results['calibration'][sc] = {}
        for d in dlist:
            rng = np.random.default_rng(12345)
            controls, n_cand = sample_controls(parent, child, combo['counts'], hop_bool, desc_bool,
                                               d, K_CTRL, rng)
            entry = {'coverage': float((n_cand > 0).mean()), 'models': {}}
            for name, pmat in probs.items():
                t, c = edge_stats(pmat, parent, child, controls)
                entry['models'][name] = {'buckets': bucket_stats(t, c, combo['counts'][child])}
            results['calibration'][sc][str(d)] = entry
            g = gains_of(entry)
            pstr = {b['bucket']: b.get('p', float('nan')) for b in entry['models']['classic']['buckets']}
            log(f'[校准 {sc} δ={d}] 覆盖率={entry["coverage"]:.4f} | ' +
                ' '.join(f'{b}: gain={g[b]:+.4f} (p={pstr[b]:.1e})' for b in g))
            for name in entry['models']:
                if name != 'classic':
                    gp = gains_of(entry, name)
                    log(f'    [{name}] ' + ' '.join(f'{b}: {gp[b]:+.4f}' for b in gp))
            del controls
    if args.smoke:
        with open(os.path.join(OUT_DIR, 'smoke.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        log('冒烟完成')
        return

    # ============ 选定 δ* ============
    if args.delta_fixed is not None:
        delta_star = args.delta_fixed
        log(f'===> 使用指定 δ* = {delta_star}')
    else:
        delta_star, info = pick_delta(results['calibration'], deltas)
    results['meta']['delta_star'] = delta_star
    if args.delta_fixed is None:
        log('===> 校准明细 (direct, 经典头): ' +
            ' | '.join(f'δ={d}: cov={info[d][0]:.3f}, gains=' +
                       ','.join(f'{b}:{info[d][1][b]:+.4f}' for b in info[d][1]) for d in deltas))
    log(f'===> 选定 δ* = {delta_star}')

    # ============ 六组合最终表 ============
    for ds, ns, ckf in COMBOS:
        ckey = f'{ds}/{ns}'
        try:
            if (ds, ns) == REF:
                cb, pdict, hb, db = combo, probs, hop_bool, desc_bool
            else:
                cb = load_combo(ds, ns, device)
                pdict = {'classic': classic_head_probs(cb, os.path.join(CKPT_ROOT, ckf), device)}
                if not args.skip_proto:
                    pp, pck = proto_probs(cb, device)
                    if pp is not None:
                        pdict['proto'] = pp
                        results['meta']['proto_ckpts'][ckey] = pck
                hb = cb['hop_int'] > 0
                db = np.ascontiguousarray(hb.T)
            # sanity（全量边）
            for sc in ('direct', 'closure'):
                r0, c0 = np.nonzero(cb['hop_int'] == 1) if sc == 'direct' else np.nonzero(cb['hop_int'] > 0)
                results['sanity'][f'{ckey}_{sc}_classic_all'] = true_violation_all(pdict['classic'], c0, r0)
            results['final'][ckey] = {}
            for sc in ('direct', 'closure'):
                parent, child = build_edges(cb['hop_int'], cb['counts'], sc)
                rng = np.random.default_rng(12345)
                controls, n_cand = sample_controls(parent, child, cb['counts'], hb, db, delta_star, K_CTRL, rng)
                cov = float((n_cand > 0).mean())
                results['final'][ckey][sc] = {'coverage': cov, 'models': {}}
                log(f'[{ckey} | {sc} | δ*={delta_star}] 有监督边={len(parent)}, 覆盖率={cov:.4f}')
                for name, pmat in pdict.items():
                    t, c = edge_stats(pmat, parent, child, controls)
                    rows = bucket_stats(t, c, cb['counts'][child])
                    results['final'][ckey][sc]['models'][name] = rows
                    print_rows(sc, name, ckey, rows)
                del controls
            if (ds, ns) != REF:
                del cb, pdict, hb, db
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            log(f'[{ckey}] 失败:\n{traceback.format_exc()}')
            gc.collect()
            torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log(f'结果已保存至 {os.path.join(OUT_DIR, "results.json")}')

    # ============ 汇总表（增益）============
    print('\n==================== 汇总：保护增益 (ctrl_viol - true_viol) ====================', flush=True)
    for sc in ('direct', 'closure'):
        for name in ('classic', 'proto'):
            print(f'---- {sc} | {name} ----', flush=True)
            hdr = f'{"组合":<10}' + ''.join(f'{b:>12}' for b, _, _ in BUCKETS)
            print(hdr, flush=True)
            for ds, ns, _ in COMBOS:
                ckey = f'{ds}/{ns}'
                fin = results['final'].get(ckey, {}).get(sc, {}).get('models', {}).get(name)
                if not fin:
                    continue
                row = f'{ckey:<10}'
                for b in fin:
                    row += f'{b.get("gain", float("nan")):>12.4f}' if b.get('gain') is not None else f'{"-":>12}'
                print(row, flush=True)
    log('全部完成')


if __name__ == '__main__':
    main()
