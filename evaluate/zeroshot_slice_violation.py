"""零样本切片（n_v=0 子节点的边）层级违反率 + 经典分类头按子桶复核。

- 切片：zero-shot(n_v=0)、1-2、3-10、11-50、>50、supervised-all(n_v>=1)、unfiltered-all
- 模型：经典分类头（根目录旧格式 ckpt）、层级感知分类头（custom/ 最新含 w_free 的新格式 ckpt）、
  原型模块（prototype/ 最新 ckpt，与 freq_matched_control 同源）
- 口径：direct（label_regular 边表）与 closure（hop>0 全对），逐边违反率宏平均
"""
import gc
import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate.freq_matched_control as F
import utils as utils
from models.custom_model import Model as CModel
from custom import build_ancestor_prior

OUT = os.path.join(F.OUT_DIR, 'zeroshot_slice_violation.json')


def load_aware_probs(combo, device):
    """从 custom/ 目录最新的新格式（含 w_free）checkpoint 构建层级感知分类头并前向。"""
    ds, ns, dp, C = combo['ds'], combo['ns'], combo['dp'], combo['C']
    cdir = os.path.join(F.CKPT_ROOT, ns, ds, 'custom')
    if not os.path.isdir(cdir):
        return None, None
    cks = sorted(os.listdir(cdir), key=lambda f: -os.path.getmtime(os.path.join(cdir, f)))
    edges = np.load(os.path.join(dp, f'{ns.lower()}_label_regular_1.npy'))
    for ck in cks:
        if not (ck.startswith('checkpoint_') and ck.endswith('.pth')):
            continue
        try:
            state = utils.extract_state_dict(
                torch.load(os.path.join(cdir, ck), map_location='cpu', weights_only=False))
            if 'w_free' not in state:
                continue  # 跳过旧格式（经典头）
            hidden = state['w_free'].shape[1]
            ancestor_mask, ic, lam, gam, tau = build_ancestor_prior(
                dp, ns, ds, edges, combo['train_seq'], combo['counts_t'], None, True)
            model = CModel(input_dim=1280, hidden_dim=hidden, num_classes=C,
                           use_ancestor_weighting=True, ancestor_mask=ancestor_mask, ic=ic,
                           class_counts=combo['counts_t'], alpha_lambda_init=lam,
                           ic_gamma_init=gam, beta_init=0.9, smooth_tau_init=tau).to(device)
            model.load_state_dict(state)
            model.eval()
            out = []
            with torch.no_grad():
                for feats, labels, indices in combo['loader']:
                    p, _, _ = model(feats.to(device), indices.to(device), labels.to(device))
                    out.append(p.float().cpu())
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return torch.cat(out).numpy().astype(np.float32), ck
        except Exception:
            F.log(f'[{ds}/{ns}] aware ckpt {ck} 失败:\n{traceback.format_exc()}')
            continue
    return None, None


def slice_violations(probs, parent, child, counts, slices):
    E = len(parent)
    tv = np.zeros(E, dtype=np.float64)
    for s in range(0, E, 2048):
        e = slice(s, min(s + 2048, E))
        tv[e] = (probs[:, child[e]] > probs[:, parent[e]] + F.EPS).mean(axis=0)
    out = {}
    for name, sel in slices.items():
        out[name] = {'edges': int(sel.sum()),
                     'viol': float(tv[sel].mean()) if sel.any() else None}
    return out


def main():
    device = 'cuda'
    res = {'meta': {'note': '切片按子节点训练正样本数 n_v 划分；逐边违反率宏平均'}}
    for ds, ns, ckf in F.COMBOS:
        key = f'{ds}/{ns}'
        try:
            combo = F.load_combo(ds, ns, device)
            counts = combo['counts']
            models = {'classic': F.classic_head_probs(
                combo, os.path.join(F.CKPT_ROOT, ckf), device)}
            aware_p, aware_ck = load_aware_probs(combo, device)
            if aware_p is not None:
                models['aware'] = aware_p
            proto_p, proto_ck = F.proto_probs(combo, device)
            if proto_p is not None:
                models['proto'] = proto_p
            res['meta'].setdefault('ckpts', {})[key] = {
                'aware': aware_ck, 'proto': proto_ck}

            hop = combo['hop_int']
            edges = np.load(os.path.join(combo['dp'], f'{ns.lower()}_label_regular_1.npy'))
            edge_sets = {
                'direct': (edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)),
                'closure': None,
            }
            r, c = np.nonzero(hop > 0)
            edge_sets['closure'] = (c.astype(np.int64), r.astype(np.int64))

            res[key] = {}
            for sc, (parent, child) in edge_sets.items():
                cv = counts[child]
                slices = {
                    'zero-shot(n_v=0)': cv == 0,
                    '1-2': (cv >= 1) & (cv <= 2),
                    '3-10': (cv >= 3) & (cv <= 10),
                    '11-50': (cv >= 11) & (cv <= 50),
                    '>50': cv > 50,
                    'supervised-all(n_v>=1)': cv >= 1,
                    'unfiltered-all': np.ones(len(parent), dtype=bool),
                }
                res[key][sc] = {}
                for name, pmat in models.items():
                    res[key][sc][name] = slice_violations(pmat, parent, child, counts, slices)
                line = f'[{key} {sc}] zero-shot边={res[key][sc]["classic"]["zero-shot(n_v=0)"]["edges"]}: '
                for name in models:
                    z = res[key][sc][name]['zero-shot(n_v=0)']['viol']
                    line += f'{name}={z if z is None else round(z, 4)} '
                F.log(line)
            del combo, models, hop
        except Exception:
            F.log(f'[{key}] 失败:\n{traceback.format_exc()}')
        gc.collect()
        torch.cuda.empty_cache()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    F.log(f'已保存 {OUT}')


if __name__ == '__main__':
    main()
