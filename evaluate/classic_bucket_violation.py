"""复核 Table 5 经典分类头行：按父节点计数分桶的层级违反率（direct + closure，全量边）。

分桶与 eval_hier_violation_rate 完全一致：rare=(0,10], mid=(10,50], high=(50,inf]，all=全部。
direct 边直接取 label_regular_1.npy（等价 hop==1）；closure 取 hop>0 全部对。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate.freq_matched_control as F

OUT = os.path.join(F.OUT_DIR, 'classic_bucket_violation.json')


def violation_buckets(probs, parent, child, counts):
    E = len(parent)
    tv = np.zeros(E, dtype=np.float64)
    for s in range(0, E, 2048):
        e = slice(s, min(s + chunk, E))
        tv[e] = (probs[:, child[e]] > probs[:, parent[e]] + F.EPS).mean(axis=0)
    bins = [('all', None, None), ('rare(0,10]', 0, 10), ('mid(10,50]', 10, 50), ('high(>50)', 50, np.inf)]
    out = {'edges': int(E)}
    for name, lo, hi in bins:
        sel = np.ones(E, dtype=bool) if lo is None else \
            (lo < counts[parent]) & (counts[parent] <= hi)
        out[name] = float(tv[sel].mean()) if sel.any() else float('nan')
    return out


chunk = 2048


def main():
    import torch
    device = 'cuda'
    res = {}
    for ds, ns, ckf in F.COMBOS:
        key = f'{ds}/{ns}'
        combo = F.load_combo(ds, ns, device)
        probs = F.classic_head_probs(combo, os.path.join(F.CKPT_ROOT, ckf), device)
        hop = combo['hop_int']
        counts = combo['counts']
        res[key] = {}
        # direct：label_regular 边表（即直接父子边）
        edges = np.load(os.path.join(combo['dp'], f'{ns.lower()}_label_regular_1.npy'))
        p_d, c_d = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        res[key]['direct'] = violation_buckets(probs, p_d, c_d, counts)
        F.log(f'[{key}] direct: ' + str(res[key]['direct']))
        # closure：hop>0 全部对
        r, c = np.nonzero(hop > 0)
        res[key]['closure'] = violation_buckets(probs, c.astype(np.int64), r.astype(np.int64), counts)
        F.log(f'[{key}] closure: ' + str(res[key]['closure']))
        del combo, probs, hop
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    F.log(f'已保存 {OUT}')


if __name__ == '__main__':
    main()
