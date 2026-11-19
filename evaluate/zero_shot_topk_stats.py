"""零样本 top-k bootstrap 置信区间与显著性检验（论文 3.2 节"统计功效的说明"补实验 P1）。

作用:
    读取 run_topk_dump.py 落盘的 6 个组合零样本概率/标签矩阵
    （tmp/topk_probs_<数据集>_<本体>_mt1.npz），调用
    utils.eval_zero_shot_topk_bootstrap_stats，以测试蛋白为重采样单元做 n_boot 次
    bootstrap，输出各 k 下 Fmax / micro_AUPRC 的 95% 置信区间，以及相对最强专用
    基线 MZSGO 的双向单侧 bootstrap p 值：
        p_sup 小（<0.05）=> 证据支持"优于基线"；
        p_inf 小（<0.05）=> 证据支持"劣于基线"；
        两者均不小      => "与基线无显著差异（持平）"。

    基线数字来源: MZSGO 复现训练日志 /archive/hot5/fty/log/mzsgo_*.log 中的
    zero-shot top-k 评估表（与论文补充材料 Table S5 同源；其中 TALE MF 的
    top-1 Fmax=0.3009 与正文 3.2 节引用值一致，已核对）。若需改用/追加其他基线
    （如 STAR-GO），修改下方 BASELINES 后重跑即可，无需改动其它代码。

用法（在 project/ppi/new 目录下，synergy 环境）:
    ~/.conda/envs/synergy/bin/python evaluate/zero_shot_topk_stats.py --n-boot 1000 --seed 42

输出:
    - 终端: k=1 汇总表（回复信主表）+ 各组合完整 per-k 明细表；
    - tmp/zero_shot_topk_bootstrap.json: 全部数值（meta + 各 k CI/p 值）。

删除影响说明:
    本文件承载统计协议与基线数字，是复现置信区间 / p 值的唯一入口；删除后虽然
    概率矩阵可由 run_topk_dump.py 重新生成，但统计结果将无法复现，请保留。
"""
import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import utils  # noqa: E402  (project root)

# ---------------------------------------------------------------------------
# 最强专用基线 MZSGO（复现）各组合 zero-shot top-k 数值，k = 1..10。
# 提取自 /archive/hot5/fty/log/mzsgo_<dataset>_<ns>.log 中 best checkpoint 的
# "Starting zero-shot top-k evaluation" 表（论文 Table S5 同源）。
# ---------------------------------------------------------------------------
BASELINES = {
    ('CAFA3', 'MF'): {
        'fmax':  [0.0357, 0.0357, 0.0268, 0.0357, 0.0349, 0.0398, 0.0483, 0.0467, 0.0455, 0.0479],
        'auprc': [0.0415, 0.0434, 0.0389, 0.0538, 0.0597, 0.0736, 0.0948, 0.1005, 0.1064, 0.1196],
    },
    ('CAFA3', 'CC'): {
        'fmax':  [0.0800, 0.1355, 0.1598, 0.1684, 0.1699, 0.1799, 0.1840, 0.1771, 0.1808, 0.1753],
        'auprc': [0.1265, 0.1551, 0.1837, 0.2123, 0.2409, 0.2695, 0.2981, 0.3089, 0.3379, 0.3498],
    },
    ('CAFA3', 'BP'): {
        'fmax':  [0.0208, 0.0292, 0.0352, 0.0420, 0.0460, 0.0401, 0.0396, 0.0462, 0.0450, 0.0465],
        'auprc': [0.0354, 0.0459, 0.0565, 0.0670, 0.0776, 0.0750, 0.0795, 0.0968, 0.1014, 0.1122],
    },
    ('TALE', 'MF'): {
        'fmax':  [0.3009, 0.3402, 0.3105, 0.2763, 0.2525, 0.2302, 0.2173, 0.2050, 0.1918, 0.1793],
        'auprc': [0.3037, 0.3689, 0.3874, 0.3951, 0.4080, 0.4174, 0.4368, 0.4528, 0.4628, 0.4690],
    },
    ('TALE', 'CC'): {
        'fmax':  [0.1515, 0.1339, 0.1086, 0.1082, 0.1079, 0.1061, 0.1014, 0.0906, 0.0858, 0.0845],
        'auprc': [0.1504, 0.1367, 0.1255, 0.1410, 0.1575, 0.1716, 0.1807, 0.1770, 0.1819, 0.1925],
    },
    ('TALE', 'BP'): {
        'fmax':  [0.1085, 0.1202, 0.1380, 0.1308, 0.1279, 0.1202, 0.1144, 0.1090, 0.1042, 0.0989],
        'auprc': [0.1059, 0.1218, 0.1564, 0.1683, 0.1843, 0.1935, 0.2036, 0.2130, 0.2213, 0.2271],
    },
}

ALPHA = 0.05


def verdict(p_sup, p_inf, alpha=ALPHA):
    """双向单侧 bootstrap p 值 -> 结论字符串。"""
    if p_sup < alpha and p_inf >= alpha:
        return '显著优于基线'
    if p_inf < alpha and p_sup >= alpha:
        return '显著劣于基线'
    return '与基线无显著差异(持平)'


def main():
    ap = argparse.ArgumentParser(description='Zero-shot top-k bootstrap CI & significance vs MZSGO')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-k', type=int, default=10)
    ap.add_argument('--model-type', type=int, default=1, help='与 run_topk_dump.py 落盘时的 model_type 一致')
    args = ap.parse_args()

    all_results = {}
    missing = []
    for (ds, ns), base in BASELINES.items():
        npz = os.path.join(BASE, 'tmp', f'topk_probs_{ds}_{ns}_mt{args.model_type}.npz')
        if not os.path.exists(npz):
            missing.append(f'{ds}:{ns}')
            continue
        data = np.load(npz)
        print('\n' + '=' * 86, flush=True)
        print(f'===== {ds} / {ns}   (npz: {os.path.basename(npz)})', flush=True)
        res = utils.eval_zero_shot_topk_bootstrap_stats(
            data['probs'], data['labels'], max_k=args.max_k,
            baseline_fmax=base['fmax'], baseline_auprc=base['auprc'],
            n_boot=args.n_boot, seed=args.seed)
        res['baseline'] = {'MZSGO_fmax': base['fmax'], 'MZSGO_auprc': base['auprc']}
        all_results[f'{ds}/{ns}'] = res

    if missing:
        print(f'\n[警告] 缺少以下组合的概率矩阵（先运行 evaluate/run_topk_dump.py）: {", ".join(missing)}')

    if not all_results:
        return

    # ---------------- k=1 汇总表（回复信主表） ----------------
    print('\n' + '=' * 100)
    print(f'【k=1 汇总】Fmax：点估计 [95% CI] vs 最强专用基线 MZSGO（单侧 bootstrap p，B={args.n_boot}）')
    print('-' * 100)
    print(f"{'组合':<10} {'n_eval':>6} {'Fmax':>7} {'95% CI':>18} {'MZSGO':>7} "
          f"{'p_sup':>7} {'p_inf':>7} {'P(F*≥base)':>10}  结论")
    for key, res in all_results.items():
        m = res['meta']
        e = res['k'][1]
        b = res['baseline']['MZSGO_fmax'][0]
        ci = f"[{e['Fmax_ci'][0]:.4f}, {e['Fmax_ci'][1]:.4f}]"
        print(f"{key:<10} {m['n_eval_proteins']:>6} {e['Fmax']:>7.4f} {ci:>18} {b:>7.4f} "
              f"{e['Fmax_p_sup']:>7.4f} {e['Fmax_p_inf']:>7.4f} {e['Fmax_p_ge_base']:>10.3f}  "
              f"{verdict(e['Fmax_p_sup'], e['Fmax_p_inf'])}")
    print('-' * 100)
    print('【k=1 汇总】micro_AUPRC（同上口径）')
    print(f"{'组合':<10} {'AUPRC':>7} {'95% CI':>18} {'MZSGO':>7} "
          f"{'p_sup':>7} {'p_inf':>7} {'P(A*≥base)':>10}  结论")
    for key, res in all_results.items():
        e = res['k'][1]
        b = res['baseline']['MZSGO_auprc'][0]
        ci = f"[{e['AUPRC_ci'][0]:.4f}, {e['AUPRC_ci'][1]:.4f}]"
        print(f"{key:<10} {e['micro_AUPRC']:>7.4f} {ci:>18} {b:>7.4f} "
              f"{e['AUPRC_p_sup']:>7.4f} {e['AUPRC_p_inf']:>7.4f} {e['AUPRC_p_ge_base']:>10.3f}  "
              f"{verdict(e['AUPRC_p_sup'], e['AUPRC_p_inf'])}")
    print('=' * 100)

    # ---------------- 各组合完整 per-k 明细（对应补充材料 Table S5 口径） ----------------
    for key, res in all_results.items():
        b_f, b_a = res['baseline']['MZSGO_fmax'], res['baseline']['MZSGO_auprc']
        print(f'\n----- {key} 明细 -----')
        print(f"{'k':>3} | {'Fmax':>7} {'[95% CI]':>19} {'MZSGO':>7} {'p_sup':>7} {'p_inf':>7} | "
              f"{'AUPRC':>7} {'[95% CI]':>19} {'MZSGO':>7} {'p_sup':>7} {'p_inf':>7}")
        for k, e in sorted(res['k'].items()):
            print(f"{k:>3} | {e['Fmax']:>7.4f} [{e['Fmax_ci'][0]:.4f},{e['Fmax_ci'][1]:.4f}] "
                  f"{b_f[k-1]:>7.4f} {e['Fmax_p_sup']:>7.4f} {e['Fmax_p_inf']:>7.4f} | "
                  f"{e['micro_AUPRC']:>7.4f} [{e['AUPRC_ci'][0]:.4f},{e['AUPRC_ci'][1]:.4f}] "
                  f"{b_a[k-1]:>7.4f} {e['AUPRC_p_sup']:>7.4f} {e['AUPRC_p_inf']:>7.4f}")

    out_json = os.path.join(BASE, 'tmp', 'zero_shot_topk_bootstrap.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'args': {'n_boot': args.n_boot, 'seed': args.seed, 'max_k': args.max_k,
                            'baseline': 'MZSGO (reproduced, see BASELINES in evaluate/zero_shot_topk_stats.py)',
                            'alpha': ALPHA},
                   'results': all_results}, f, ensure_ascii=False, indent=2)
    print(f'\n全部数值已保存至 {out_json}')


if __name__ == '__main__':
    main()
