"""log 频率窗口校准与谱系保护增益实验。
用法: python evaluate/log_window_calibration.py --dataset DS --ontology NS --classic CKPT --device cuda:0
"""
import argparse, os, sys, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils as utils
from datasets.dataset import Dataset
from datasets.collator import collator
from torch.utils.data import DataLoader, SequentialSampler
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

pa = argparse.ArgumentParser()
pa.add_argument('--dataset', required=True)
pa.add_argument('--ontology', required=True)
pa.add_argument('--classic', required=True)
pa.add_argument('--device', default='cuda:0')
pa.add_argument('--kctrl', type=float, default=30)
a = pa.parse_args()
FEAT = '/archive/hot5/fty'
CK = '/archive/hot5/fty/checkpoints'
nsl = a.ontology.lower()
dp = os.path.join('data_tale', a.dataset)

C = np.load(os.path.join(dp, nsl + '_label_matrix_1_sparse.npy')).shape[0]
train_seq = utils.load_data_from_pkl(os.path.join(dp, 'train_seq_' + nsl))
pi = utils.get_prototype_index(train_seq, C)
pi[pi[:, 0] == 0, 0] = 1
_, pim = utils.get_prototype_index_tensor(train_seq)
_, class_counts = utils.compute_go_term_frequency(pim, len(train_seq))
counts = class_counts.cpu().numpy().astype(np.float64)
edges = np.load(os.path.join(dp, nsl + '_label_regular_1.npy'))
hop = utils.get_ancestor_hop_matrix(edges).numpy()
U, V = [], []
for i, j in zip(*np.nonzero(hop == 1)):
    if counts[i] >= 1:
        U.append(int(j)); V.append(int(i))
U = np.array(U); V = np.array(V)
print('C=%d 直接边=%d' % (C, len(U)), flush=True)

feats = torch.load(os.path.join(FEAT, a.dataset, 'protein_feats', 'test_' + nsl + '_protein_feats.pt'), weights_only=True, map_location='cpu')
tds = Dataset(dp, a.ontology, train_mode='test', dataset_mode='test', valid_mask=None)
loader = DataLoader(tds, batch_size=512, shuffle=False, sampler=SequentialSampler(tds), collate_fn=collator(C, feats), worker_init_fn=utils.seed_worker)
sd = utils.extract_state_dict(torch.load(a.classic, map_location='cpu', weights_only=False))
dd = sd['proj.0.weight'].shape[0]; hh = sd['proj.1.weight'].shape[0]
proj = nn.Sequential(nn.LayerNorm(dd), nn.Linear(dd, hh), nn.GELU(), nn.Dropout(),
                     nn.LayerNorm(hh), nn.Linear(hh, 2*hh), nn.GELU(), nn.Dropout(), nn.Linear(2*hh, hh))
proj.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('proj.')})
proj.eval().to(a.device)
Wc = sd['classifier.0.weight'].float().to(a.device)
bc = sd['classifier.0.bias'].float().to(a.device)
P = []
with torch.no_grad():
    for bf, labels, _ in loader:
        P.append(torch.sigmoid(proj(bf.to(a.device)) @ Wc.T + bc).cpu().numpy())
P = np.concatenate(P)
print('概率矩阵:', P.shape, flush=True)

# ---- 频率匹配对照实验 ----
print('[对照实验] P(子) > P(父/对照) + 1e-4 蛋白占比', flush=True)
print('%10s | %6s | %9s | %9s | %8s | %10s' % ('桶', '边数', '真实父类', '匹配对照', '差值', 'p(配对)'))
print('-' * 78)
rng = np.random.default_rng(42)
for bname, lo, hi in [('1-2', 1, 2), ('3-10', 3, 10), ('11-50', 11, 50), ('>50', 51, 10**9)]:
    tv, cv, nu = [], [], 0
    for (u, v) in zip(U, V):
        if not (lo <= counts[v] <= hi): continue
        true_v = (P[:, v] > P[:, u] + 1e-4).mean()
        tv.append(true_v)
        cu = counts[u]
        cand = np.where((counts >= cu / 3) & (counts <= 3 * cu))[0]
        cand = cand[(hop[v, cand] == 0) & (hop[cand, v] == 0) & (cand != v)]
        if len(cand) == 0: continue
        take = min(30, len(cand))
        w = rng.choice(cand, size=take, replace=False)
        cvv = np.mean([(P[:, v] > P[:, int(wc)] + 1e-4).mean() for wc in w])
        cv.append(cvv); nu += 1
    tv = np.array(tv); cv = np.array(cv)
    if nu == 0:
        print('%10s | %6d | (无可用边)' % (bname, 0)); continue
    pv = float('nan')
    if wilcoxon is not None:
        try: pv = wilcoxon(cv, tv).pvalue
        except Exception: pv = float('nan')
    print('%10s | %6d | %9.4f | %9.4f | %+8.4f | %10.3g' % (bname, nu, tv.mean(), cv.mean(), cv.mean() - tv.mean(), pv))
print('done', flush=True)
