"""零样本 top-k bootstrap 实验第 1 步：按 eval.yml 注释的检查点映射跑 6 组合推理并落盘。

作用:
    按 config/eval.yml 头部注释记录的检查点编号，为 CAFA3 / TALE × BP / CC / MF
    共 6 个数据-本体组合依次构建门控融合模型（model_type=1），以 --only_topk 模式
    复用 evaluate.py 的完整加载与评估代码路径执行零样本 top-k 评估（跳过其余分析），
    并将零样本类上的概率/标签矩阵落盘到 tmp/topk_probs_<数据集>_<本体>_mt1.npz，
    作为 evaluate/zero_shot_topk_stats.py 计算 bootstrap 置信区间与显著性检验的输入。

    检查点映射（与 config/eval.yml 头部注释一致，复现论文数字请勿改动）:
        custom : CAFA3(BP)=97  CAFA3(MF)=93  CAFA3(CC)=51  TALE(BP)=65  TALE(MF)=84  TALE(CC)=63
        proto  : CAFA3(BP)=30  CAFA3(MF)=60  CAFA3(CC)=57  TALE(BP)=55  TALE(MF)=86  TALE(CC)=87

用法（在 project/ppi/new 目录下，synergy 环境）:
    ~/.conda/envs/synergy/bin/python evaluate/run_topk_dump.py --device cuda:0
    # 冒烟/调试：仅跑一个组合
    ~/.conda/envs/synergy/bin/python evaluate/run_topk_dump.py --only CAFA3:MF

删除影响说明:
    本文件是统计实验推理环节的入口脚本。删除后 tmp/*.npz 仍可手动复现（逐组合修改
    eval.yml 的 dataset/namespace/checkpoint_* 后运行 evaluate.py --only_topk），但失去
    一键复现能力；tmp/*.npz 本身可由本脚本随时重新生成，属于可删除的中间产物。
    建议与 zero_shot_topk_stats.py 一并保留。
"""
import argparse
import gc
import importlib.util
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)  # evaluate.py / eval.yml 内部使用相对路径（./data_tale 等）

# (数据集, 本体) -> (checkpoint_custom 编号, checkpoint_proto 编号)，源自 config/eval.yml 注释
CKPT = {
    ('CAFA3', 'MF'): ('checkpoint_93.pth', 'checkpoint_60.pth'),
    ('CAFA3', 'CC'): ('checkpoint_51.pth', 'checkpoint_57.pth'),
    ('CAFA3', 'BP'): ('checkpoint_97.pth', 'checkpoint_30.pth'),
    ('TALE', 'MF'): ('checkpoint_84.pth', 'checkpoint_86.pth'),
    ('TALE', 'CC'): ('checkpoint_63.pth', 'checkpoint_87.pth'),
    ('TALE', 'BP'): ('checkpoint_65.pth', 'checkpoint_55.pth'),
}


def load_evaluate_module():
    """以文件路径加载 evaluate/evaluate.py 入口（与手动运行该脚本的代码路径一致）。"""
    entry = os.path.join(BASE, 'evaluate', 'evaluate.py')
    spec = importlib.util.spec_from_file_location('evaluate_entry', entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description='Dump zero-shot top-k probability matrices for bootstrap stats')
    ap.add_argument('--device', type=str, default='cuda:0', help='device id')
    ap.add_argument('--config', type=str, default='./config/eval.yml', help='base config yml')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--only', type=str, default=None,
                    help='仅跑一个组合，格式 数据集:本体，如 CAFA3:MF（冒烟验证用）')
    args = ap.parse_args()
    args.only_topk = True  # evaluate.main 据此落盘 npz 并跳过其余分析

    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        base_cfg = yaml.load(f)

    combos = list(CKPT.items())
    if args.only:
        ds, ns = args.only.split(':')
        combos = [((ds, ns), CKPT[(ds.upper(), ns.upper())])]

    evaluate_mod = load_evaluate_module()
    for (ds, ns), (ckpt_custom, ckpt_proto) in combos:
        cfg = dict(base_cfg)
        cfg.update({'dataset': ds, 'namespace': ns,
                    'checkpoint_custom': ckpt_custom, 'checkpoint_proto': ckpt_proto})
        print('\n' + '#' * 70, flush=True)
        print(f'##### {ds} / {ns}   custom={ckpt_custom}  proto={ckpt_proto}', flush=True)
        print('#' * 70, flush=True)
        evaluate_mod.main(args, cfg)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == '__main__':
    main()
