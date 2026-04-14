#!/usr/bin/env python3
"""Seashell纯TV对照 - 无蒸馏, lambda_tv=0.2"""

import os, sys, subprocess, time, json
import numpy as np

PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)

exp = {
    "name": "seashell_tv_only",
    "config": "experiments/distill/pine_ablation/seashell_tv_only.yaml",
    "output": "experiments/distill/pine_ablation/seashell_tv_only",
}


def evaluate(output_dir, iter_n):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    vp = np.load(
        os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_pred.npy")
    )
    vg = np.load(
        os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_gt.npy")
    )
    if vg.max() > 1.0:
        vp, vg = vp / vg.max(), vg / vg.max()
    psnr = peak_signal_noise_ratio(vg, vp, data_range=1.0)
    ssim = float(
        np.mean(
            [
                structural_similarity(vg[i], vp[i], data_range=1.0)
                for i in range(vg.shape[0])
            ]
        )
    )
    return {"psnr": float(psnr), "ssim": ssim}


os.makedirs(exp["output"], exist_ok=True)
log_file = f"experiments/logs/distill/{exp['name']}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

cmd = [
    sys.executable,
    "train_with_distillation.py",
    "--config",
    exp["config"],
    "--output_dir",
    exp["output"],
]
print(f"开始: {exp['name']} (无蒸馏, tv=0.2)")
start = time.time()
with open(log_file, "w", encoding="utf-8") as log:
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    for line in p.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    p.wait()
elapsed = (time.time() - start) / 60

print(f"\n完成! 用时: {elapsed:.1f}min")
metrics = {}
for it in [2500, 5000, 7500, 10000]:
    try:
        m = evaluate(exp["output"], it)
        metrics[it] = m
        print(f"  Iter {it}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.6f}")
    except:
        pass

final = metrics.get(10000, {})
print(f"\n{'=' * 70}")
print(
    f"TV only(0.2): PSNR={final.get('psnr', 0):.2f}, SSIM={final.get('ssim', 0):.6f}, {elapsed:.1f}min"
)
print(f"Baseline(tv=0.05): PSNR=40.42, SSIM=0.9428, 8min")
print(f"I(distill+tv=0.2): PSNR=39.39, SSIM=0.9442, 15min")
print(f"{'=' * 70}")

with open("experiments/distill/pine_ablation/results_seashell_tv.json", "w") as f:
    json.dump(
        {
            "seashell_tv_only": {
                "psnr": final.get("psnr", 0),
                "ssim": final.get("ssim", 0),
                "time_min": elapsed,
                "all_iters": metrics,
            }
        },
        f,
        indent=2,
    )
