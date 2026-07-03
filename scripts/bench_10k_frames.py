"""10,000-frame inference-time protocol (user-specified):
   multistream pipeline_i = facemesh(10k, measured separately on CPU) + backbone_i(10k)
   raw pipeline_j        = raw_model_j(10k)
GPU side: 10,000 sequential batch-1 forwards per model (streaming-faithful)."""
import sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.metacompare import _load_base_checkpoint
from scripts.raw_vs_multistream_eval import load_raw_ckpt, RUNS

N = 10000
device = torch.device("cuda")
TOP5 = ["face_only_mobile_vit", "eyes_only_convnextv2_atto_binocular",
        "eyes_only_convnextv2_binocular", "foveal_vit", "mobile_vit"]
RAWS = {"raw_vit": "meta_pipeline_clean_raw_vit_384",
        "raw_mobile_vit": "meta_pipeline_clean_raw_mobile_vit_384",
        "raw_foveal_vit": "meta_pipeline_clean_raw_foveal_vit_384"}

def time_n(model, ex, n=N):
    with torch.no_grad():
        for _ in range(50): model(*ex)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): model(*ex)
        torch.cuda.synchronize()
    return time.perf_counter() - t0

rows = []
ms_ex = (torch.randn(1,3,224,224,device=device), torch.randn(1,3,224,224,device=device),
         torch.randn(1,3,224,224,device=device), torch.randn(1,625,device=device))
for bb in TOP5:
    ck = RUNS / f"meta_pipeline_clean_metacmp_{bb}/base/seed42/fold0_best_{bb}_gaze_segmenter.pth"
    m, _, _ = _load_base_checkpoint(str(ck), device)
    t = time_n(m, ms_ex)
    rows.append((bb, "multistream", t)); print(f"{bb:40s} {t:8.1f}s /10k", flush=True)
    del m; torch.cuda.empty_cache()
raw_ex = (torch.randn(1,3,384,384,device=device),)
for bb, run in RAWS.items():
    ck = RUNS / f"{run}/base/seed42/fold0_best_{bb}_gaze_segmenter.pth"
    m, _, _ = load_raw_ckpt(str(ck), device)
    t = time_n(m, raw_ex)
    rows.append((bb, "raw", t)); print(f"{bb:40s} {t:8.1f}s /10k", flush=True)
    del m; torch.cuda.empty_cache()
out = Path("/springbrook/share/eng/esrpxk/runs/raw_vs_ms/bench_10k.csv")
with open(out, "w") as fh:
    fh.write("model,kind,seconds_per_10k\n")
    for r in rows: fh.write(f"{r[0]},{r[1]},{r[2]:.2f}\n")
print("wrote", out)
