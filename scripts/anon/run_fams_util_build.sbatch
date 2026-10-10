#!/bin/bash
# Assemble the FAMS utility roots on the every-10th-frame subsample.
#
#   ProcessedFAMSsub10_per_frame          FAMS face AND eyes      (Full Synthetic)
#   ProcessedFAMSsub10_per_frame_oldeye   FAMS face, ORIGINAL eyes (Hybrid)
#   ProcessedData/meanno7_clean_sub10     the subset manifest
#
# Every arm of the FAMS utility comparison, the real-data baseline included, uses
# meanno7_clean_sub10, so frame count is a stated property of the comparison and
# not an accident (see make_subset_manifest.py for the failure that motivates it).
# Fails unless every manifest row has its three crops in the FAMS root.
#SBATCH --job-name=fams_ubuild
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/fams_ubuild_%j.log
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
A=$D/ProcessedFAMSsub10_per_frame; H=${A}_oldeye; M=meanno7_clean_sub10
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python

$PY scripts/anon/make_subset_manifest.py --root "$A" --src-mean meanno7_clean --out-mean $M
ln -sfn "$D/ProcessedData/$M" "$A/$M"

$PY - <<EOF
import scipy.io as sio, numpy as np, os, sys
md = sio.loadmat("$D/ProcessedData/$M/metadata.mat")
rec = np.asarray(md["labelRecNum"]).ravel().astype(int); frm = np.asarray(md["frameIndex"]).ravel().astype(int)
miss = [(r, f, s) for r, f in zip(rec, frm) for s in ("appleFace", "appleLeftEye", "appleRightEye")
        if not os.path.isfile(f"$A/{r:05d}/{s}/{f:05d}.jpg")]
print(f"subset manifest rows {len(rec)}, per participant ~{len(rec)//len(set(rec))}, missing crops {len(miss)}")
if miss: print("FATAL first missing:", miss[:5]); sys.exit(1)
EOF

mkdir -p "$H"; ln -sfn "$D/ProcessedData/$M" "$H/$M"
for d in "$A"/000*; do
  r=$(basename "$d"); mkdir -p "$H/$r"
  ln -sfn "$A/$r/appleFace" "$H/$r/appleFace"
  ln -sfn "$D/ProcessedData/$r/appleLeftEye"  "$H/$r/appleLeftEye"
  ln -sfn "$D/ProcessedData/$r/appleRightEye" "$H/$r/appleRightEye"
done
echo BUILD_OK
