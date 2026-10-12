#!/bin/bash
# Assemble the FULL FAMS roots (every task frame, per_frame identity):
#   ProcessedFAMSfull_per_frame          FAMS face AND eyes       (Full Synthetic)
#   ProcessedFAMSfull_per_frame_oldeye   FAMS face, ORIGINAL eyes (Hybrid)
# sharing ProcessedData's full meanno7_clean manifest, exactly like the SimSwap and
# DeepPrivacy2 roots. Fails unless every manifest row has its three crops.
#SBATCH --job-name=fams_fbuild
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/fams_fbuild_%j.log
set -euo pipefail
D=/springbrook/share/eng/esrpxk/datasets
A=$D/ProcessedFAMSfull_per_frame; H=${A}_oldeye; M=meanno7_clean
ln -sfn "$D/ProcessedData/$M" "$A/$M"
/springbrook/share/eng/esrpxk/envs/gaze/bin/python - <<PY
import scipy.io as sio, numpy as np, os, sys
md = sio.loadmat("$D/ProcessedData/$M/metadata.mat")
rec = np.asarray(md["labelRecNum"]).ravel().astype(int); frm = np.asarray(md["frameIndex"]).ravel().astype(int)
miss = [(r, f, s) for r, f in zip(rec, frm) for s in ("appleFace", "appleLeftEye", "appleRightEye")
        if not os.path.isfile(f"$A/{r:05d}/{s}/{f:05d}.jpg")]
print(f"manifest rows {len(rec)}, missing crops {len(miss)}")
if miss: print("FATAL first missing:", miss[:5]); sys.exit(1)
PY
mkdir -p "$H"; ln -sfn "$D/ProcessedData/$M" "$H/$M"
for d in "$A"/000*; do
  r=$(basename "$d"); mkdir -p "$H/$r"
  ln -sfn "$A/$r/appleFace" "$H/$r/appleFace"
  ln -sfn "$D/ProcessedData/$r/appleLeftEye"  "$H/$r/appleLeftEye"
  ln -sfn "$D/ProcessedData/$r/appleRightEye" "$H/$r/appleRightEye"
done
echo BUILD_OK
