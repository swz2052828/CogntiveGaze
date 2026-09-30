#!/bin/bash
# Repair ProcessedSwap2: the eye crops of 00020 and 00023 were truncated when the
# group inode quota hit its hard limit on 2026-09-10 (00023/appleRightEye holds
# only frames 10614-16880, 5517 of 11718). The source zips were deleted in the
# storage cleanup; the zip is still on the project Google Drive.
#
# Extracts ONLY the four short folders, never overwrites (unzip -n), verifies
# every folder against ProcessedData, then deletes the zip. Also rebuilds the
# 00023 symlinks of ProcessedSwap2_oldeye, which stopped at the same point.
#SBATCH --job-name=repair_sw2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/repair_sw2_%j.log
set -uo pipefail
RCLONE=/springbrook/home/e/esrpxk/.local/bin/rclone
D=/springbrook/share/eng/esrpxk/datasets
TMP=$D/_repair_swap2
mkdir -p "$TMP"

echo "### connectivity"
timeout 60 $RCLONE lsl "gdrive_root:Experiment 2/datasets" --max-depth 1 2>&1 | grep -q ProcessedSwap2.zip \
  || { echo "FATAL: compute node cannot reach Google Drive"; exit 3; }

echo "### download $(date +%T)"
$RCLONE copy "gdrive_root:Experiment 2/datasets/ProcessedSwap2.zip" "$TMP" --progress=false \
  || { echo "FATAL download"; exit 1; }
ls -l "$TMP/ProcessedSwap2.zip"

echo "### locate internal paths"
unzip -Z1 "$TMP/ProcessedSwap2.zip" | grep -E '00020/appleLeftEye/|00020/appleRightEye/|00023/appleLeftEye/|00023/appleRightEye/' > "$TMP/want.txt"
wc -l < "$TMP/want.txt"
head -2 "$TMP/want.txt"
PREFIX=$(head -1 "$TMP/want.txt" | sed -E 's#(.*/)?(000[0-9]{2})/.*#\1#')
echo "zip prefix: '${PREFIX}'"

echo "### extract $(date +%T)"
mkdir -p "$TMP/x"
unzip -n -q "$TMP/ProcessedSwap2.zip" -d "$TMP/x" '*00020/appleLeftEye/*' '*00020/appleRightEye/*' \
      '*00023/appleLeftEye/*' '*00023/appleRightEye/*' || echo "WARN unzip nonzero"

FAIL=0
for RF in 00020/appleLeftEye 00020/appleRightEye 00023/appleLeftEye 00023/appleRightEye; do
  SRC="$TMP/x/${PREFIX}${RF}"
  DST="$D/ProcessedSwap2/$RF"
  mkdir -p "$DST"
  # move without overwriting anything already in place
  find "$SRC" -maxdepth 1 -name '*.jpg' -exec mv -n -t "$DST" {} + 2>/dev/null
  H=$(ls "$DST" | wc -l); W=$(ls "$D/ProcessedData/$RF" | wc -l)
  echo "  $RF: $H / $W"
  [ "$H" -eq "$W" ] || FAIL=1
done

echo "### ProcessedSwap2_oldeye/00023 symlinks"
O=$D/ProcessedSwap2_oldeye/00023
mkdir -p "$O"
[ -e "$O/appleFace" ]     || ln -s "$D/ProcessedSwap2/00023/appleFace" "$O/appleFace"
[ -e "$O/appleLeftEye" ]  || ln -s "$D/ProcessedData/00023/appleLeftEye" "$O/appleLeftEye"
[ -e "$O/appleRightEye" ] || ln -s "$D/ProcessedData/00023/appleRightEye" "$O/appleRightEye"
ls -l "$O"

if [ $FAIL -eq 0 ]; then
  rm -rf "$TMP"; echo "REPAIR_OK (zip and temp removed)"
else
  echo "REPAIR_INCOMPLETE -- temp kept at $TMP for inspection"
fi
