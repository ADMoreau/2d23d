#!/bin/bash
# Smooth-sequence poses for player 0, then the color fit (with half turns) on the smooth poses for players 0 and 1.
L=$(cd "$(dirname "$0")" && pwd); cd $L/paddles
run() { systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet nice -n 19 $L/venv/bin/python joint.py ../work-smooth run "$@"; }
[ -f run/joint_smooth0.1/0.npz ] || INIT=joint_fine CELL=0.003 SAMPLES=96 STEPS=2000 POSES=smooth W_SMOOTH=0.1 JOINT_OUT=joint_smooth0.1 run 0 2>&1 | grep -E "^\{|Error|Traceback" | sed "s/^/smooth p0 /"
for t in 0 1; do
  ( [ -f run/joint_smooth_color/${t}_colors.npz ] || INIT=joint_smooth0.1 CELL=0.003 SAMPLES=96 STEPS=0 COLOR_STEPS=2400 POSES=smooth W_SMOOTH=0.1 JOINT_OUT=joint_smooth_color run $t 2>&1 \
      | grep --line-buffered -E "^\{|Error|Traceback|flips at" | sed -u "s/^/color p$t /"; $L/venv/bin/python recolor.py run/joint_smooth_color $t ) &
done
wait
echo "== $(date '+%F %T') next done"
