#!/bin/bash
# Players 2 and 3 (far side, tracked with others hidden) through the same chain as 0 and 1, side by side:
# 3 mm pass from the 6 mm fit, then smooth-sequence poses, then colors (with half turns), then the colored mesh.
L=$(cd "$(dirname "$0")" && pwd); cd $L/paddles
export TRACK_DIR=track_hidden
run() { systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet nice -n 19 $L/venv/bin/python joint.py ../work-smooth run "$@" 2>&1 | grep -E "^\{|Error|Traceback"; }
chain() {
  t=$1
  [ -f run/joint_fine/$t.npz ] || INIT=joint CELL=0.003 SAMPLES=96 JOINT_OUT=joint_fine run $t | sed "s/^/fine p$t /"
  [ -f run/joint_smooth0.1/$t.npz ] || INIT=joint_fine CELL=0.003 SAMPLES=96 STEPS=2000 POSES=smooth W_SMOOTH=0.1 JOINT_OUT=joint_smooth0.1 run $t | sed "s/^/smooth p$t /"
  [ -f run/joint_smooth_color/${t}_colors.npz ] || INIT=joint_smooth0.1 CELL=0.003 SAMPLES=96 STEPS=0 COLOR_STEPS=2400 POSES=smooth W_SMOOTH=0.1 JOINT_OUT=joint_smooth_color run $t | sed "s/^/color p$t /"
  $L/venv/bin/python recolor.py run/joint_smooth_color $t
}
chain 2 & chain 3 & wait
echo "== $(date '+%F %T') far done"
