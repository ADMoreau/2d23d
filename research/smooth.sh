#!/bin/bash
# The smooth-sequence poses for player 1 at two smoothing weights, side by side, from the 3 mm fit (joint_fine).
L=$(cd "$(dirname "$0")" && pwd); cd $L/paddles
t=${1:-1}
for w in 0.1 1.0; do
  ( [ -f run/joint_smooth$w/$t.npz ] || INIT=joint_fine CELL=0.003 SAMPLES=96 STEPS=2000 POSES=smooth W_SMOOTH=$w JOINT_OUT=joint_smooth$w \
      systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet nice -n 19 $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 \
      | grep --line-buffered -E "^\{|Error|Traceback|reweighted" | sed -u "s/^/w$w /" ) &
done
wait
echo "== $(date '+%F %T') smooth done"
