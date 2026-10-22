#!/bin/bash
# The color fit for players 0 and 1, from the 3 mm shape and poses (joint_fine), side by side.
L=$(cd "$(dirname "$0")" && pwd); cd $L/paddles
for t in 0 1; do
  ( [ -f run/joint_flip/${t}_colors.npz ] || INIT=joint_fine CELL=0.003 SAMPLES=96 STEPS=0 COLOR_STEPS=2400 JOINT_OUT=joint_flip \
      systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet nice -n 19 $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 \
      | grep --line-buffered -E "^\{|Error|Traceback|color step (0|500|1000|1500|1999):" | sed -u "s/^/p$t /" ) &
done
wait
for t in 0 1; do $L/venv/bin/python $L/paddles/recolor.py $L/paddles/run/joint_flip $t; done
echo "== $(date '+%F %T') flip done"
