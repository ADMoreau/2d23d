#!/bin/bash
# The temporal curve for the grip, players 0 and 1, from the 3 mm fit (joint_fine), side by side.
L=$(cd "$(dirname "$0")" && pwd); cd $L/paddles
for t in 0 1; do
  ( [ -f run/joint_curve/$t.npz ] || INIT=joint_fine CELL=0.003 SAMPLES=96 STEPS=2000 POSES=curve JOINT_OUT=joint_curve \
      systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet nice -n 19 $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 \
      | grep --line-buffered -E "^\{|Error|Traceback|reweighted|^step (0|1000|1999):" | sed -u "s/^/p$t /" ) &
done
wait
echo "== $(date '+%F %T') curve done"
