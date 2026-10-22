#!/bin/bash
# Step 2: the finer pass (3 mm cells) and texture for players 0 and 1, once recover.sh has refit them.
L=$(cd "$(dirname "$0")" && pwd); P=$L/paddles
until grep -q "players 0 and 1 done" $L/recover.log; do sleep 60; done
cd $P
for t in 0 1; do
  [ -f run/joint_fine/$t.npz ] && continue
  echo "== $(date '+%F %T') fine $t"
  INIT=joint CELL=0.003 SAMPLES=96 JOINT_OUT=joint_fine systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet \
    nice -n 19 $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 | grep -E "^\{|Error|Traceback|step (0|1000|2000|2999):"
done
echo "== $(date '+%F %T') step 2 done"
