#!/bin/bash
# Track every player's paddle, one at a time; rerunning picks up where it stopped.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3
for t in 0 1 2 3; do
  [ -f run/track/$t.npz ] && continue
  echo "== player $t: $(date)"
  nice -n 19 /home/austin/2d23d/.venv/bin/python track_paddle.py ../work-smooth run $t 2>&1 | grep -v -e Warning -e "Loading weights" -e deprecated
done
echo "== all done: $(date)"
