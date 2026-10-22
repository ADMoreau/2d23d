#!/bin/bash
# Players 2 and 3 again, side by side, with other people hidden from the tracker; rerunning resumes.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 HIDE_OTHERS=1 TRACK_OUT=track_hidden
for t in 2 3; do
  ( [ -f run/track_hidden/$t.npz ] || nice -n 19 /home/austin/2d23d/.venv/bin/python track_paddle.py ../work-smooth run $t 2>&1 \
      | grep --line-buffered -v -e Warning -e "Loading weights" -e deprecated | sed -u "s/^/p$t /" ) &
done
wait
echo "== all done: $(date)"
