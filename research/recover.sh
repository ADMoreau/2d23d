#!/bin/bash
# Rebuild the paddle experiments after /tmp was wiped (2026-10-20). Each step skips when its output exists.
# Players 0 and 1 first; 2 and 3 after. Every heavy step runs under a memory cap.
set -u
L=$(cd "$(dirname "$0")" && pwd); R=$HOME/2d23d; P=$L/paddles
export HF_HUB_OFFLINE=1
cap() { local mem=$1; shift; systemd-run --user --scope -p MemoryMax=$mem -p MemorySwapMax=0 --quiet nice -n 19 "$@"; }
log() { echo "== $(date '+%F %T') $*"; }

# 1. Smoothed pipeline copy (bodies reused from work/pickleball) and paddle detections, side by side
if [ ! -f $L/work-smooth/check/bodies.npz ]; then
  log "pipeline"
  mkdir -p $L/work-smooth && cp -rn $R/work/pickleball/{ingest,track,bodies} $L/work-smooth/
  (cd $R && OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 cap 10G .venv/bin/2d23d run samples/pickleball.mp4 --work $L/work-smooth -o $L/pickleball-smooth.glb > $L/pipeline.log 2>&1) &
fi
if [ ! -f $P/run/results.json ]; then
  log "detect"
  (cd $R && cap 6G .venv/bin/python $P/detect.py work/pickleball $P/run 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|sam2_video" > $P/detect.log) &
fi
wait
[ -f $L/work-smooth/check/bodies.npz ] && [ -f $P/run/results.json ] || { log "FAILED before fit2"; exit 1; }

# 2. Grips from the detections
cd $P
[ -f run/fit2/grips.npz ] || { log "fit2"; OMP_NUM_THREADS=3 cap 4G $R/.venv/bin/python fit2.py ../work-smooth run > fit2.log 2>&1 || { log "FAILED fit2"; exit 1; }; }

# 3. Tracking, two players at a time
track() { [ -f run/track/$1.npz ] || OMP_NUM_THREADS=3 cap 3G $R/.venv/bin/python track_paddle.py ../work-smooth run $1 2>&1 | grep --line-buffered -v -e Warning -e "Loading weights" -e deprecated | sed -u "s/^/p$1 /"; }
log "track 0 1"; track 0 & track 1 & wait
# 4. Grips on every frame, then shape and poses together
[ -f run/fit3/grips.npz ] || { log "fit3 0 1"; OMP_NUM_THREADS=3 cap 4G $R/.venv/bin/python fit3.py ../work-smooth run 0 1 > fit3_01.log 2>&1; }
for t in 0 1; do [ -f run/joint/$t.npz ] || { log "joint $t"; cap 3G $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 | grep -E "^\{|Error|Traceback" > joint_$t.log; }; done
log "players 0 and 1 done"

# 5. Players 2 and 3: tracked with others hidden
trackh() { [ -f run/track_hidden/$1.npz ] || HIDE_OTHERS=1 TRACK_OUT=track_hidden OMP_NUM_THREADS=3 cap 3G $R/.venv/bin/python track_paddle.py ../work-smooth run $1 2>&1 | grep --line-buffered -v -e Warning -e "Loading weights" -e deprecated | sed -u "s/^/p$1 /"; }
log "track 2 3"; trackh 2 & trackh 3 & wait
for t in 2 3; do grep -q "\"$t\"" run/fit3/report.json 2>/dev/null || { log "fit3 $t"; TRACK_DIR=track_hidden OMP_NUM_THREADS=3 cap 4G $R/.venv/bin/python fit3.py ../work-smooth run $t > fit3_$t.log 2>&1; }; done
for t in 2 3; do [ -f run/joint/$t.npz ] || { log "joint $t"; TRACK_DIR=track_hidden cap 3G $L/venv/bin/python joint.py ../work-smooth run $t 2>&1 | grep -E "^\{|Error|Traceback" > joint_$t.log; }; done
log "all done"
