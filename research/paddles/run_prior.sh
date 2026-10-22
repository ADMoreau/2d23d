#!/bin/bash
# prior.py in the TripoSR venv. usage: run_prior.sh <track> [frames]
S=$(cd "$(dirname "$0")" && pwd); I=$S/../i23d
exec "$I/venv/bin/python" -I -c "import sys, runpy; sys.path[:0] = ['$I/src', '$I/shim']; sys.argv = ['prior.py', '$S/../work-smooth', '$S/run'] + sys.argv[1:]; runpy.run_path('$S/prior.py', run_name='__main__')" "$@"
