#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
LOGDIR="results/recoverability/v1/logs"
PIDFILE="$LOGDIR/pipeline.pid"
OUT="$LOGDIR/pipeline.out"
mkdir -p "$LOGDIR"

cmd="${1:-start}"
[ "$cmd" = "--" ] && cmd=start
case "$cmd" in
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    echo "--- last progress lines ($LOGDIR/run_all.log):"
    tail -n 25 "$LOGDIR/run_all.log" 2>/dev/null || echo "(no log yet)"
    exit 0 ;;
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      pid="$(cat "$PIDFILE")"
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
      echo "stopped pid $pid (resume with: bash scripts/remote/run_pipeline.sh)"
    else
      echo "not running"
    fi
    exit 0 ;;
  start) shift || true ;;
  *) ;;
esac
[ "${1:-}" = "--" ] && shift

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE")).  Progress: bash scripts/remote/run_pipeline.sh status"
  exit 1
fi
[ -f "$REPO/.bootstrap_ok" ] || {
  echo "Run the one-time setup first:   bash scripts/remote/bootstrap.sh"; exit 1; }

if ! command -v conda >/dev/null 2>&1; then export PATH="$HOME/miniforge3/bin:$PATH"; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate utrscan

nohup python -u -m analysis.recoverability.run_all "$@" > "$OUT" 2>&1 &
echo $! > "$PIDFILE"
sleep 2
echo "Pipeline started (pid $(cat "$PIDFILE")).  It keeps running after you log out."
echo
echo "  follow live:     tail -f $LOGDIR/run_all.log"
echo "  progress:        bash scripts/remote/run_pipeline.sh status"
echo "  stop / resume:   bash scripts/remote/run_pipeline.sh stop   |   bash scripts/remote/run_pipeline.sh"
echo "  results:         results/recoverability/v1/   (figures/, summary/recoverability_summary.tsv)"
echo
head -n 5 "$OUT" 2>/dev/null || true
