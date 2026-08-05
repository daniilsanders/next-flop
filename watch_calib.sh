#!/bin/bash
# Live monitor for the K-chain calibration grid. Exits when all 216 runs land.
#   bash watch_calib.sh
cd "$(dirname "$0")" || exit 1
TOTAL=216

while :; do
  N=$(ls runs/calib/*.json 2>/dev/null | wc -l | tr -d ' ')
  WORKERS=$(pgrep -f spawn_main | wc -l | tr -d ' ')

  # elapsed measured from the oldest completed record
  if [ "$N" -gt 0 ]; then
    T0=$(ls -tr runs/calib/*.json 2>/dev/null | head -1 | xargs stat -f %m)
    NOW=$(date +%s); EL=$((NOW - T0))
    if [ "$N" -gt 1 ] && [ "$EL" -gt 0 ]; then
      ETA=$(( (TOTAL - N) * EL / N / 60 ))
    else ETA="?"; fi
  else EL=0; ETA="?"; fi

  PCT=$((N * 100 / TOTAL))
  BAR=$(printf '%*s' $((PCT / 3)) '' | tr ' ' '#')

  clear
  echo "K-chain calibration grid"
  echo "════════════════════════════════════════════════"
  printf "  %-34s %3d%%\n" "[$BAR]" "$PCT"
  printf "  %d / %d runs   ·   %d workers   ·   %dm elapsed   ·   ~%sm left\n\n" \
         "$N" "$TOTAL" "$WORKERS" "$((EL / 60))" "$ETA"

  echo "  latest:"
  grep -E '^\s+\[' runs/calib_grid.log 2>/dev/null | tail -6 | sed 's/^/  /'

  echo
  echo "  regret so far, by K / epsK (mean over completed h,seeds):"
  python3 - <<'PY' 2>/dev/null
import glob, json, collections
d = collections.defaultdict(list)
for p in glob.glob("runs/calib/*.json"):
    try:
        r = json.load(open(p)); d[(r["K"], r["epsK"])].append(r["regret"])
    except Exception: pass
for k in sorted(d):
    v = d[k]
    print(f"    K={k[0]:<3} epsK={k[1]:<7} n={len(v):<3} mean regret {sum(v)/len(v):.3f}")
PY

  [ "$N" -ge "$TOTAL" ] && { echo; echo "  DONE — all $TOTAL runs complete."; break; }
  sleep 15
done
