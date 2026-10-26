#!/usr/bin/env bash
# runs every entry of a config on one gpu box, spread over parallel lanes, one process per run.
#   ./run_overnight.sh preflight   checks the box, then runs every config entry for 6 steps
#   ./run_overnight.sh run         runs the batch in the foreground and writes the comparison
#   ./run_overnight.sh start       same as run, detached, so ssh can be closed
#   ./run_overnight.sh status      last progress line of every run
# env: CFG (default configs/stoich_overnight.yaml), OUT (default runs/<config name>), DATA (npz, with
#      baseline_10k_experiments.csv beside it), DEV (cuda), THREADS (per process), LANES (3),
#      DEADLINE (unix time no run may run past; timeouts shrink to fit), AUTO_STOP=1 stops the vast instance after.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"
CFG=${CFG:-configs/stoich_overnight.yaml}
NAME=$(basename "$CFG" .yaml)
OUT=${OUT:-runs/$NAME}
LOGS=${LOGS:-logs/$NAME}
DATA=${DATA:-data/baseline_10k/baseline_10k.npz}
DEV=${DEV:-cuda}
DEVDIR=${DEV%%:*}
THREADS=${THREADS:-4}
LANES=${LANES:-3}
DEADLINE=${DEADLINE:-0}
AUTO_STOP=${AUTO_STOP:-0}

die() { echo "PREFLIGHT FAILED: $*"; exit 1; }

# "name hours" per run; hours is the kill timeout, ~3x a 1200-step run sharing the gpu, longer for symbolic.
run_list() {
    python - "$CFG" <<'EOF'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
for r in spec["runs"]:
    rc = {**spec["defaults"], **r}
    hours = 4 * rc["steps"] / 1200 * (1.5 if rc.get("stoich") == "symbolic" else 1)
    print(rc["name"], max(1, round(hours)))
EOF
}

preflight() {
    echo "== environment"
    python - "$DEV" <<'EOF' || die "python environment"
import sys, torch, kan, yaml, sklearn, pandas
from importlib.metadata import version
print(f"torch {torch.__version__}, pykan {version('pykan')}, cuda build {torch.version.cuda}")
assert version("pykan") == "0.2.8", "the patches in kan_patches.py are written against pykan 0.2.8"
if sys.argv[1] == "cuda":
    assert torch.cuda.is_available(), "torch can't see a gpu (driver too old for this cuda build?)"
    print(f"gpu: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    torch.ones(1, device="cuda").sum().item()
EOF
    echo "== data"
    [ -f "$DATA" ] || die "$DATA not found"
    CSV="$(dirname "$DATA")/baseline_10k_experiments.csv"
    [ -f "$CSV" ] || die "$CSV not found (needed for the 16-species runs)"
    echo "ok: $DATA, $CSV"
    echo "== disk"
    FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
    [ "$FREE_GB" -ge 5 ] || die "only ${FREE_GB} GB free, want 5+"
    echo "ok: ${FREE_GB} GB free"
    echo "== stoich matrix against the data"
    python stoich.py "$DATA" | tee /dev/stderr | grep -q " ok$" || die "stoich check"
    echo "== every run for 6 steps (catches config and data errors before the long runs)"
    rm -rf runs/preflight
    python -u phase1.py --config "$CFG" --data "$DATA" --device $DEV --threads $THREADS \
        --out runs/preflight --steps 6 > logs_preflight.txt 2>&1 \
        || { tail -30 logs_preflight.txt; die "a run failed, full log in logs_preflight.txt"; }
    grep -E "^=== |done:" logs_preflight.txt
    for r in $(run_list | cut -d' ' -f1); do
        [ -f runs/preflight/$DEVDIR/$r/metrics.json ] || die "$r wrote no metrics.json"
    done
    echo "PREFLIGHT OK"
}

lane() {
    while read -r run hours; do
        [ -n "$run" ] || continue
        local secs=$((hours * 3600))
        if [ "$DEADLINE" -gt 0 ]; then
            local left=$((DEADLINE - $(date +%s)))
            [ $left -lt 600 ] && { echo "$(date '+%F %T') SKIPPED $run (deadline)"; continue; }
            [ $left -lt $secs ] && secs=$left
        fi
        echo "$(date '+%F %T') start $run (timeout $((secs / 60)) min)"
        # sigint lets the run stop training and still save its best checkpoint; kill only if saving hangs.
        timeout --signal=INT --kill-after=600 "$secs" \
            python -u phase1.py --config "$CFG" --data "$DATA" --device $DEV --threads $THREADS \
            --out "$OUT" --runs "$run" > "$LOGS/$run.log" 2>&1 < /dev/null
        local rc=$?
        if [ $rc -eq 0 ]; then echo "$(date '+%F %T') done $run"
        elif [ $rc -eq 124 ]; then echo "$(date '+%F %T') TIMEOUT $run"
        else echo "$(date '+%F %T') FAILED $run (exit $rc)"; fi
    done
}

finish() {
    # each run is its own process, so the per-process summary.csv only ever holds one run; compare_runs rebuilds it.
    rm -f "$OUT/$DEVDIR/summary.csv" "$OUT/$DEVDIR/summary.png"
    local dirs=()
    for r in $(run_list | cut -d' ' -f1); do
        [ -f "$OUT/$DEVDIR/$r/metrics.json" ] && dirs+=("$OUT/$DEVDIR/$r")
    done
    [ ${#dirs[@]} -gt 0 ] && python compare_runs.py "${dirs[@]}" > "$LOGS/comparison.md" 2>&1
    echo "$(date '+%F %T') batch finished"
}

batch() {
    mkdir -p "$LOGS"
    local list i
    list=$(run_list) || { echo "could not read $CFG"; exit 1; }
    # longest first, each run to the lane with the fewest hours so far, so no lane ends hours after the rest.
    local assigned
    assigned=$(echo "$list" | sort -k2 -n -r | awk -v n="$LANES" '
        { b = 0; for (i = 1; i < n; i++) if (t[i] < t[b]) b = i; t[b] += $2; print b, $1, $2 }')
    for i in $(seq 0 $((LANES - 1))); do
        echo "$assigned" | awk -v i="$i" '$1 == i { print $2, $3 }' | lane > "$LOGS/lane_$i.txt" 2>&1 &
    done
    wait
    finish > "$LOGS/finish.txt" 2>&1
    cat "$LOGS"/lane_*.txt
    if [ "$AUTO_STOP" = 1 ]; then stop_instance; fi
}

stop_instance() {
    if [ -n "${CONTAINER_ID:-}" ]; then
        echo "$(date '+%F %T') stopping vast instance $CONTAINER_ID"
        pip install -q vastai >/dev/null 2>&1
        vastai stop instance "$CONTAINER_ID" || echo "auto stop failed; stop the instance by hand"
    else
        echo "AUTO_STOP=1 but CONTAINER_ID is not set; stop the instance by hand"
    fi
}

status() {
    cat "$LOGS"/lane_*.txt 2>/dev/null
    for r in $(run_list | cut -d' ' -f1); do
        [ -f "$LOGS/$r.log" ] || continue
        printf '%-18s %s\n' "$r" "$(grep -E "step .*/|early stop|interrupted|non-finite|FAILED|Error" "$LOGS/$r.log" | tail -1)"
    done
    [ -f "$LOGS/finish.txt" ] && cat "$LOGS/finish.txt"
}

case "${1:-}" in
    preflight) preflight ;;
    run) batch ;;
    start)
        mkdir -p "$LOGS"
        export CFG OUT LOGS DATA DEV THREADS LANES DEADLINE AUTO_STOP
        nohup setsid bash "$SELF" run > "$LOGS/batch.txt" 2>&1 < /dev/null &
        echo "started; check with ./run_overnight.sh status (logs in $LOGS/)" ;;
    status) status ;;
    *) sed -n 2,9p "$0"; exit 1 ;;
esac
