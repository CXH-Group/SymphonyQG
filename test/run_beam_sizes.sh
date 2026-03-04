#!/bin/bash
# Submit telerun jobs for different beam sizes and collect results.
# Runs each beam size multiple times and keeps the best (highest QPS) result.
# Usage: ./run_beam_sizes.sh [-n NUM_TRIALS] [beam_sizes...]
# Example: ./run_beam_sizes.sh -n 3 128 256 512

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/bench_sift1m"
TELERUN="$HOME/.local/bin/telerun"
CORES=16

DATA="/home/cxh/telerun/executor/data/sift"
BASE="$DATA/sift_base.fvecs"
QUERY="$DATA/sift_query.fvecs"
GT="$DATA/sift_groundtruth.ivecs"
GRAPH="$DATA/graph-d32"

NUM_TRIALS=3

# Parse flags
while [ $# -gt 0 ]; do
    case "$1" in
        -n) NUM_TRIALS="$2"; shift 2 ;;
        --cores) CORES="$2"; shift 2 ;;
        *) break ;;
    esac
done

BEAM_SIZES=("$@")
if [ ${#BEAM_SIZES[@]} -eq 0 ]; then
    BEAM_SIZES=(128 256 512)
fi

RESULTS_FILE="$PROJECT_DIR/beam_results.csv"
echo "ef,recall,qps,avg_latency_us" > "$RESULTS_FILE"

printf "%-8s %-14s %-10s %-16s %-8s\n" "EF" "Recall@100(%)" "QPS" "Avg Latency(us)" "Trial"
printf "%s\n" "--------------------------------------------------------"

for ef in "${BEAM_SIZES[@]}"; do
    best_qps=0
    best_recall=""
    best_latency=""
    best_trial=0

    for trial in $(seq 1 "$NUM_TRIALS"); do
        output=$("$TELERUN" --override-pending --cores "$CORES" "$BINARY" "$BASE" "$QUERY" "$GT" "$GRAPH" "$ef" 2>&1)

        if echo "$output" | grep -q "Job completed successfully"; then
            line=$(echo "$output" | grep -E "^\s+$ef\s+" | tail -1)
            if [ -n "$line" ]; then
                recall=$(echo "$line" | awk '{print $2}')
                qps=$(echo "$line" | awk '{print $3}')
                latency=$(echo "$line" | awk '{print $4}')
                printf "  %-8s %-14s %-10s %-16s [%d/%d]\n" "$ef" "$recall" "$qps" "$latency" "$trial" "$NUM_TRIALS"

                # Compare QPS (integer comparison)
                if [ "$qps" -gt "$best_qps" ] 2>/dev/null || \
                   awk "BEGIN{exit !($qps > $best_qps)}"; then
                    best_qps="$qps"
                    best_recall="$recall"
                    best_latency="$latency"
                    best_trial="$trial"
                fi
            else
                printf "  EF=%-6s could not parse output [%d/%d]\n" "$ef" "$trial" "$NUM_TRIALS"
            fi
        else
            printf "  EF=%-6s job failed [%d/%d]\n" "$ef" "$trial" "$NUM_TRIALS"
        fi
    done

    if [ "$best_trial" -gt 0 ]; then
        printf "* %-8s %-14s %-10s %-16s (best of %d)\n" "$ef" "$best_recall" "$best_qps" "$best_latency" "$NUM_TRIALS"
        echo "$ef,$best_recall,$best_qps,$best_latency" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "Results saved to $RESULTS_FILE"
