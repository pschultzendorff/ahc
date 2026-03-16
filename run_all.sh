#!/usr/bin/env bash
# run_all.sh — Run all simulation scripts in SPE10_2D and SPE11_B.
#
# Usage:
#   ./run_all.sh [OPTIONS]
#
# Options:
#   --parallel        Use run_all_layers_parallel_main.py instead of
#                     run_all_layers.py for the all-layers sweep (SPE10_2D).
#                     The parallel script spawns one process per CPU core via
#                     taskset/nohup and returns immediately.
#   --spe10-only      Run only the SPE10_2D scripts.
#   --spe11-only      Run only the SPE11_B scripts.
#   --skip-all-layers Skip the all-layers sweep (run_all_layers*.py).
#   -h, --help        Show this help message and exit.
#
# Scripts run (in order):
#   SPE10_2D:
#     1. run.py                        — main solver comparison
#     2. linear_vs_nonlinear.py        — linear vs. nonlinear constitutive laws
#     3. plot_iterations.py            — iteration-level study (single time step)
#     4. run_for_plotting.py           — small uniform time steps for visualization
#     5. run_all_layers.py             — all-layers sweep (sequential)
#        OR run_all_layers_parallel_main.py (with --parallel)
#   SPE11_B:
#     6. run.py                        — main solver comparison
#     7. run_for_plotting.py           — small uniform time steps for visualization
#     8. convergence_study.py          — estimator convergence study
#        NOTE: run_simulation() calls are skipped by a `continue` statement in
#              the main loop; only the plotting section runs unless edited.
#     9. debug.py                      — debugging / development runs

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PARALLEL=false
SPE10_ONLY=false
SPE11_ONLY=false
SKIP_ALL_LAYERS=false

usage() {
    sed -n '/^# Usage:/,/^[^#]/{ /^[^#]/d; s/^# \{0,3\}//; p }' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)        PARALLEL=true ;;
        --spe10-only)      SPE10_ONLY=true ;;
        --spe11-only)      SPE11_ONLY=true ;;
        --skip-all-layers) SKIP_ALL_LAYERS=true ;;
        -h|--help)         usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
    shift
done

if $SPE10_ONLY && $SPE11_ONLY; then
    echo "Error: --spe10-only and --spe11-only are mutually exclusive." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPE10_DIR="$SCRIPT_DIR/examples/SPE10_2D"
SPE11_DIR="$SCRIPT_DIR/examples/SPE11_B"

run_script() {
    local script="$1"
    local dir
    dir="$(dirname "$script")"
    local name
    name="$(basename "$script")"

    echo ""
    echo "=========================================="
    echo "Running: $script"
    echo "=========================================="
    (cd "$dir" && python "$name")
}

# ---------------------------------------------------------------------------
# SPE10_2D
# ---------------------------------------------------------------------------
if ! $SPE11_ONLY; then
    run_script "$SPE10_DIR/run.py"
    run_script "$SPE10_DIR/linear_vs_nonlinear.py"
    run_script "$SPE10_DIR/plot_iterations.py"
    run_script "$SPE10_DIR/run_for_plotting.py"

    if ! $SKIP_ALL_LAYERS; then
        if $PARALLEL; then
            echo ""
            echo "=========================================="
            echo "Running: $SPE10_DIR/run_all_layers_parallel_main.py"
            echo "(spawns background processes — check logs for completion)"
            echo "=========================================="
            (cd "$SPE10_DIR" && python run_all_layers_parallel_main.py)
        else
            run_script "$SPE10_DIR/run_all_layers.py"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# SPE11_B
# ---------------------------------------------------------------------------
if ! $SPE10_ONLY; then
    run_script "$SPE11_DIR/run.py"
    run_script "$SPE11_DIR/run_for_plotting.py"
    run_script "$SPE11_DIR/convergence_study.py"
    run_script "$SPE11_DIR/debug.py"
fi

echo ""
echo "=========================================="
echo "All scripts finished."
echo "=========================================="
