"""Script to run ``run_all_layers.py`` in parallel (4 simulations at a time)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dirname = Path(__file__).parent.resolve()

OUTPUT_LOG_ENABLED = False

# List of runscroüt suffixes to run.
suffixes = [0, 1, 2, 3]

cpu_core = -1  # Need to start one below the first core to use.
cpu_skipping = []
for idx, suffix in enumerate(suffixes):
    runfile = dirname / f"run_all_layers_parallel_{suffix}.py"
    output_log = dirname / f"output_run_all_layers_parallel_{suffix}.log"
    cpu_core += 1
    while cpu_core in cpu_skipping:
        cpu_core += 1
    cmd = [
        "n",
        "taskset",
        "-c",
        str(cpu_core),
        "python",
        runfile,
    ]
    if OUTPUT_LOG_ENABLED:
        with output_log.open("w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        logger.info(
            f"Started sweep with {runfile} on CPU core {cpu_core}. Logging to {output_log}."
        )
    else:
        logger.info(f"Started sweep with {runfile} on CPU core {cpu_core}. ")
