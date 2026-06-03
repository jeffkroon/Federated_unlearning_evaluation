#!/usr/bin/env bash
# Re-run all 18 adult thesis experiments (6 scenarios × 3 seeds)
# Outputs a Python dict for THESIS_RUNS after completion

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
SCENARIOS=(0 1 4 10 20 34)
SEEDS=(42 43 44)

echo "# Adult thesis runs" > /tmp/adult_run_dirs.txt

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "Seed $SEED"
    for S in "${SCENARIOS[@]}"; do
        echo ""
        echo "  Scenario $S, seed $SEED"

        # Create temp config with this seed
        TMP_CONFIG="mock_etcd/.tmp_adult_seed${SEED}.json"
        python3 -c "
import json
with open('mock_etcd/configuration.json') as f:
    cfg = json.load(f)
cfg['experiment']['type'] = 'adult'
cfg['experiment']['random_seed'] = $SEED
with open('$TMP_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
"
        # Run and capture the output dir
        OUT=$($PYTHON scripts/run_fl.py -S $S -e adult -r 10 -l 5 -c 5 -C $TMP_CONFIG --no-viz 2>&1 | tee /dev/stderr | grep "Results:" | awk '{print $2}')

        if [ -z "$OUT" ]; then
            # Fallback: find the most recent adult dir
            OUT=$(ls -td results/fl_simulation_*_adult_s${S} 2>/dev/null | head -1)
        fi

        echo "seed${SEED}_s${S}: $OUT" >> /tmp/adult_run_dirs.txt
        echo "  => $OUT"

        # Cleanup temp config
        rm -f "$TMP_CONFIG"
    done
done

echo ""
echo "Done. Run dirs saved to /tmp/adult_run_dirs.txt:"
cat /tmp/adult_run_dirs.txt
