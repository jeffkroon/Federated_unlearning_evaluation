"""Results index for tracking and querying FL experiment runs."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

DEFAULT_INDEX_PATH = "results/index.json"


def load_index(index_path=DEFAULT_INDEX_PATH):
    """Load the results index from disk.

    Returns:
        list[dict]: List of run entries, or empty list if index doesn't exist.
    """
    if not os.path.exists(index_path):
        return []
    with open(index_path, 'r') as f:
        return json.load(f)


def save_index(entries, index_path=DEFAULT_INDEX_PATH):
    """Save the results index to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(entries, f, indent=2)


def _next_id(entries):
    """Get the next available ID."""
    if not entries:
        return 1
    return max(e["id"] for e in entries) + 1


def register_run(results_dir, metadata=None, index_path=DEFAULT_INDEX_PATH):
    """Register a completed run in the index.

    Args:
        results_dir: Path to the results directory.
        metadata: Optional metadata dict (from run_metadata.json).
        index_path: Path to the index file.

    Returns:
        dict: The created index entry.
    """
    entries = load_index(index_path)

    #Avoid duplicate entries for the same directory
    for e in entries:
        if e.get("dir") == results_dir:
            return e

    entry = _build_entry(_next_id(entries), results_dir, metadata)
    entries.append(entry)
    save_index(entries, index_path)
    return entry


def rebuild_index(results_base="results", index_path=DEFAULT_INDEX_PATH):
    """Rebuild the index from existing result directories.

    Scans all fl_simulation_* directories, reading run_metadata.json where
    available and parsing directory names as fallback.

    Args:
        results_base: Base results directory to scan.
        index_path: Path to the index file.

    Returns:
        list[dict]: The rebuilt index entries.
    """
    results_path = Path(results_base)
    if not results_path.exists():
        return []

    sim_dirs = sorted(
        results_path.glob("fl_simulation_*"),
        key=lambda p: p.name
    )

    entries = []
    for i, sim_dir in enumerate(sim_dirs, start=1):
        if not sim_dir.is_dir():
            continue

        #Try to load run_metadata.json
        metadata = None
        metadata_path = sim_dir / "run_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass

        entry = _build_entry(i, str(sim_dir), metadata)
        entries.append(entry)

    save_index(entries, index_path)
    return entries


def query_runs(experiment=None, scenario=None, last_n=None, index_path=DEFAULT_INDEX_PATH):
    """Filter runs from the index.

    Args:
        experiment: Filter by experiment type (e.g. "mnist").
        scenario: Filter by scenario number.
        last_n: Return only the last N entries.
        index_path: Path to the index file.

    Returns:
        list[dict]: Filtered index entries.
    """
    entries = load_index(index_path)

    if experiment:
        entries = [e for e in entries if e.get("experiment_type") == experiment]

    if scenario is not None:
        s = str(scenario)
        entries = [e for e in entries if str(e.get("scenario", "")) == s]

    if last_n:
        entries = entries[-last_n:]

    return entries


def _build_entry(entry_id, results_dir, metadata=None):
    """Build an index entry from a results directory.

    Uses metadata if available, falls back to parsing the directory name.
    """
    entry = {
        "id": entry_id,
        "dir": results_dir,
    }

    if metadata:
        entry["timestamp"] = metadata.get("timestamp", "")
        entry["experiment_type"] = metadata.get("experiment_type", "")
        entry["scenario"] = metadata.get("scenario")
        entry["num_clients"] = metadata.get("num_clients", 0)
        entry["fl_rounds"] = metadata.get("fl_rounds", 0)
        entry["unlearning_enabled"] = metadata.get("unlearning", {}).get("enabled", False)
    else:
        # Parse from directory name
        dirname = os.path.basename(results_dir)
        entry.update(_parse_dirname(dirname))

        # Try to enrich from consolidated_results.json
        consolidated = os.path.join(results_dir, "consolidated_results.json")
        if os.path.exists(consolidated):
            try:
                with open(consolidated, 'r') as f:
                    cr = json.load(f)
                if cr.get("experiment_type"):
                    entry["experiment_type"] = cr["experiment_type"]
                if cr.get("fl_rounds"):
                    entry["fl_rounds"] = cr["fl_rounds"]
                if cr.get("scenario") is not None:
                    raw = str(cr["scenario"])
                    # Strip 'scenario' prefix if present (e.g. 'scenario10' -> '10')
                    raw = re.sub(r"^scenario", "", raw)
                    try:
                        entry["scenario"] = int(raw)
                    except (ValueError, TypeError):
                        entry["scenario"] = raw
                #Check if unlearning strategies exist beyond original_model
                strats = cr.get("strategies", {})
                entry["unlearning_enabled"] = any(
                    k != "original_model" for k in strats
                )
            except Exception:
                pass

    entry["status"] = "completed"
    return entry


#Pattern: fl_simulation_YYYYMMDD_HHMMSS_<experiment_type>[_s<scenario>][_s<scenario>_<model>_<strategy>]
_DIR_PATTERN = re.compile(
    r"fl_simulation_(\d{8}_\d{6})_(\w+?)(?:_s(\w+))?$"
)


def _parse_dirname(dirname):
    """Extract metadata from a results directory name."""
    info = {
        "timestamp": "",
        "experiment_type": "",
        "scenario": None,
        "num_clients": 0,
        "fl_rounds": 0,
        "unlearning_enabled": False,
    }

    m = _DIR_PATTERN.match(dirname)
    if m:
        ts_str = m.group(1)
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            info["timestamp"] = dt.isoformat()
        except ValueError:
            info["timestamp"] = ts_str
        info["experiment_type"] = m.group(2)
        if m.group(3):
            # Try to extract just the scenario number
            scenario_raw = m.group(3)
            # Handle formats like "scenario1" or just "1"
            scenario_num = re.sub(r"^scenario", "", scenario_raw)
            try:
                info["scenario"] = int(scenario_num)
            except ValueError:
                info["scenario"] = scenario_raw

    return info
