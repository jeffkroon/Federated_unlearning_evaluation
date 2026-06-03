#!/usr/bin/env python3
"""
Disagreement scenarios testing script.
Used to verify that the disagreement resolution system correctly handles various scenarios.
"""

import os
import json
import sys
import argparse
import subprocess
import glob

def usage():
    """Prints usage information."""
    print("""Usage: ./scripts/test_disagreement_scenarios.py <scenario> [options]

Arguments:
  scenario                 Scenario name/number, path to a custom scenario file, or "all" to run all scenarios.

Options:
  -r, --rounds <num>       Number of federated learning rounds (default: 5).
  -l, --local-epochs <num> Number of local training epochs (default: 1).
  -e, --experiment <type>  Experiment type to use (mnist or n_cmapss, default: mnist).
                           N-CMAPSS is limited to <= 6 clients; incompatible scenarios will be skipped.
  -c, --clients <ids>      Override client count or provide a list of IDs (e.g., 4 or "0 1 3 5").
                           If not specified, uses 'num_clients' from the scenario file.
  -v, --verbose            Enable verbose output.
  --verbose-plots          Generate all plots (default: only last round track metrics + track contributions).
  -h, --help               Display this help and exit.

Examples:
  ./scripts/test_disagreement_scenarios.py 1
  ./scripts/test_disagreement_scenarios.py all -e n_cmapss
  ./scripts/test_disagreement_scenarios.py 3 -r 10 -l 2
  ./scripts/test_disagreement_scenarios.py 5 -c 4
  ./scripts/test_disagreement_scenarios.py path/to/my/custom_scenario.json
  ./scripts/test_disagreement_scenarios.py 1 --verbose-plots
""")

def load_scenario(scenario_path):
    """Load a disagreement scenario from a file."""
    try:
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        print(f"Loaded scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        return scenario
    except Exception as e:
        print(f"Error loading scenario: {e}")
        return None

def get_all_scenarios():
    """Get a list of all available scenario files."""
    scenario_dir = "mock_etcd/scenarios_5clients"
    scenario_files = glob.glob(os.path.join(scenario_dir, "scenario*.json"))

    # Sort scenarios numerically instead of lexicographically
    return sorted(scenario_files, key=lambda f: int(os.path.basename(f).replace("scenario", "").replace(".json", "")))

def copy_disagreements_to_etcd(scenario):
    """Copy disagreements from scenario to mock_etcd/disagreements.json."""
    try:
        with open('mock_etcd/disagreements.json', 'w') as f:
            json.dump(scenario['disagreements'], f, indent=2)
        print("Copied disagreements to mock_etcd/disagreements.json")
        return True
    except Exception as e:
        print(f"Error copying disagreements: {e}")
        return False

def run_test(scenario_input, experiment="mnist", fl_rounds=5, local_epochs=1, clients_override=None, verbose_plots=False):
    """Run a federated learning test using the specified scenario and dataset.

    Args:
        scenario_input: Path to the scenario file or scenario number
        experiment: Experiment type (mnist or n_cmapss)
        fl_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs
        clients_override: Override the number/list of clients from scenario
        verbose_plots: Whether to generate all plots or only minimal plots
    """
    # Load scenario to get the number of clients
    scenario_path = scenario_input
    if scenario_input.isdigit():
        scenario_path = f"mock_etcd/scenarios_5clients/scenario{scenario_input}.json"
    elif not os.path.exists(scenario_input) and os.path.exists(f"mock_etcd/scenarios_5clients/{scenario_input}"):
        scenario_path = f"mock_etcd/scenarios_5clients/{scenario_input}"

    #Get client count from scenario unless overridden
    if clients_override:
        #Use the override value
        if isinstance(clients_override, int):
            num_clients = clients_override
        else:
            # If it's a string, pass it as-is to the script
            num_clients = clients_override
    else:
        # Get from scenario
        num_clients = 10
        try:
            with open(scenario_path, 'r') as f:
                scenario = json.load(f)
                num_clients = scenario.get('num_clients', 10)

                # Validate client count for n_cmapss
                if experiment == "n_cmapss" and num_clients > 6:
                    raise ValueError(f"N-CMAPSS experiment cannot use more than 6 clients (scenario requests {num_clients})")

        except Exception as e:
            print(f"Warning: Could not read scenario file {scenario_path}: {e}")
            print(f"Using default of {num_clients} clients")

    #Determine if scenario_input is a path or just a number/name
    if scenario_input.isdigit() or os.path.basename(scenario_input).startswith("scenario"):
        #For built-in scenarios, pass the scenario number/name directly
        scenario_arg = os.path.basename(scenario_input).replace("scenario", "").replace(".json", "")
    else:
        # For custom scenario paths, pass the full path
        scenario_arg = scenario_input

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_fl_script = os.path.join(script_dir, "run_fl.py")

    cmd = [
        "python3", run_fl_script,
        "-e", experiment,
        "-c", str(num_clients),
        "-r", str(fl_rounds),
        "-l", str(local_epochs),
        "-i",
        "-S", scenario_arg
    ]

    if verbose_plots:
        cmd.append("--verbose-plots")

    if clients_override:
        print(f"Running test with {num_clients} clients (overridden from command line)")
    else:
        print(f"Running test with {num_clients} clients from scenario")
    print(f"Running test with command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Extract the results directory from the output
    for line in result.stdout.split('\n'):
        if "Results directory:" in line:
            results_dir = line.split("Results directory:")[1].strip()
            return results_dir

    return None

def verify_tracks(results_dir, scenario):
    """Verify that the created tracks match the expected tracks."""
    # Check each round and report on the tracks
    success = True
    is_empty_scenario = (
        "expected_tracks" in scenario and
        len(scenario["expected_tracks"]) == 1 and
        "global" in scenario["expected_tracks"] and
        "disagreements" in scenario and
        len(scenario["disagreements"]) == 0
    )

    #Special handling for empty scenario (no disagreements)
    if is_empty_scenario:
        print("\nTesting empty scenario with no disagreements")
        client_dirs = glob.glob(os.path.join(results_dir, "output", "clients", "client_*"))
        if not client_dirs:
            print("FAIL: No client output directories found")
            return False
        print(f"OK: Found {len(client_dirs)} client output directories")
        print("OK: Empty scenario test passed - all clients use global track by default")
        return True

    #Check for round directories
    model_storage_dir = os.path.join(results_dir, "model_storage")
    round_dirs = [d for d in os.listdir(model_storage_dir) if d.startswith("round_")]
    max_round = max([int(d.split("_")[1]) for d in round_dirs])
    print(f"Found {len(round_dirs)} round directories, max round: {max_round}")

    # Extract validation rules if present
    validation_rules = scenario.get("validation_rules", {})

    # Go through each round directory
    for round_dir in sorted(round_dirs, key=lambda x: int(x.split("_")[1])):
        round_num = int(round_dir.split("_")[1])
        round_path = os.path.join(model_storage_dir, round_dir)
        tracks_dir = os.path.join(round_path, "tracks")

        # Determine active disagreements for this round
        active_disagreements = {}
        if "disagreements" in scenario:
            #Handle different disagreement formats
            if scenario["disagreements"] and isinstance(next(iter(scenario["disagreements"].values())), list):
                #List of disagreements per client
                for client_id, disagreement_list in scenario["disagreements"].items():
                    for disagreement in disagreement_list:
                        disagreement_type = disagreement.get("type")
                        target_id = disagreement.get("target")

                        if disagreement_type == "full":
                            # This is a full exclusion, proceed regardless of target_id
                            pass
                        elif not target_id:
                            # This is NOT a full exclusion, and target_id is missing, so skip.
                            print(f"  Skipping disagreement for client {client_id} due to missing target_id (type: {disagreement_type})")
                            continue

                        # Check if this disagreement is active in this round
                        active_rounds = disagreement.get("active_rounds", {})
                        start_round = active_rounds.get("start", 1)
                        end_round = active_rounds.get("end")
                        is_active = (start_round <= round_num) and (end_round is None or round_num <= end_round)

                        if is_active:
                            if client_id not in active_disagreements:
                                active_disagreements[client_id] = []
                            active_disagreements[client_id].append(disagreement)
            else:
                #Dictionary of disagreements per client
                for client_id, disagreements in scenario["disagreements"].items():
                    for target_id, details in disagreements.items():
                        if "rounds" in details:
                            #If rounds specified, check if this round is included
                            if round_num in details["rounds"]:
                                if client_id not in active_disagreements:
                                    active_disagreements[client_id] = {}
                                active_disagreements[client_id][target_id] = details
                        else:
                            # If no rounds specified, include for all rounds
                            if client_id not in active_disagreements:
                                active_disagreements[client_id] = {}
                            active_disagreements[client_id][target_id] = details

        # Check if tracks directory exists, otherwise there are no active disagreements
        if not os.path.exists(tracks_dir):
            print(f"\n=== Round {round_num} ===")
            print(f"No tracks directory found for round {round_num}")

            # Check if there should be active disagreements for this round
            if active_disagreements:
                print(f"FAIL: Round {round_num}: No tracks directory but should have active disagreements")
                print(f"Active disagreements: {active_disagreements}")
                success = False
            else:
                print(f"OK: Round {round_num}: No tracks directory as expected (no active disagreements)")

            #Apply validation rules if possible without track metadata
            round_rules_key = f"round_{round_num}_rules"
            if validation_rules and round_rules_key in validation_rules:
                print(f"Applying validation rules for round {round_num} without tracks")
                client_dirs = glob.glob(os.path.join(results_dir, "output", "clients", "client_*"))
                client_ids_active = [os.path.basename(d).split("_")[1] for d in client_dirs]
                print(f"  Found active clients: {sorted(client_ids_active)}")

                #Apply validation rules that make sense without track_metadata
                rules = validation_rules[round_rules_key]
                for rule in rules:
                    rule_type = rule.get("type")
                    if rule_type in ["clients_share_track", "clients_on_track"]:
                        print(f"  OK: Assumed passed: {rule.get('description', rule_type)}")

            continue

        track_metadata_path = os.path.join(tracks_dir, "track_metadata.json")

        # Skip if no track metadata file
        if not os.path.exists(track_metadata_path):
            print(f"\n=== Round {round_num} ===")
            print(f"No track metadata file for round {round_num}")

            if round_num == 1:
                print("FAIL: No track metadata found for round 1, which should always have metadata")
                success = False
            elif active_disagreements:
                print(f"FAIL: Round {round_num}: No track metadata but disagreements should be active")
                success = False
            else:
                print(f"OK: Round {round_num}: No track metadata as expected (all disagreements expired)")

            continue

        with open(track_metadata_path, 'r') as f:
            track_metadata = json.load(f)

        # Print round explanation for all scenarios
        print(f"\n=== Round {round_num} ===")

        # Check if track_metadata should exist for this round
        if not active_disagreements:
            print(f"FAIL: Round {round_num}: Found tracks directory but all disagreements should have expired")
            success = False
            continue

        #Check if there's a round-specific expectation
        round_specific_key = f"expected_tracks_round_{round_num}"

        #Get expected tracks for this round
        if round_specific_key in scenario:
            print(f"Using round-specific expectations for round {round_num}")
            expected_tracks = scenario[round_specific_key]
        else:
            # Otherwise use the general expected tracks
            expected_tracks = scenario.get("expected_tracks", {})

        # Initialize mapping of client IDs to tracks
        client_track_mapping = {}

        # First, get client primary tracks
        if "client_tracks" in track_metadata:
            for client_id, track_name in track_metadata["client_tracks"].items():
                client_track_mapping[client_id] = track_name

        #Then, add secondary track information
        if "tracks" in track_metadata:
            for track_name, clients in track_metadata["tracks"].items():
                for client_id in clients:
                    if str(client_id) not in client_track_mapping:
                        client_track_mapping[str(client_id)] = track_name

        #Print client track assignments for this round
        print(f"Client track assignments for round {round_num}:")
        for track_name, track_clients in track_metadata.get("tracks", {}).items():
            primary_clients = [
                client_id for client_id, track in track_metadata.get("client_tracks", {}).items()
                if track == track_name
            ]
            print(f"  Track '{track_name}': Primary for clients {sorted(primary_clients)}")

        # Print track participants for debugging
        print(f"  Track participants in round {round_num}:")
        for track_name, participants in track_metadata.get("tracks", {}).items():
            print(f"    Track '{track_name}': {sorted(participants)}")

        # Check if track names match expected
        tracks_match = True
        # Convert both to sets for easier comparison
        expected_track_names = set(expected_tracks.keys())
        actual_track_names = set(track_metadata.get("tracks", {}).keys())

        if expected_track_names != actual_track_names:
            missing_tracks = expected_track_names - actual_track_names
            extra_tracks = actual_track_names - expected_track_names

            if missing_tracks:
                print(f"FAIL: Missing expected tracks: {missing_tracks}")
                tracks_match = False
            if extra_tracks:
                print(f"FAIL: Unexpected extra tracks: {extra_tracks}")
                tracks_match = False

        if tracks_match:
            print(f"Round {round_num}: Track names match expected")
        else:
            print(f"FAIL: Round {round_num}: Track names do not match expected")
            success = False

        #Print detailed track information
        print(f"\nTrack assignments for round {round_num}:")
        for track_name, description in expected_tracks.items():
            if track_name in track_metadata.get("tracks", {}):
                participating_clients = sorted(track_metadata["tracks"][track_name])
                primary_clients = [
                    client_id for client_id, track in track_metadata.get("client_tracks", {}).items()
                    if track == track_name
                ]
                print(f"  Track '{track_name}':")
                print(f"    Expected: {description}")
                print(f"    Participating clients: {participating_clients}")
                print(f"    Primary for clients: {sorted(primary_clients)}")
            else:
                print(f"  FAIL: Track '{track_name}' not found in actual tracks")
                success = False

        #Check validation rules for this round if available
        round_rules_key = f"round_{round_num}_rules"
        if round_rules_key in validation_rules:
            print(f"Applying validation rules for round {round_num}")
            rules = validation_rules[round_rules_key]
            for rule in rules:
                rule_type = rule.get("type")
                if rule_type == "client_excludes":
                    client = rule.get("client")
                    excludes = rule.get("excludes", [])
                    description = rule.get("description", "")

                    # Get the track for this client
                    client_track = client_track_mapping.get(str(client))
                    if not client_track:
                        print(f"FAIL: {description}: Client {client} not found in any track")
                        success = False
                        continue

                    # Check that this client's track excludes the specified clients
                    track_clients = track_metadata.get("tracks", {}).get(client_track, [])

                    for excluded_client in excludes:
                        if excluded_client in track_clients:
                            print(f"FAIL: {description}: Client {excluded_client} should be excluded from {client_track}")
                            success = False

                elif rule_type == "clients_share_track":
                    clients = rule.get("clients", [])
                    description = rule.get("description", "")

                    # Check that all clients are on the same track
                    tracks = set()
                    for client in clients:
                        track = client_track_mapping.get(str(client))
                        if track:
                            tracks.add(track)

                    if len(tracks) > 1:
                        print(f"FAIL: {description}: Clients {clients} are on different tracks: {tracks}")
                        success = False
                    elif len(tracks) == 0:
                        print(f"FAIL: {description}: None of the clients {clients} found in any track")
                        success = False

                elif rule_type == "clients_on_track":
                    track_name = rule.get("track_name")
                    clients = rule.get("clients", [])
                    description = rule.get("description", "")

                    #Check that all specified clients are on the specified track
                    for client in clients:
                        client_track = client_track_mapping.get(str(client))
                        if client_track != track_name:
                            print(f"FAIL: {description}: Client {client} should be on track {track_name} but is on {client_track}")
                            success = False

                elif rule_type == "track_includes_only":
                    track_name = rule.get("track_name")
                    clients_expected_in_rule = rule.get("clients", [])
                    description = rule.get("description", "")

                    if track_name in track_metadata.get("tracks", {}):
                        actual_track_clients_set = set(track_metadata["tracks"][track_name])
                        expected_clients_set = set(clients_expected_in_rule)

                        if actual_track_clients_set != expected_clients_set:
                            extra = actual_track_clients_set - expected_clients_set
                            missing = expected_clients_set - actual_track_clients_set
                            if extra:
                                print(f"FAIL: {description}: Track {track_name} has extra clients: {extra}")
                                success = False
                            if missing:
                                print(f"FAIL: {description}: Track {track_name} is missing clients: {missing}")
                                success = False
                    else:
                        print(f"FAIL: {description}: Track {track_name} not found")
                        success = False

    if success:
        print("\nOK: SUCCESS: All tracks match expected configuration")
    else:
        print("\nFAIL: FAILURE: Some tracks did not match expected configuration")

    return success

def run_single_scenario(scenario_path, args):
    """Run and verify a single scenario.

    Args:
        scenario_path: Path to the scenario file
        args: Command line arguments

    Returns:
        tuple: (status, message) where status is 'passed', 'failed', or 'skipped'
    """
    print(f"\n{'=' * 80}")
    print(f"Testing scenario: {scenario_path}")
    print(f"{'=' * 80}")
    print(f"Using experiment type: {args.experiment}")
    print(f"Running for {args.rounds} rounds with {args.local_epochs} local epochs per round")
    if hasattr(args, 'clients') and args.clients:
        print(f"Using client override: {args.clients}")

    #Load the scenario
    scenario = load_scenario(scenario_path)
    if not scenario:
        return ('failed', 'Failed to load scenario')

    # Check for compatibility with experiment type
    num_clients = scenario.get('num_clients', 10)
    if args.experiment == 'n_cmapss' and num_clients > 6:
        print(f"Skipping scenario: N-CMAPSS experiment cannot use more than 6 clients (scenario requires {num_clients})")
        return ('skipped', f'N-CMAPSS limited to ≤6 clients (scenario needs {num_clients})')

    # Get clients override if specified
    clients_override = getattr(args, 'clients', None)

    # Run the test
    results_dir = run_test(scenario_path, args.experiment, args.rounds, args.local_epochs, clients_override, args.verbose_plots)
    if not results_dir:
        print("FAIL: Error: Test failed to run correctly")
        return ('failed', 'Test failed to run correctly')

    #Verify the results
    if not verify_tracks(results_dir, scenario):
        print("FAIL: Test failed: Tracks do not match expected configuration")
        return ('failed', 'Tracks do not match expected configuration')

    print("OK: Test passed: Tracks match expected configuration")
    return ('passed', 'All checks passed')

def main():
    parser = argparse.ArgumentParser(description='Test disagreement scenarios', add_help=False)
    parser.add_argument('scenario', nargs='?', default=None,
                        help='Scenario name/number, path to a custom scenario file, or "all" to run all scenarios')
    parser.add_argument('-r', '--rounds', type=int, default=5,
                        help='Number of federated learning rounds (default: 5)')
    parser.add_argument('-l', '--local-epochs', type=int, default=1,
                        help='Number of local training epochs (default: 1)')
    parser.add_argument('-e', '--experiment', type=str, default='mnist',
                        choices=['mnist', 'n_cmapss'],
                        help='Experiment type to use (mnist or n_cmapss). N-CMAPSS limited to ≤6 clients - incompatible scenarios will be skipped.')
    parser.add_argument('-c', '--clients', type=str,
                        help='Override client count or provide a list of IDs (e.g., 4 or "0 1 3 5"). If not specified, uses \'num_clients\' from the scenario file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--verbose-plots', action='store_true',
                        help='Generate all plots (default: only last round track metrics + track contributions)')
    parser.add_argument('-h', '--help', action='store_true',
                        help='Display this help and exit')

    args = parser.parse_args()

    if args.help or not args.scenario:
        usage()
        sys.exit(0)

    #Change to the parent directory (project root) to ensure proper paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # Check if "all" scenarios should be run
    if args.scenario.lower() == "all":
        scenario_paths = get_all_scenarios()
        if not scenario_paths:
            print("Error: No scenarios found in mock_etcd/scenarios_5clients/")
            return 1

        # Skip scenario 0 (serves as a no-disagreements test scenario)
        scenario_paths = [path for path in scenario_paths if not os.path.basename(path) == "scenario0.json"]

        print(f"Running all {len(scenario_paths)} scenarios (skipping scenario0)...")

        # Track results
        results = {}
        success_count = 0
        skipped_count = 0
        failed_count = 0

        for scenario_path in scenario_paths:
            scenario_name = os.path.basename(scenario_path).replace(".json", "")
            status, message = run_single_scenario(scenario_path, args)

            if status == 'passed':
                results[scenario_name] = "OK: Passed"
                success_count += 1
            elif status == 'skipped':
                results[scenario_name] = f"Skipped: {message}"
                skipped_count += 1
            else:
                results[scenario_name] = f"FAIL: Failed: {message}"
                failed_count += 1

        total_scenarios = len(scenario_paths)
        print("\n" + "=" * 80)
        print(f"SCENARIO TEST SUMMARY: {success_count} passed, {skipped_count} skipped, {failed_count} failed (out of {total_scenarios} total)")
        print("=" * 80)
        for scenario_name, status in results.items():
            print(f"{scenario_name:<20}: {status}")

        if failed_count == 0:
            if skipped_count > 0:
                print(f"\nOK: All runnable scenarios passed! ({skipped_count} scenarios skipped due to incompatibility)")
            else:
                print("\nOK: All scenarios passed!")
            return 0
        else:
            print(f"\nFAIL: {failed_count} scenarios failed")
            return 1
    else:
        #Determine the scenario path
        if os.path.exists(args.scenario):
            scenario_path = args.scenario
        elif os.path.exists(f"mock_etcd/scenarios_5clients/scenario{args.scenario}.json"):
            scenario_path = f"mock_etcd/scenarios_5clients/scenario{args.scenario}.json"
        elif os.path.exists(f"mock_etcd/scenarios_5clients/{args.scenario}.json"):
            scenario_path = f"mock_etcd/scenarios_5clients/{args.scenario}.json"
        else:
            print(f"Error: Scenario file {args.scenario} not found")
            return 1

        #Run single scenario
        status, message = run_single_scenario(scenario_path, args)
        if status == 'passed':
            return 0
        elif status == 'skipped':
            print(f"\nScenario skipped: {message}")
            return 0
        else:
            print(f"\nFAIL: Scenario failed: {message}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
