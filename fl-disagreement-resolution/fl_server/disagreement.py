"""Disagreement resolution for federated learning server."""

import os
import json
import time
from collections import defaultdict

def load_disagreements(etcd_dir):
    """Load the client disagreements JSON from mock_etcd (or FL_DISAGREEMENTS_PATH)."""
    #Per-run isolation: a unique FL_DISAGREEMENTS_PATH (set by parallel runners) overrides
    # the shared mock_etcd/disagreements.json so concurrent runs cannot clobber each other.
    # Unset (the default) preserves the original behaviour exactly.
    disagreements_path = os.environ.get("FL_DISAGREEMENTS_PATH") or os.path.join(etcd_dir, "disagreements.json")
    if not os.path.exists(disagreements_path):
        print(f"No disagreements file found at {disagreements_path}")
        return {}

    try:
        with open(disagreements_path, 'r') as f:
            disagreements = json.load(f)
        print(f"Loaded disagreements: {disagreements}")
        return disagreements
    except Exception as e:
        print(f"Error loading disagreements: {e}")
        return {}

def get_active_disagreements(disagreements, current_round):
    """Keep only the disagreements whose active_rounds window covers current_round."""
    active_disagreements = {}
    expired_disagreements = {}

    if not disagreements:
        print(f"Round {current_round}: No disagreements defined in the system")
        return {}

    print(f"Filtering disagreements for round {current_round}...")

    for client_id, client_disagreements in disagreements.items():
        active_client_disagreements = []
        expired_client_disagreements = []

        for disagreement in client_disagreements:
            start_round = disagreement.get("active_rounds", {}).get("start", 0)
            end_round = disagreement.get("active_rounds", {}).get("end", float('inf'))

            # Check if the disagreement is active in the current round
            if start_round <= current_round and (end_round is None or current_round <= end_round):
                active_client_disagreements.append(disagreement)

                #Log the active disagreement
                target_info = ""
                if "target" in disagreement:
                    target_info = f" with client {disagreement['target']}"

                time_info = ""
                if end_round is not None:
                    time_info = f" (expires after round {end_round})"

                print(f"  Active: Client {client_id} {disagreement.get('type')} disagreement{target_info}{time_info}")
            else:
                #This disagreement has expired or not yet started
                expired_client_disagreements.append(disagreement)

                reason = "not yet started" if current_round < start_round else "expired"
                target_info = ""
                if "target" in disagreement:
                    target_info = f" with client {disagreement['target']}"

                print(f"  Inactive: Client {client_id} {disagreement.get('type')} disagreement{target_info} ({reason})")

        if active_client_disagreements:
            active_disagreements[client_id] = active_client_disagreements

        if expired_client_disagreements:
            expired_disagreements[client_id] = expired_client_disagreements

    if not active_disagreements:
        print(f"Round {current_round}: All disagreements have expired or not yet started")
    else:
        print(f"Round {current_round}: Found {len(active_disagreements)} clients with active disagreements")

    return active_disagreements

def create_model_tracks(active_disagreements, all_client_ids):
    """Turn active disagreements into model tracks (each track = which clients it aggregates)."""
    resolution_start_time = time.time()

    print(f"Creating model tracks for active disagreements: {active_disagreements}")

    # Initial track structures
    tracks = {}
    client_primary_tracks = {}
    client_background_participations = defaultdict(set)

    # Clients involved in disagreements (either as initiators or targets)
    clients_with_disagreements = set()

    # Process and organize disagreements by client
    inbound_exclusions = defaultdict(set)   #client_id -> set of excluded clients
    outbound_exclusions = defaultdict(set)  #client_id -> set of excluded from clients
    bidirectional_exclusions = defaultdict(set)  # client_id -> set of clients with mutual exclusion
    fully_excluded_clients = set()  # Keep for logging; no longer used to drop clients from baseline

    # organize disagreements by type
    for client_id, disagreements in active_disagreements.items():
        #Convert client_id to the ID format used in code (numeric)
        client_id_str = client_id if isinstance(client_id, str) else f"client_{client_id}"
        client_num_id = int(client_id_str.split('_')[1]) if client_id_str.startswith('client_') else int(client_id_str)

        for disagreement in disagreements:
            disagreement_type = disagreement.get("type")

            #Mark this client as involved in disagreements
            clients_with_disagreements.add(client_num_id)

            if disagreement_type == "full":
                # Treat "full" as a request to forget this client post-baseline:
                # mark as fully excluded and add to every other's exclusion pattern
                fully_excluded_clients.add(client_num_id)
                for other_id in all_client_ids:
                    if other_id == client_num_id:
                        continue
                    inbound_exclusions[other_id].add(client_num_id)
                continue

            # For regular disagreement types, process the target
            target = disagreement.get("target")

            #Convert target to numeric ID
            target_str = target if isinstance(target, str) else f"client_{target}"
            target_num_id = int(target_str.split('_')[1]) if target_str.startswith('client_') else int(target_str)

            #Mark target client as involved in disagreements
            clients_with_disagreements.add(target_num_id)

            if disagreement_type == "inbound":
                inbound_exclusions[client_num_id].add(target_num_id)
            elif disagreement_type == "outbound":
                # Client wants to be excluded from target's model
                outbound_exclusions[client_num_id].add(target_num_id)
            elif disagreement_type == "bidirectional":
                # For bidirectional, treat as inbound for both sides
                inbound_exclusions[client_num_id].add(target_num_id)
                inbound_exclusions[target_num_id].add(client_num_id)

                # Also record in bidirectional for reference
                bidirectional_exclusions[client_num_id].add(target_num_id)
                bidirectional_exclusions[target_num_id].add(client_num_id)

    #fold outbound exclusions into the targets' inbound exclusions
    for client_id, target_clients in outbound_exclusions.items():
        for target_id in target_clients:
            #Target should exclude this client from its model
            inbound_exclusions[target_id].add(client_id)

    # Build simplified tracks:
    # - Global track: all clients (baseline/reference)
    # - Retain track: all clients minus fully excluded (if any full exclusions)
    #- Additional pattern tracks: apply other exclusions on top of the retain set

    #Base sets
    retain_clients_base = set(all_client_ids) - fully_excluded_clients if fully_excluded_clients else set(all_client_ids)

    # Create a global track for all clients
    tracks["global"] = set(all_client_ids)

    # retain track for fully excluded clients
    if fully_excluded_clients:
        tracks["track_retain_no_full"] = set(retain_clients_base)
        for cid in retain_clients_base:
            client_primary_tracks[cid] = "track_retain_no_full"
        # fully excluded clients still map to global (baseline reference)
        for cid in fully_excluded_clients:
            client_primary_tracks[cid] = "global"
    else:
        #No full exclusions: default primary will be set below
        pass

    #use inbound exclusions to spot patterns beyond full exclusion
    exclusion_patterns = {}  # pattern -> set of client_ids with that pattern
    for client_id, excluded_set in inbound_exclusions.items():
        # apply exclusions on retain base (all except full-excluded)
        adjusted_excluded = set(excluded_set) | fully_excluded_clients
        pattern = tuple(sorted(adjusted_excluded))
        if pattern not in exclusion_patterns:
            exclusion_patterns[pattern] = set()
        exclusion_patterns[pattern].add(client_id)

    # if it's all full exclusions, just use the retain track
    patterns_to_apply = {}
    for pattern, group in exclusion_patterns.items():
        if set(pattern) == fully_excluded_clients:
            #Handled by retain track
            continue
        patterns_to_apply[pattern] = group

    #Create tracks for each distinct non-full pattern
    for pattern, client_group in patterns_to_apply.items():
        if not pattern:
            continue

        excluded_str = "_".join([f"no{target}" for target in sorted(pattern)])

        if len(client_group) > 1:
            clients_str = "_".join([f"c{cid}" for cid in sorted(client_group)])
            track_name = f"track_{clients_str}_{excluded_str}"
        else:
            client_id = next(iter(client_group))
            track_name = f"track_{client_id}_{excluded_str}"

        # Track members = retain base minus the exclusion pattern
        track_clients = set()
        for cid in retain_clients_base:
            if cid not in pattern:
                track_clients.add(cid)

        tracks[track_name] = track_clients

        # Primary assignments for clients that own this pattern
        for client_id in client_group:
            if client_id in fully_excluded_clients:
                continue
            client_primary_tracks[client_id] = track_name
            # Background: other retain clients not excluded
            for cid in track_clients:
                if cid != client_id:
                    client_background_participations[client_id].add(cid)

    #clients without disagreements go to global, or retain if it exists
    clients_without_disagreements = set(all_client_ids) - clients_with_disagreements
    for client_id in clients_without_disagreements:
        if client_id in fully_excluded_clients:
            client_primary_tracks[client_id] = "global"
        elif "track_retain_no_full" in tracks:
            client_primary_tracks[client_id] = "track_retain_no_full"
        else:
            client_primary_tracks[client_id] = "global"
        print(f"Client {client_id} assigned to {client_primary_tracks[client_id]} (no disagreements)")

    for client_id in all_client_ids:
        if client_id in fully_excluded_clients:
            continue
        if client_id not in client_primary_tracks:
            if "track_retain_no_full" in tracks:
                client_primary_tracks[client_id] = "track_retain_no_full"
            else:
                client_primary_tracks[client_id] = "global"

    if clients_without_disagreements:
        print(f"Clients not involved in disagreements: {clients_without_disagreements}")

    if fully_excluded_clients:
        print(f"Fully excluded clients (note: kept in baseline, will be forgotten via unlearning): {fully_excluded_clients}")

    print(f"\nModel track assignments, {len(tracks)} tracks total:")

    for track_name, track_clients in tracks.items():
        primary_clients = [cid for cid, track in client_primary_tracks.items() if track == track_name]
        print(f"\nTrack '{track_name}':")
        print(f"  Participating clients: {sorted(track_clients)}")
        print(f"  Primary for clients: {sorted(primary_clients)}")

    print("\nClient primary track assignments:")
    for client_id, track_name in sorted(client_primary_tracks.items()):
        print(f"  Client {client_id} -> Primary track: {track_name}")

    print("\nClient background participations:")
    for client_id, bg_tracks in sorted(client_background_participations.items()):
        primary_track = client_primary_tracks.get(client_id, "none")
        print(f"  Client {client_id} (primary: {primary_track}) -> Background participations: {sorted(bg_tracks)}")
    print()

    #Final pass: Identify and consolidate redundant tracks
    # This eliminates tracks that have the exact same members
    track_membership_to_name = {}
    redundant_tracks = []

    # identify redundant tracks
    for track_name, track_members in tracks.items():
        # Create a hashable representation of track membership
        track_key = tuple(sorted(track_members))

        #Skip the global track for special handling
        if track_name == "global":
            track_membership_to_name[track_key] = track_name
            continue

        if track_key in track_membership_to_name:
            canonical_track = track_membership_to_name[track_key]
            redundant_tracks.append((track_name, canonical_track))
            print(f"Redundant track detected: '{track_name}' is identical to '{canonical_track}'")
        else:
            track_membership_to_name[track_key] = track_name

    #Handle special case: track identical to global
    global_key = tuple(sorted(tracks["global"]))
    for track_name, track_members in tracks.items():
        if track_name != "global":
            track_key = tuple(sorted(track_members))
            if track_key == global_key:
                redundant_tracks.append((track_name, "global"))
                print(f"Redundant track detected: '{track_name}' is identical to 'global'")

    for redundant_track, canonical_track in redundant_tracks:
        # Redirect all clients from redundant track to canonical track
        for client_id, track in client_primary_tracks.items():
            if track == redundant_track:
                client_primary_tracks[client_id] = canonical_track
                print(f"Client {client_id} redirected from redundant track '{redundant_track}' to '{canonical_track}'")

        # Remove the redundant track
        if redundant_track in tracks:
            del tracks[redundant_track]

    # stringify client_primary_tracks keys for JSON
    string_client_primary_tracks = {}
    for client_id, track_name in client_primary_tracks.items():
        string_client_primary_tracks[str(client_id)] = track_name

    #Calculate timing metrics
    resolution_time = time.time() - resolution_start_time
    print(f"Disagreement resolution completed in {resolution_time:.4f} seconds")

    return {
        "tracks": tracks,
        "client_tracks": string_client_primary_tracks,
        "client_participations": client_background_participations,
        "timing_metrics": {
            "resolution_time_seconds": resolution_time
        }
    }

def get_track_for_client(client_id, track_info):
    """Primary track name for a client."""
    return track_info.get("client_tracks", {}).get(client_id, "default")

def get_clients_in_track(track_name, track_info):
    """Set of client IDs in a track."""
    return track_info.get("tracks", {}).get(track_name, set())

def get_client_participation_in_tracks(client_id, track_info):
    """All track names a client takes part in."""
    participating_tracks = []

    for track_name, clients in track_info.get("tracks", {}).items():
        if client_id in clients:
            participating_tracks.append(track_name)

    return participating_tracks
