"""Disagreement resolution for federated learning server."""

import os
import json
import time
from collections import defaultdict

def load_disagreements(etcd_dir):
    """Load client disagreements from the mock_etcd directory.

    Args:
        etcd_dir: Path to the mock_etcd directory

    Returns:
        dict: Dictionary mapping client IDs to their disagreements
    """
    disagreements_path = os.path.join(etcd_dir, "disagreements.json")
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
    """Filter disagreements to only those active in the current round.

    Args:
        disagreements: Dictionary of all client disagreements
        current_round: Current federated learning round

    Returns:
        dict: Dictionary of active disagreements for the current round
    """
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
    """Create model tracks based on active disagreements.

    Args:
        active_disagreements: Dictionary of active disagreements
        all_client_ids: List of all client IDs in the federation

    Returns:
        dict: Dictionary mapping track IDs to sets of client IDs
    """
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
    fully_excluded_clients = set()  # Clients that are fully excluded from all tracks

    # First pass: organize all disagreements by type
    for client_id, disagreements in active_disagreements.items():
        #Convert client_id to the ID format used in code (numeric)
        client_id_str = client_id if isinstance(client_id, str) else f"client_{client_id}"
        client_num_id = int(client_id_str.split('_')[1]) if client_id_str.startswith('client_') else int(client_id_str)

        for disagreement in disagreements:
            disagreement_type = disagreement.get("type")

            #Mark this client as involved in disagreements
            clients_with_disagreements.add(client_num_id)

            if disagreement_type == "full":
                # Client wants to be fully excluded from all tracks
                fully_excluded_clients.add(client_num_id)
                continue

            # For regular disagreement types, process the target
            target = disagreement.get("target")

            # Convert target to numeric ID
            target_str = target if isinstance(target, str) else f"client_{target}"
            target_num_id = int(target_str.split('_')[1]) if target_str.startswith('client_') else int(target_str)

            #Mark target client as involved in disagreements
            clients_with_disagreements.add(target_num_id)

            if disagreement_type == "inbound":
                inbound_exclusions[client_num_id].add(target_num_id)
            elif disagreement_type == "outbound":
                #Client wants to be excluded from target's model
                outbound_exclusions[client_num_id].add(target_num_id)
            elif disagreement_type == "bidirectional":
                # For bidirectional, treat as inbound for both sides
                inbound_exclusions[client_num_id].add(target_num_id)
                inbound_exclusions[target_num_id].add(client_num_id)

                # Also record in bidirectional for reference
                bidirectional_exclusions[client_num_id].add(target_num_id)
                bidirectional_exclusions[target_num_id].add(client_num_id)

    # Process outbound exclusions - add to inbound exclusions of the targets
    for client_id, target_clients in outbound_exclusions.items():
        for target_id in target_clients:
            #Target should exclude this client from its model
            inbound_exclusions[target_id].add(client_id)

    #Identify distinct exclusion patterns
    exclusion_patterns = {}  # pattern -> set of client_ids with that pattern
    pattern_to_track = {}    # pattern -> track_name

    # Process inbound exclusions to identify patterns
    for client_id, excluded_set in inbound_exclusions.items():
        #Convert excluded set to a hashable tuple for pattern matching
        pattern = tuple(sorted(excluded_set))
        if pattern not in exclusion_patterns:
            exclusion_patterns[pattern] = set()
        exclusion_patterns[pattern].add(client_id)

    #Process outbound-only clients that don't have inbound exclusions
    outbound_only_clients = set()
    for client_id, disagreements in active_disagreements.items():
        client_id_str = client_id if isinstance(client_id, str) else f"client_{client_id}"
        client_num_id = int(client_id_str.split('_')[1]) if client_id_str.startswith('client_') else int(client_id_str)

        # Skip fully excluded clients
        if client_num_id in fully_excluded_clients:
            continue

        has_outbound = any(d.get("type") == "outbound" for d in disagreements)
        has_inbound = client_num_id in inbound_exclusions

        if has_outbound and not has_inbound:
            outbound_only_clients.add(client_num_id)

    # Create consolidated tracks for each distinct exclusion pattern
    for pattern, client_group in exclusion_patterns.items():
        # Skip empty patterns
        if not pattern:
            continue

        #Create descriptive track name based on exclusion pattern
        excluded_str = "_".join([f"no{target}" for target in sorted(pattern)])

        #If multiple clients share this pattern, include them in the track name
        if len(client_group) > 1:
            clients_str = "_".join([f"c{cid}" for cid in sorted(client_group)])
            track_name = f"track_{clients_str}_{excluded_str}"
        else:
            # Single client with this pattern
            client_id = next(iter(client_group))
            track_name = f"track_{client_id}_{excluded_str}"

        # Store mapping from pattern to track
        pattern_to_track[pattern] = track_name

        # Build the track membership - include all clients except those in the exclusion pattern
        #and fully excluded clients
        track_clients = set()
        for cid in all_client_ids:
            if cid not in pattern and cid not in fully_excluded_clients:
                track_clients.add(cid)

        #Add the track
        tracks[track_name] = track_clients

        # Assign this track as primary for all clients in the group
        for client_id in client_group:
            # Skip fully excluded clients
            if client_id in fully_excluded_clients:
                continue

            client_primary_tracks[client_id] = track_name

            # Set up background participations
            for cid in all_client_ids:
                if cid != client_id and cid not in pattern and cid not in fully_excluded_clients:
                    client_background_participations[client_id].add(cid)

    #Create a single track for outbound-only clients if they exist
    if outbound_only_clients:
        #Check if these clients can just use the global track
        # Outbound-only clients should use the global track if their pattern includes all clients
        for client_id in outbound_only_clients:
            client_primary_tracks[client_id] = "global"

            # Set up background participations
            for cid in all_client_ids:
                if cid != client_id and cid not in fully_excluded_clients:
                    client_background_participations[client_id].add(cid)

    # Create a global track for all clients except fully excluded ones
    global_track_clients = set(cid for cid in all_client_ids if cid not in fully_excluded_clients)
    tracks["global"] = global_track_clients

    clients_without_disagreements = set(all_client_ids) - clients_with_disagreements

    #Assign global track to clients without disagreements
    for client_id in clients_without_disagreements:
        client_primary_tracks[client_id] = "global"
        print(f"Client {client_id} assigned to global track (no disagreements)")

    #Check for track consolidation opportunities
    # If no special tracks were created yet, all clients use the global track
    if len(tracks) == 1:  # Only global track exists
        for client_id in all_client_ids:
            if client_id not in fully_excluded_clients:
                client_primary_tracks[client_id] = "global"
                print(f"Client {client_id} assigned to global track (no special tracks)")

    for client_id in all_client_ids:
        if client_id in fully_excluded_clients:
            print(f"Client {client_id} is fully excluded and not assigned to any track")
            continue

        if client_id not in client_primary_tracks:
            # Check if client is a target of outbound exclusions
            is_target_of_outbound = False
            for source_client, target_clients in outbound_exclusions.items():
                if client_id in target_clients:
                    is_target_of_outbound = True
                    break

            #If a client has no primary track yet and is not a target of outbound exclusions,
            #assign it to the global track
            if not is_target_of_outbound:
                client_primary_tracks[client_id] = "global"
                print(f"Client {client_id} assigned to global track (no outbound exclusions)")
            else:
                # Only create a special track for clients that are targets of outbound exclusions
                track_name = f"track_{client_id}_excluded"
                if track_name not in tracks:
                    # Include all clients except those that have outbound exclusions to this client
                    # and fully excluded clients
                    track_clients = set()
                    for cid in all_client_ids:
                        if (cid not in outbound_exclusions or client_id not in outbound_exclusions[cid]) and cid not in fully_excluded_clients:
                            track_clients.add(cid)
                    tracks[track_name] = track_clients
                client_primary_tracks[client_id] = track_name

    if clients_without_disagreements:
        print(f"Clients not involved in disagreements: {clients_without_disagreements}")

    if fully_excluded_clients:
        print(f"Fully excluded clients: {fully_excluded_clients}")

    print("\n=== MODEL TRACK ASSIGNMENTS ===")
    print(f"Total tracks created: {len(tracks)}")

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
    print("=== END TRACK ASSIGNMENTS ===\n")

    #Convert client_primary_tracks keys to strings for JSON compatibility
    string_client_primary_tracks = {}
    for client_id, track_name in client_primary_tracks.items():
        string_client_primary_tracks[str(client_id)] = track_name

    #Final pass: Identify and consolidate redundant tracks
    # This eliminates tracks that have the exact same members
    track_membership_to_name = {}
    redundant_tracks = []

    # First identify redundant tracks
    for track_name, track_members in tracks.items():
        # Create a hashable representation of track membership
        track_key = tuple(sorted(track_members))

        #Skip the global track for special handling
        if track_name == "global":
            track_membership_to_name[track_key] = track_name
            continue

        if track_key in track_membership_to_name:
            #We found a redundant track
            canonical_track = track_membership_to_name[track_key]
            redundant_tracks.append((track_name, canonical_track))
            print(f"Redundant track detected: '{track_name}' is identical to '{canonical_track}'")
        else:
            track_membership_to_name[track_key] = track_name

    # Handle special case: track identical to global
    global_key = tuple(sorted(tracks["global"]))
    for track_name, track_members in tracks.items():
        if track_name != "global":
            track_key = tuple(sorted(track_members))
            if track_key == global_key:
                redundant_tracks.append((track_name, "global"))
                print(f"Redundant track detected: '{track_name}' is identical to 'global'")

    # Then redirect clients from redundant tracks
    for redundant_track, canonical_track in redundant_tracks:
        # Redirect all clients from redundant track to canonical track
        for client_id, track in client_primary_tracks.items():
            if track == redundant_track:
                client_primary_tracks[client_id] = canonical_track
                print(f"Client {client_id} redirected from redundant track '{redundant_track}' to '{canonical_track}'")

        #Remove the redundant track
        if redundant_track in tracks:
            del tracks[redundant_track]

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
    """Get the track name for a specific client.

    Args:
        client_id: The client ID
        track_info: Dictionary containing track information

    Returns:
        str: Name of the track for this client
    """
    return track_info.get("client_tracks", {}).get(client_id, "default")

def get_clients_in_track(track_name, track_info):
    """Get the set of clients in a specific track.

    Args:
        track_name: Name of the track
        track_info: Dictionary containing track information

    Returns:
        set: Set of client IDs in this track
    """
    return track_info.get("tracks", {}).get(track_name, set())

def get_client_participation_in_tracks(client_id, track_info):
    """Get all tracks in which a client participates.

    Args:
        client_id: The client ID
        track_info: Dictionary containing track information

    Returns:
        list: List of track names in which this client participates
    """
    participating_tracks = []

    for track_name, clients in track_info.get("tracks", {}).items():
        if client_id in clients:
            participating_tracks.append(track_name)

    return participating_tracks
