#!/usr/bin/env python3
"""
Script to visualize track contributions for federated learning simulations.

This script reads track metadata from FL simulation results and creates a heatmap
visualization showing which clients contributed to which tracks in each round.
"""

import json
import os
import argparse
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def usage():
    """Print usage information."""
    print("""Usage: visualize_track_contributions.py <simulation_path> [options]

Arguments:
  simulation_path          Path to the FL simulation folder.

Options:
  --save <filename>        Custom filename to save the plot (saved in simulation/output/ directory).
  --display                Display the plot instead of saving it.
  --dpi <num>              DPI for saved figures (default: 300).
  --rounds <num>           Total number of rounds (used to fill missing rounds with global track).
  -h, --help               Display this help and exit.

Examples:
  visualize_track_contributions.py results/fl_simulation_20250608_203202_mnist_s1
  visualize_track_contributions.py results/fl_simulation_20250608_203202_mnist_s1 --save custom_name.png
  visualize_track_contributions.py results/fl_simulation_20250608_203202_mnist_s1 --display
  visualize_track_contributions.py results/fl_simulation_20250608_203202_mnist_s1 --rounds 10
""")


def load_track_metadata(simulation_path: str, total_rounds: int = None) -> Dict[int, Dict]:
    """
    Load track metadata from all rounds in a simulation.

    Args:
        simulation_path: Path to the FL simulation folder
        total_rounds: Total number of rounds to include (fills missing rounds with global track)

    Returns:
        Dictionary mapping round numbers to track metadata
    """
    model_storage_path = Path(simulation_path) / "model_storage"
    round_data = {}

    # Find all round directories
    round_dirs = glob.glob(str(model_storage_path / "round_*"))

    for round_dir in sorted(round_dirs):
        round_num = int(Path(round_dir).name.split('_')[1])
        metadata_file = Path(round_dir) / "tracks" / "track_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                round_data[round_num] = json.load(f)
        else:
            print(f"Warning: No track metadata found for round {round_num}")

    # Fill in missing rounds if total_rounds is specified
    if total_rounds and round_data:
        # Get all clients from existing rounds
        all_clients = set()
        for data in round_data.values():
            all_clients.update(data.get('client_tracks', {}).keys())

        client_list = sorted(list(all_clients), key=lambda x: int(x))

        #Fill missing rounds with global track containing all clients
        for round_num in range(1, total_rounds + 1):
            if round_num not in round_data:
                print(f"Filling missing round {round_num} with global track")
                round_data[round_num] = {
                    "round": round_num,
                    "tracks": {
                        "global": [int(c) for c in client_list]
                    },
                    "client_tracks": {c: "global" for c in client_list}
                }

    return round_data


def extract_clients_and_tracks(round_data: Dict[int, Dict]) -> Tuple[List[str], List[str]]:
    """
    Extract unique clients and tracks from all rounds.

    Args:
        round_data: Dictionary mapping round numbers to track metadata

    Returns:
        Tuple of (sorted client list, sorted track list)
    """
    all_clients = set()
    all_tracks = set()

    for round_num, data in round_data.items():
        #Extract clients from client_tracks mapping
        all_clients.update(data.get('client_tracks', {}).keys())
        # Extract tracks from tracks mapping
        all_tracks.update(data.get('tracks', {}).keys())

    # Convert to sorted lists
    clients = sorted(all_clients, key=lambda x: int(x))
    tracks = sorted(all_tracks)

    # Move 'global' to the front if it exists
    if 'global' in tracks:
        tracks.remove('global')
        tracks.insert(0, 'global')

    return clients, tracks


def build_contribution_matrix(round_data: Dict[int, Dict], clients: List[str], tracks: List[str]) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Build a matrix showing client contributions to tracks across rounds.

    Args:
        round_data: Dictionary mapping round numbers to track metadata
        clients: List of client IDs
        tracks: List of track names

    Returns:
        Tuple of (contribution matrix, column info list)
        Matrix values: 0=no participation, 1=participation, 2=inactive track
    """
    rounds = sorted(round_data.keys())
    matrix = np.zeros((len(clients), len(rounds) * len(tracks)), dtype=int)
    col_info = []

    for ri, round_num in enumerate(rounds):
        for ti, track in enumerate(tracks):
            col_info.append((round_num, track))
            col_idx = ri * len(tracks) + ti

            #Get tracks that are active in this round
            round_tracks = round_data[round_num].get('tracks', {})

            #Check if this track is active in this round
            if track not in round_tracks:
                # Track is inactive - set all clients for this track to 2 (inactive)
                for ci, client in enumerate(clients):
                    matrix[ci, col_idx] = 2
            else:
                # Track is active - set participation based on contributing clients
                contributing_clients = set(str(c) for c in round_tracks.get(track, []))
                for ci, client in enumerate(clients):
                    if client in contributing_clients:
                        matrix[ci, col_idx] = 1
                    else:
                        matrix[ci, col_idx] = 0

    return matrix, col_info


def create_visualization(matrix: np.ndarray, col_info: List[Tuple[int, str]],
                        clients: List[str], tracks: List[str],
                        simulation_name: str = None) -> None:
    """
    Create and display the track contribution visualization.

    Args:
        matrix: Contribution matrix (0=no participation, 1=participation, 2=inactive track)
        col_info: List of (round, track) tuples for each column
        clients: List of client IDs
        tracks: List of track names
        simulation_name: Optional name for the simulation
    """
    # Extract unique rounds for major ticks
    rounds = sorted(list(set(round_num for round_num, _ in col_info)))
    n_tracks = len(tracks)
    total_cols = len(col_info)

    #Define 3-color discrete colormap: no participation, participation, inactive track
    cmap = ListedColormap(['#ECEFF1', '#40916C', '#9E9E9E'])  #0: light slate, 1: muted mint, 2: grey

    # Calculate figure size based on data dimensions
    fig_width = max(8, min(20, total_cols * 0.4))
    fig_height = max(3, len(clients) * 0.4)

    # Plot heatmap with rectangular cells
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)  # Updated to handle 3 states

    #Setup ticks: major for rounds, minor for tracks
    major_pos = []
    round_labels = []

    for ri, round_num in enumerate(rounds):
        start_idx = ri * n_tracks
        center_pos = start_idx + (n_tracks - 1) / 2
        major_pos.append(center_pos)
        round_labels.append(f'Round {round_num}')

    #Set up dual x-axis: major ticks (rounds) at bottom, minor ticks (tracks) at top
    ax.set_xticks(major_pos)
    ax.set_xticklabels(round_labels, fontsize=10, fontweight='bold', rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='major', pad=5)  # Padding for round labels at bottom

    # Create second x-axis for track labels at top
    ax2 = ax.twiny()
    positions = np.arange(total_cols)
    track_labels = [track for (_round, track) in col_info]
    ax2.set_xticks(positions)
    ax2.set_xticklabels(track_labels, fontsize=8, rotation=90, ha='center', va='bottom')
    ax2.tick_params(axis='x', pad=6)
    ax2.set_xlim(ax.get_xlim())

    # Y-axis labels
    client_labels = [f'C$_{{{client}}}$' for client in clients]
    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels(client_labels, fontsize=8)

    #Remove default grid
    ax.grid(False)

    #Separator lines: dotted black between tracks, prominent solid black between rounds
    for sep in np.arange(0.5, total_cols, 1):
        if (sep + 0.5) % n_tracks == 0:
            # Round separator: thicker line that extends beyond the plot area
            ax.axvline(sep, color='black', linewidth=3, linestyle='-',
                      ymin=-1, ymax=2, clip_on=False)  # round separator
        else:
            ax.axvline(sep, color='black', linewidth=0.8, linestyle=':')  # track separator

    #Horizontal dotted lines between each client row
    for y in np.arange(0.5, len(clients), 1):
        ax.axhline(y, color='black', linewidth=0.8, linestyle=':')

    #Add compact legend inside plot area
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#40916C', edgecolor='black', linewidth=0.5, label='Participates'),
        Patch(facecolor='#ECEFF1', edgecolor='black', linewidth=0.5, label='No participation')
    ]

    # Only add "Inactive track" to legend if there are actually inactive tracks in the matrix
    if np.any(matrix == 2):
        legend_elements.append(
            Patch(facecolor='#9E9E9E', edgecolor='black', linewidth=0.5, label='Inactive track')
        )

    legend = ax.legend(handles=legend_elements, loc='lower right',
                       frameon=True, fontsize=6, handlelength=0.6, handletextpad=0.2,
                       columnspacing=0.3, borderpad=0.2)
    legend.get_frame().set_alpha(0.5)

    # Final layout
    plt.tight_layout()
    plt.show()


def main():
    """Main function to parse arguments and create visualization."""
    # Check for help flag before parsing to avoid required argument errors
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()
        return 0

    parser = argparse.ArgumentParser(
        description='Visualize track contributions for FL simulations',
        add_help=False)  #We handle help manually

    parser.add_argument('simulation_path',
                       help='Path to the FL simulation folder')
    parser.add_argument('--save',
                       help='Custom filename to save the plot (saved in simulation/output/ directory)')
    parser.add_argument('--display', action='store_true',
                       help='Display the plot instead of saving it')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--rounds', type=int,
                       help='Total number of rounds (used to fill missing rounds with global track)')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help and exit')

    args = parser.parse_args()

    if args.help:
        usage()
        return 0

    #Validate simulation path
    if not os.path.exists(args.simulation_path):
        print(f"Error: Simulation path '{args.simulation_path}' does not exist")
        return 1

    model_storage_path = Path(args.simulation_path) / "model_storage"
    if not model_storage_path.exists():
        print(f"Error: No model_storage directory found in '{args.simulation_path}'")
        return 1

    # Load and process data
    print(f"Loading track metadata from: {args.simulation_path}")
    round_data = load_track_metadata(args.simulation_path, args.rounds)

    if not round_data:
        print("Error: No track metadata found in any rounds")
        return 1

    print(f"Found {len(round_data)} rounds: {sorted(round_data.keys())}")

    # Extract clients and tracks
    clients, tracks = extract_clients_and_tracks(round_data)
    print(f"Clients: {clients}")
    print(f"Tracks: {tracks}")

    # Build contribution matrix
    matrix, col_info = build_contribution_matrix(round_data, clients, tracks)

    #Create visualization
    simulation_name = Path(args.simulation_path).name

    #Determine save behavior
    if args.display:
        # Display the plot
        create_visualization(matrix, col_info, clients, tracks, simulation_name)
    else:
        # Save the plot (default behavior)
        # Create output directory if it doesn't exist
        output_dir = Path(args.simulation_path) / "output"
        output_dir.mkdir(exist_ok=True)

        #Determine filename
        if args.save:
            filename = args.save
        else:
            filename = f"track_contributions_{simulation_name}.png"

        save_path = output_dir / filename

        #Set backend for saving
        import matplotlib
        matplotlib.use('Agg')
        create_visualization(matrix, col_info, clients, tracks, simulation_name)
        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    return 0


if __name__ == "__main__":
    exit(main())
