import math
import os
import sys
import signal
import json
import subprocess
import time
import fileinput
import matplotlib.pyplot as plt
import timeit
import pandas as pd

from pysat.formula import CNF
from pysat.solvers import Glucose42

# Global variables to track best solution found so far
best_num_bins = float('inf')
best_solution = None
variables_length = 0
clauses_length = 0
upper_bound = 0

# Signal handler for graceful interruption
def handle_interrupt(signum, frame):
    print(f"\nReceived interrupt signal {signum}. Saving current best solution.")
    
    current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
    print(f"Best number of bins found before interrupt: {current_bins}")
    
    # Save result as JSON for the controller to pick up
    result = {
        'Instance': instances[instance_id],
        'Variables': variables_length,
        'Clauses': clauses_length,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': current_bins,
        'Status': 'TIMEOUT'
    }
    
    with open(f'results_BPP_INC_C1_{instance_id}.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_interrupt)
signal.signal(signal.SIGINT, handle_interrupt)

# Create BPP_C1 folder if it doesn't exist
if not os.path.exists('BPP_INC_C1'):
    os.makedirs('BPP_INC_C1')

def display_solution(bin_width, bin_height, rectangles, bins_assignment, positions, instance_name):
    num_bins = len(bins_assignment)
    
    if num_bins == 0:
        return

    ncols = min(num_bins, 5)
    nrows = (num_bins + ncols - 1) // ncols
    
    # Create subplots for each bin
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    fig.suptitle(f'BPP_INC_C1 - {instance_name} - {num_bins} bins', fontsize=16)
    
    # Handle different subplot configurations
    if num_bins == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for bin_idx, items_in_bin in enumerate(bins_assignment):
        ax = axes[bin_idx]
        ax.set_title(f'Bin {bin_idx + 1}')
        
        # Draw rectangles in this bin
        for item_idx in items_in_bin:
            # Get dimensions based on rotation
            w, h = rectangles[item_idx][0], rectangles[item_idx][1]
            
            rect = plt.Rectangle(positions[item_idx], w, h, 
                               edgecolor="#333", facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            
            # Add item label with rotation indicator
            label = f"{item_idx + 1}"
            ax.text(positions[item_idx][0] + w/2,
                   positions[item_idx][1] + h/2,
                   label, ha='center', va='center')
        
        ax.set_xlim(0, bin_width)
        ax.set_ylim(0, bin_height)
        ax.set_xticks(range(0,bin_width, max(2, bin_width // 10)))
        ax.set_yticks(range(0,bin_height, max(2, bin_height // 10)))
        ax.set_xlabel('width')
        ax.set_ylabel('height')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(num_bins, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'BPP_INC_C1/{instance_name}.png')
    plt.close()


def read_file_instance(instance_name):
    """Read instance file based on instance name"""
    s = ''
    
    # Determine file path based on instance name
    if instance_name.startswith('BENG'):
        filepath = f"inputs/BENG/{instance_name}.txt"
    elif instance_name.startswith('CL_'):
        filepath = f"inputs/CLASS/{instance_name}.txt"
    else:
        # For other instances, try different folders
        possible_paths = [
            f"inputs/{instance_name}.txt",
            f"inputs/BENG/{instance_name}.txt",
            f"inputs/CLASS/{instance_name}.txt"
        ]
        
        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Could not find instance file for {instance_name}")
    
    try:
        with open(filepath, 'r') as f:
            s = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    return s.splitlines()

# Updated instance list with actual available instances
instances = [
    "",
    # BENG instances (10 instances)
    "BENG01", "BENG02", "BENG03", "BENG04", "BENG05",
    "BENG06", "BENG07", "BENG08", "BENG09", "BENG10",
    
    # CLASS instances (500 instances)
    # CL_1_20_x (10 instances)
    "CL_1_20_1", "CL_1_20_2", "CL_1_20_3", "CL_1_20_4", "CL_1_20_5",
    "CL_1_20_6", "CL_1_20_7", "CL_1_20_8", "CL_1_20_9", "CL_1_20_10",
    
    # CL_1_40_x (10 instances)
    "CL_1_40_1", "CL_1_40_2", "CL_1_40_3", "CL_1_40_4", "CL_1_40_5",
    "CL_1_40_6", "CL_1_40_7", "CL_1_40_8", "CL_1_40_9", "CL_1_40_10",
    
    # CL_1_60_x (10 instances)
    "CL_1_60_1", "CL_1_60_2", "CL_1_60_3", "CL_1_60_4", "CL_1_60_5",
    "CL_1_60_6", "CL_1_60_7", "CL_1_60_8", "CL_1_60_9", "CL_1_60_10",
    
    # CL_1_80_x (10 instances)
    "CL_1_80_1", "CL_1_80_2", "CL_1_80_3", "CL_1_80_4", "CL_1_80_5",
    "CL_1_80_6", "CL_1_80_7", "CL_1_80_8", "CL_1_80_9", "CL_1_80_10",
    
    # CL_1_100_x (10 instances)
    "CL_1_100_1", "CL_1_100_2", "CL_1_100_3", "CL_1_100_4", "CL_1_100_5",
    "CL_1_100_6", "CL_1_100_7", "CL_1_100_8", "CL_1_100_9", "CL_1_100_10",
    
    # CL_2_20_x (10 instances)
    "CL_2_20_1", "CL_2_20_2", "CL_2_20_3", "CL_2_20_4", "CL_2_20_5",
    "CL_2_20_6", "CL_2_20_7", "CL_2_20_8", "CL_2_20_9", "CL_2_20_10",
    
    # CL_2_40_x (10 instances)
    "CL_2_40_1", "CL_2_40_2", "CL_2_40_3", "CL_2_40_4", "CL_2_40_5",
    "CL_2_40_6", "CL_2_40_7", "CL_2_40_8", "CL_2_40_9", "CL_2_40_10",
    
    # CL_2_60_x (10 instances)
    "CL_2_60_1", "CL_2_60_2", "CL_2_60_3", "CL_2_60_4", "CL_2_60_5",
    "CL_2_60_6", "CL_2_60_7", "CL_2_60_8", "CL_2_60_9", "CL_2_60_10",
    
    # CL_2_80_x (10 instances)
    "CL_2_80_1", "CL_2_80_2", "CL_2_80_3", "CL_2_80_4", "CL_2_80_5",
    "CL_2_80_6", "CL_2_80_7", "CL_2_80_8", "CL_2_80_9", "CL_2_80_10",
    
    # CL_2_100_x (10 instances)
    "CL_2_100_1", "CL_2_100_2", "CL_2_100_3", "CL_2_100_4", "CL_2_100_5",
    "CL_2_100_6", "CL_2_100_7", "CL_2_100_8", "CL_2_100_9", "CL_2_100_10",
    
    # CL_3_20_x (10 instances)
    "CL_3_20_1", "CL_3_20_2", "CL_3_20_3", "CL_3_20_4", "CL_3_20_5",
    "CL_3_20_6", "CL_3_20_7", "CL_3_20_8", "CL_3_20_9", "CL_3_20_10",
    
    # CL_3_40_x (10 instances)
    "CL_3_40_1", "CL_3_40_2", "CL_3_40_3", "CL_3_40_4", "CL_3_40_5",
    "CL_3_40_6", "CL_3_40_7", "CL_3_40_8", "CL_3_40_9", "CL_3_40_10",
    
    # CL_3_60_x (10 instances)
    "CL_3_60_1", "CL_3_60_2", "CL_3_60_3", "CL_3_60_4", "CL_3_60_5",
    "CL_3_60_6", "CL_3_60_7", "CL_3_60_8", "CL_3_60_9", "CL_3_60_10",
    
    # CL_3_80_x (10 instances)
    "CL_3_80_1", "CL_3_80_2", "CL_3_80_3", "CL_3_80_4", "CL_3_80_5",
    "CL_3_80_6", "CL_3_80_7", "CL_3_80_8", "CL_3_80_9", "CL_3_80_10",
    
    # CL_3_100_x (10 instances)
    "CL_3_100_1", "CL_3_100_2", "CL_3_100_3", "CL_3_100_4", "CL_3_100_5",
    "CL_3_100_6", "CL_3_100_7", "CL_3_100_8", "CL_3_100_9", "CL_3_100_10",
    
    # CL_4_20_x (10 instances)
    "CL_4_20_1", "CL_4_20_2", "CL_4_20_3", "CL_4_20_4", "CL_4_20_5",
    "CL_4_20_6", "CL_4_20_7", "CL_4_20_8", "CL_4_20_9", "CL_4_20_10",
    
    # CL_4_40_x (10 instances)
    "CL_4_40_1", "CL_4_40_2", "CL_4_40_3", "CL_4_40_4", "CL_4_40_5",
    "CL_4_40_6", "CL_4_40_7", "CL_4_40_8", "CL_4_40_9", "CL_4_40_10",
    
    # CL_4_60_x (10 instances)
    "CL_4_60_1", "CL_4_60_2", "CL_4_60_3", "CL_4_60_4", "CL_4_60_5",
    "CL_4_60_6", "CL_4_60_7", "CL_4_60_8", "CL_4_60_9", "CL_4_60_10",
    
    # CL_4_80_x (10 instances)
    "CL_4_80_1", "CL_4_80_2", "CL_4_80_3", "CL_4_80_4", "CL_4_80_5",
    "CL_4_80_6", "CL_4_80_7", "CL_4_80_8", "CL_4_80_9", "CL_4_80_10",
    
    # CL_4_100_x (10 instances)
    "CL_4_100_1", "CL_4_100_2", "CL_4_100_3", "CL_4_100_4", "CL_4_100_5",
    "CL_4_100_6", "CL_4_100_7", "CL_4_100_8", "CL_4_100_9", "CL_4_100_10",
    
    # CL_5_20_x (10 instances)
    "CL_5_20_1", "CL_5_20_2", "CL_5_20_3", "CL_5_20_4", "CL_5_20_5",
    "CL_5_20_6", "CL_5_20_7", "CL_5_20_8", "CL_5_20_9", "CL_5_20_10",
    
    # CL_5_40_x (10 instances)
    "CL_5_40_1", "CL_5_40_2", "CL_5_40_3", "CL_5_40_4", "CL_5_40_5",
    "CL_5_40_6", "CL_5_40_7", "CL_5_40_8", "CL_5_40_9", "CL_5_40_10",
    
    # CL_5_60_x (10 instances)
    "CL_5_60_1", "CL_5_60_2", "CL_5_60_3", "CL_5_60_4", "CL_5_60_5",
    "CL_5_60_6", "CL_5_60_7", "CL_5_60_8", "CL_5_60_9", "CL_5_60_10",
    
    # CL_5_80_x (10 instances)
    "CL_5_80_1", "CL_5_80_2", "CL_5_80_3", "CL_5_80_4", "CL_5_80_5",
    "CL_5_80_6", "CL_5_80_7", "CL_5_80_8", "CL_5_80_9", "CL_5_80_10",
    
    # CL_5_100_x (10 instances)
    "CL_5_100_1", "CL_5_100_2", "CL_5_100_3", "CL_5_100_4", "CL_5_100_5",
    "CL_5_100_6", "CL_5_100_7", "CL_5_100_8", "CL_5_100_9", "CL_5_100_10",
    
    # CL_6_20_x (10 instances)
    "CL_6_20_1", "CL_6_20_2", "CL_6_20_3", "CL_6_20_4", "CL_6_20_5",
    "CL_6_20_6", "CL_6_20_7", "CL_6_20_8", "CL_6_20_9", "CL_6_20_10",
    
    # CL_6_40_x (10 instances)
    "CL_6_40_1", "CL_6_40_2", "CL_6_40_3", "CL_6_40_4", "CL_6_40_5",
    "CL_6_40_6", "CL_6_40_7", "CL_6_40_8", "CL_6_40_9", "CL_6_40_10",
    
    # CL_6_60_x (10 instances)
    "CL_6_60_1", "CL_6_60_2", "CL_6_60_3", "CL_6_60_4", "CL_6_60_5",
    "CL_6_60_6", "CL_6_60_7", "CL_6_60_8", "CL_6_60_9", "CL_6_60_10",
    
    # CL_6_80_x (10 instances)
    "CL_6_80_1", "CL_6_80_2", "CL_6_80_3", "CL_6_80_4", "CL_6_80_5",
    "CL_6_80_6", "CL_6_80_7", "CL_6_80_8", "CL_6_80_9", "CL_6_80_10",
    
    # CL_6_100_x (10 instances)
    "CL_6_100_1", "CL_6_100_2", "CL_6_100_3", "CL_6_100_4", "CL_6_100_5",
    "CL_6_100_6", "CL_6_100_7", "CL_6_100_8", "CL_6_100_9", "CL_6_100_10",
    
    # CL_7_20_x (10 instances)
    "CL_7_20_1", "CL_7_20_2", "CL_7_20_3", "CL_7_20_4", "CL_7_20_5",
    "CL_7_20_6", "CL_7_20_7", "CL_7_20_8", "CL_7_20_9", "CL_7_20_10",
    
    # CL_7_40_x (10 instances)
    "CL_7_40_1", "CL_7_40_2", "CL_7_40_3", "CL_7_40_4", "CL_7_40_5",
    "CL_7_40_6", "CL_7_40_7", "CL_7_40_8", "CL_7_40_9", "CL_7_40_10",
    
    # CL_7_60_x (10 instances)
    "CL_7_60_1", "CL_7_60_2", "CL_7_60_3", "CL_7_60_4", "CL_7_60_5",
    "CL_7_60_6", "CL_7_60_7", "CL_7_60_8", "CL_7_60_9", "CL_7_60_10",
    
    # CL_7_80_x (10 instances)
    "CL_7_80_1", "CL_7_80_2", "CL_7_80_3", "CL_7_80_4", "CL_7_80_5",
    "CL_7_80_6", "CL_7_80_7", "CL_7_80_8", "CL_7_80_9", "CL_7_80_10",
    
    # CL_7_100_x (10 instances)
    "CL_7_100_1", "CL_7_100_2", "CL_7_100_3", "CL_7_100_4", "CL_7_100_5",
    "CL_7_100_6", "CL_7_100_7", "CL_7_100_8", "CL_7_100_9", "CL_7_100_10",
    
    # CL_8_20_x (10 instances)
    "CL_8_20_1", "CL_8_20_2", "CL_8_20_3", "CL_8_20_4", "CL_8_20_5",
    "CL_8_20_6", "CL_8_20_7", "CL_8_20_8", "CL_8_20_9", "CL_8_20_10",
    
    # CL_8_40_x (10 instances)
    "CL_8_40_1", "CL_8_40_2", "CL_8_40_3", "CL_8_40_4", "CL_8_40_5",
    "CL_8_40_6", "CL_8_40_7", "CL_8_40_8", "CL_8_40_9", "CL_8_40_10",
    
    # CL_8_60_x (10 instances)
    "CL_8_60_1", "CL_8_60_2", "CL_8_60_3", "CL_8_60_4", "CL_8_60_5",
    "CL_8_60_6", "CL_8_60_7", "CL_8_60_8", "CL_8_60_9", "CL_8_60_10",
    
    # CL_8_80_x (10 instances)
    "CL_8_80_1", "CL_8_80_2", "CL_8_80_3", "CL_8_80_4", "CL_8_80_5",
    "CL_8_80_6", "CL_8_80_7", "CL_8_80_8", "CL_8_80_9", "CL_8_80_10",
    
    # CL_8_100_x (10 instances)
    "CL_8_100_1", "CL_8_100_2", "CL_8_100_3", "CL_8_100_4", "CL_8_100_5",
    "CL_8_100_6", "CL_8_100_7", "CL_8_100_8", "CL_8_100_9", "CL_8_100_10",
    
    # CL_9_20_x (10 instances)
    "CL_9_20_1", "CL_9_20_2", "CL_9_20_3", "CL_9_20_4", "CL_9_20_5",
    "CL_9_20_6", "CL_9_20_7", "CL_9_20_8", "CL_9_20_9", "CL_9_20_10",
    
    # CL_9_40_x (10 instances)
    "CL_9_40_1", "CL_9_40_2", "CL_9_40_3", "CL_9_40_4", "CL_9_40_5",
    "CL_9_40_6", "CL_9_40_7", "CL_9_40_8", "CL_9_40_9", "CL_9_40_10",
    
    # CL_9_60_x (10 instances)
    "CL_9_60_1", "CL_9_60_2", "CL_9_60_3", "CL_9_60_4", "CL_9_60_5",
    "CL_9_60_6", "CL_9_60_7", "CL_9_60_8", "CL_9_60_9", "CL_9_60_10",
    
    # CL_9_80_x (10 instances)
    "CL_9_80_1", "CL_9_80_2", "CL_9_80_3", "CL_9_80_4", "CL_9_80_5",
    "CL_9_80_6", "CL_9_80_7", "CL_9_80_8", "CL_9_80_9", "CL_9_80_10",
    
    # CL_9_100_x (10 instances)
    "CL_9_100_1", "CL_9_100_2", "CL_9_100_3", "CL_9_100_4", "CL_9_100_5",
    "CL_9_100_6", "CL_9_100_7", "CL_9_100_8", "CL_9_100_9", "CL_9_100_10",
    
    # CL_10_20_x (10 instances)
    "CL_10_20_1", "CL_10_20_2", "CL_10_20_3", "CL_10_20_4", "CL_10_20_5",
    "CL_10_20_6", "CL_10_20_7", "CL_10_20_8", "CL_10_20_9", "CL_10_20_10",
    
    # CL_10_40_x (10 instances)
    "CL_10_40_1", "CL_10_40_2", "CL_10_40_3", "CL_10_40_4", "CL_10_40_5",
    "CL_10_40_6", "CL_10_40_7", "CL_10_40_8", "CL_10_40_9", "CL_10_40_10",
    
    # CL_10_60_x (10 instances)
    "CL_10_60_1", "CL_10_60_2", "CL_10_60_3", "CL_10_60_4", "CL_10_60_5",
    "CL_10_60_6", "CL_10_60_7", "CL_10_60_8", "CL_10_60_9", "CL_10_60_10",
    
    # CL_10_80_x (10 instances)
    "CL_10_80_1", "CL_10_80_2", "CL_10_80_3", "CL_10_80_4", "CL_10_80_5",
    "CL_10_80_6", "CL_10_80_7", "CL_10_80_8", "CL_10_80_9", "CL_10_80_10",
    
    # CL_10_100_x (10 instances)
    "CL_10_100_1", "CL_10_100_2", "CL_10_100_3", "CL_10_100_4", "CL_10_100_5",
    "CL_10_100_6", "CL_10_100_7", "CL_10_100_8", "CL_10_100_9", "CL_10_100_10"
]

if os.path.exists("BPP_INC_C1_timeout.txt"):
    with open("BPP_INC_C1_timeout.txt", "r") as f:
        instances = [""] + [line.strip() for line in f if line.strip()]

def positive_range(end):
    if end < 0:
        return []
    return range(end)

def calculate_lower_bound(bin_width, bin_height, rectangles):
    """Calculate lower bound for number of bins needed"""
    total_area = sum(w * h for w, h in rectangles)
    bin_area = bin_width * bin_height
    return math.ceil(total_area / bin_area)

def first_fit_upper_bound(rectangles, W, H):
    """Finite First-Fit (FFF) upper bound for 2D bin packing without rotation."""
    # Each bin is a list of placed rectangles: (x, y, w, h)
    bins = []
    def fits(bin_rects, w, h, W, H):
        # Try to place at the lowest possible y for each x in the bin
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                rect = (x, y, w, h)
                overlap = False
                for (px, py, pw, ph) in bin_rects:
                    if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
        return None
    
    for rect in rectangles:
        placed = False
        w, h = rect[0], rect[1]
        
        # Check if rectangle fits in bin at all
        if w > W or h > H:
            return float('inf')  # Infeasible rectangle
        
        # Try to place in existing bins
        for bin_rects in bins:
            pos = fits(bin_rects, w, h, W, H)
            if pos is not None:
                bin_rects.append((pos[0], pos[1], w, h))
                placed = True
                break
        
        if not placed:
            # Start a new bin, place at (0,0)
            bins.append([(0, 0, w, h)])
    
    return len(bins)

def save_checkpoint(instance_id, variables, clauses, num_bins, status="IN_PROGRESS"):
    """Save checkpoint for current progress"""
    checkpoint = {
        'Instance': instances[instance_id],
        'Variables': variables,
        'Clauses': clauses,
        'Runtime': timeit.default_timer() - start,
        'Optimal_Bins': num_bins if num_bins != float('inf') else upper_bound,
        'Status': status
    }
    
    with open(f'checkpoint_BPP_INC_C1_{instance_id}.json', 'w') as f:
        json.dump(checkpoint, f)

def OPP(rectangles, upper_bound, bin_width, bin_height):
    """Solve 2D Bin Packing with given number of bins"""
    global variables_length, clauses_length, best_num_bins, best_solution
    
    cnf = CNF()
    variables = {}
    counter = 1

    # Create assignment variables: x[i,j] = item i assigned to bin j
    for i in range(len(rectangles)):
        for j in range(upper_bound):
            variables[f"x{i + 1},{j + 1}"] = counter
            counter += 1

    # Create position variables for each item
    for i in range(len(rectangles)):
        # Position variables for x-coordinate
        for e in positive_range(bin_width - rectangles[i][0] + 1):
            variables[f"px{i + 1},{e}"] = counter
            counter += 1
        # Position variables for y-coordinate  
        for f in positive_range(bin_height - rectangles[i][1] + 1):
            variables[f"py{i + 1},{f}"] = counter
            counter += 1

    # Create relative position variables for non-overlapping constraints
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = counter  # i is left of j
                counter += 1
                variables[f"ud{i + 1},{j + 1}"] = counter  # i is below j
                counter += 1

    # Create bin usage variables
    for j in range(upper_bound):
        variables[f"b{j + 1}"] = counter
        counter += 1

    # Constraint 1: Each item must be assigned to exactly one bin
    for i in range(len(rectangles)):
        # At least one bin
        cnf.append([variables[f"x{i + 1},{j + 1}"] for j in range(upper_bound)])
        # At most one bin
        for j1 in range(upper_bound):
            for j2 in range(j1 + 1, upper_bound):
                cnf.append([-variables[f"x{i + 1},{j1 + 1}"], -variables[f"x{i + 1},{j2 + 1}"]])

    # Constraint 2: Order constraints for position variables
    for i in range(len(rectangles)):
        # x-coordinate order: px[i,e] → px[i,e+1]
        for e in range(bin_width - rectangles[i][0]):
            cnf.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
        # y-coordinate order: py[i,f] → py[i,f+1]
        for f in range(bin_height - rectangles[i][1]):
            cnf.append([-variables[f"py{i + 1},{f}"], variables[f"py{i + 1},{f + 1}"]])

    # Constraint 3: Bin usage constraints
    for j in range(upper_bound):
        for i in range(len(rectangles)):
            # If item i is in bin j, then bin j is used
            cnf.append([-variables[f"x{i + 1},{j + 1}"], variables[f"b{j + 1}"]])

    # Constraint 4: Symmetry Breaking C1 - bin ordering
    for j in range(1, upper_bound):
        cnf.append([-variables[f"b{j + 1}"], variables[f"b{j}"]])

    # Constraint 5: Non-overlapping constraints
    max_height = max([int(rectangle[1]) for rectangle in rectangles])
    max_width = max([int(rectangle[0]) for rectangle in rectangles])
    def add_non_overlapping(i, j, bin_idx):
        """Add non-overlapping constraints for items i and j in bin bin_idx using C1 approach"""
        i_width = rectangles[i][0]
        i_height = rectangles[i][1]
        j_width = rectangles[j][0]
        j_height = rectangles[j][1]
        
        bin_condition = [-variables[f"x{i + 1},{bin_idx + 1}"], -variables[f"x{j + 1},{bin_idx + 1}"]]
        
        # Determine which constraints to apply based on C1 symmetry breaking rules
        h1, h2, v1, v2 = True, True, True, True  # Default: all four directions
        
        # Large-rectangles horizontal
        if i_width + j_width > bin_width:
            h1, h2 = False, False
        # Large-rectangles vertical
        elif i_height + j_height > bin_height:
            v1, v2 = False, False
        # Same-sized rectangles
        elif rectangles[i] == rectangles[j]:
            h2 = False  # Only allow i left of j, not j left of i
        elif i_width == max_width and j_width > (bin_width - i_width) / 2:
            h1 = False
        elif i_height == max_height and j_height > (bin_height - i_height) / 2:
            v1 = False
        
        # Apply the four-literal clause
        four_literal = bin_condition.copy()
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
        cnf.append(four_literal)

        # Add position-based constraints
        if h1:
            for e in range(i_width):
                if f"px{j + 1},{e}" in variables:
                    cnf.append(bin_condition + [-variables[f"lr{i + 1},{j + 1}"], 
                              -variables[f"px{j + 1},{e}"]])
        
        if h2:
            for e in range(j_width):
                if f"px{i + 1},{e}" in variables:
                    cnf.append(bin_condition + [-variables[f"lr{j + 1},{i + 1}"], 
                              -variables[f"px{i + 1},{e}"]])

        if v1:
            for f in range(i_height):
                if f"py{j + 1},{f}" in variables:
                    cnf.append(bin_condition + [-variables[f"ud{i + 1},{j + 1}"], 
                              -variables[f"py{j + 1},{f}"]])
        
        if v2:
            for f in range(j_height):
                if f"py{i + 1},{f}" in variables:
                    cnf.append(bin_condition + [-variables[f"ud{j + 1},{i + 1}"], 
                              -variables[f"py{i + 1},{f}"]])

        # Position-based non-overlapping
        for e in positive_range(bin_width - i_width):
            if h1 and f"px{j + 1},{e + i_width}" in variables:
                cnf.append(bin_condition + [-variables[f"lr{i + 1},{j + 1}"],
                          variables[f"px{i + 1},{e}"],
                          -variables[f"px{j + 1},{e + i_width}"]])
            
            if h2 and f"px{i + 1},{e + j_width}" in variables:
                cnf.append(bin_condition + [-variables[f"lr{j + 1},{i + 1}"],
                          variables[f"px{j + 1},{e}"],
                          -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(bin_height - i_height):
            if v1 and f"py{j + 1},{f + i_height}" in variables:
                cnf.append(bin_condition + [-variables[f"ud{i + 1},{j + 1}"],
                          variables[f"py{i + 1},{f}"],
                          -variables[f"py{j + 1},{f + i_height}"]])
            
            if v2 and f"py{i + 1},{f + j_height}" in variables:
                cnf.append(bin_condition + [-variables[f"ud{j + 1},{i + 1}"],
                          variables[f"py{j + 1},{f}"],
                          -variables[f"py{i + 1},{f + j_height}"]])

    # Apply non-overlapping constraints for all pairs in all bins
    for bin_idx in range(upper_bound):
        for i in range(len(rectangles)):
            for j in range(i + 1, len(rectangles)):
                add_non_overlapping(i, j, bin_idx)

    # Constraint 6: Domain constraints - items must fit within bins
    for i in range(len(rectangles)):
        for bin_idx in range(upper_bound):
            if rectangles[i][0] == max_width:
                # If item is wider than bin, it cannot be placed
                cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"],
                            variables[f"px{i + 1},{(bin_width - rectangles[i][0]) // 2}"]])
            # Item must fit horizontally: px[i, bin_width - width[i]] = true
            else:
                cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], 
                       variables[f"px{i + 1},{bin_width - rectangles[i][0]}"]])
            # Item must fit vertically: py[i, bin_height - height[i]] = true
            cnf.append([-variables[f"x{i + 1},{bin_idx + 1}"], 
                       variables[f"py{i + 1},{bin_height - rectangles[i][1]}"]])

    variables_length = len(variables)
    clauses_length = len(cnf.clauses)
    
    # Save checkpoint
    save_checkpoint(instance_id, variables_length, clauses_length, best_num_bins)

    # Solve with SAT solver
    with Glucose42() as solver:
        solver.append_formula(cnf)
        lb = calculate_lower_bound(bin_width, bin_height, rectangles)
        ub = upper_bound
        print(f"Lower bound: {lb}, Upper bound: {ub}")
        while lb <= ub:
            mid = (lb + ub) // 2
            print(f"Trying {mid} bins")
            # Add bin usage constraint for mid bins
            assumptions = [-variables[f"b{j + 1}"] for j in range(mid, upper_bound)]
            is_sat = solver.solve(assumptions=assumptions)
        
            if is_sat:
                print(f'found solution with {mid} bins')
                model = solver.get_model()
                
                # Update best solution if this is better
                if mid < best_num_bins:
                    best_num_bins = mid
                    save_checkpoint(instance_id, variables_length, clauses_length, best_num_bins)
                
                # Extract solution
                result = {}
                for var in model:
                    if var > 0:
                        result[list(variables.keys())[list(variables.values()).index(var)]] = True
                    else:
                        result[list(variables.keys())[list(variables.values()).index(-var)]] = False
                
                # Extract bin assignments
                bins_assignment = [[] for _ in range(mid)]
                for i in range(len(rectangles)):
                    for j in range(mid):
                        if result[f"x{i + 1},{j + 1}"] == True:
                            bins_assignment[j].append(i)
                            break
                
                # Extract positions
                positions = [[0, 0] for _ in range(len(rectangles))]
                for i in range(len(rectangles)):
                    # Extract x position
                    for e in range(bin_width - rectangles[i][0] + 1):
                        if e == 0 and result[f"px{i + 1},{e}"] == True:
                            positions[i][0] = 0
                            break
                        elif e > 0 and result[f"px{i + 1},{e - 1}"] == False and result[f"px{i + 1},{e}"] == True:
                            positions[i][0] = e
                            break
                    
                    # Extract y position
                    for f in range(bin_height - rectangles[i][1] + 1):
                        if f == 0 and result[f"py{i + 1},{f}"] == True:
                            positions[i][1] = 0
                            break
                        elif f > 0 and result[f"py{i + 1},{f - 1}"] == False and result[f"py{i + 1},{f}"] == True:
                            positions[i][1] = f
                            break
                
                # Filter out empty bins
                used_bins = [bin_items for bin_items in bins_assignment if bin_items]
                best_solution = (used_bins, positions)
                ub = mid - 1
                
            else:
                print("UNSAT")
                lb = mid + 1
    return best_solution

if __name__ == "__main__":
    # Controller mode
    if len(sys.argv) == 1:
        # Create BPP_C1 folder if it doesn't exist
        
        # Read existing Excel file to check completed instances
        excel_file = 'BPP_INC_C1.xlsx'
        if os.path.exists(excel_file):
            try:
                existing_df = pd.read_excel(excel_file)
                completed_instances = existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else []
            except:
                existing_df = pd.DataFrame()
                completed_instances = [
    # CL_10_100_x (10 instances)
    "CL_10_100_1", "CL_10_100_2", "CL_10_100_3", "CL_10_100_4", "CL_10_100_5",
    "CL_10_100_6", "CL_10_100_7", "CL_10_100_8", "CL_10_100_9", "CL_10_100_10"
]
        else:
            existing_df = pd.DataFrame()
            completed_instances = []
        
        # Set timeout
        TIMEOUT = 900  
        
        # Start from instance 1 (skip index 0 which is empty)
        for instance_id in range(1, len(instances)):
            instance_name = instances[instance_id]
            
            # Skip if already completed
            if instance_name in completed_instances:
                print(f"\nSkipping instance {instance_id}: {instance_name} (already completed)")
                continue
                
            print(f"\n{'=' * 50}")
            print(f"Running instance {instance_id}: {instance_name}")
            print(f"{'=' * 50}")
            
            # Clean up previous result files
            for temp_file in [f'results_BPP_INC_C1_{instance_id}.json', f'checkpoint_BPP_INC_C1_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Run instance with timeout
            command = f"./runlim -r  {TIMEOUT} python3 BPP_INC_C1.py {instance_id}"
            
            try:
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(1)
                
                # Check results
                result = None
                
                if os.path.exists(f'results_BPP_INC_C1_{instance_id}.json'):
                    with open(f'results_BPP_INC_C1_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                elif os.path.exists(f'checkpoint_BPP_INC_C1_{instance_id}.json'):
                    with open(f'checkpoint_BPP_INC_C1_{instance_id}.json', 'r') as f:
                        result = json.load(f)
                    result['Status'] = 'TIMEOUT'
                    result['Instance'] = instance_name
                    print(f"Instance {instance_name} timed out. Using checkpoint data.")
                
                # Process results
                if result:
                    print(f"Instance {instance_name} - Status: {result['Status']}")
                    print(f"Optimal Bins: {result['Optimal_Bins']}, Runtime: {result['Runtime']}")
                    
                    if result['Status'] == 'TIMEOUT':
                        result['Runtime'] = "TIMEOUT"
                        if 'Instance' not in result:
                            result['Instance'] = instance_name
                        
                        # Update Excel
                        if os.path.exists(excel_file):
                            try:
                                existing_df = pd.read_excel(excel_file)
                                result_df = pd.DataFrame([result])
                                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                            except:
                                existing_df = pd.DataFrame([result])
                        else:
                            existing_df = pd.DataFrame([result])
                        
                        existing_df.to_excel(excel_file, index=False)
                        print(f"Timeout results saved to {excel_file}")
                else:
                    print(f"No results found for instance {instance_name}")
                    
            except Exception as e:
                print(f"Error running instance {instance_name}: {str(e)}")
            
            # Clean up temp files
            for temp_file in [f'results_BPP_INC_C1_{instance_id}.json', f'checkpoint_BPP_INC_C1_{instance_id}.json']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        print(f"\nAll instances completed. Results saved to {excel_file}")
    
    # Single instance mode
    else:
        instance_id = int(sys.argv[1])
        instance_name = instances[instance_id]
        
        start = timeit.default_timer()
        
        try:
            print(f"\nProcessing instance {instance_name}")
            
            # Reset global variables
            best_num_bins = float('inf')
            best_solution = None
            optimal_bins = float('inf')
            optimal_solution = None
            
            # Read input
            input_data = read_file_instance(instance_name)
            n_items = int(input_data[0])
            bin_size = input_data[1].split()
            bin_width = int(bin_size[0])
            bin_height = int(bin_size[1])
            rectangles = [[int(val) for val in line.split()] for line in input_data[2:2 + n_items]]
            
            # Calculate bounds
            lower_bound = calculate_lower_bound(bin_width, bin_height, rectangles)
            upper_bound = first_fit_upper_bound(rectangles, bin_width, bin_height)
            
            print(f"Solving 2D Bin Packing for instance {instance_name}")
            print(f"Bin size: {bin_width} x {bin_height}")
            print(f"Number of items: {n_items}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            
            # Solve
            final_bins = OPP(rectangles, upper_bound, bin_width, bin_height)
            
            stop = timeit.default_timer()
            runtime = stop - start
            optimal_solution = final_bins if final_bins else None
            print(f"Optimal number of bins found: {final_bins}")
            optimal_bins = best_num_bins if best_num_bins != float('inf') else upper_bound

            # Display solution
            if optimal_solution:
                bins_assignment, positions = optimal_solution
                display_solution(bin_width, bin_height, rectangles, bins_assignment, positions, instance_name)

            # Create result
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': runtime,
                'Optimal_Bins': optimal_bins if optimal_bins != float('inf') else upper_bound,
                'Status': 'COMPLETE'
            }
            
            # Save to Excel
            excel_file = 'BPP_INC_C1.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
            
            # Save JSON result for controller
            with open(f'results_BPP_INC_C1_{instance_id}.json', 'w') as f:
                json.dump(result, f)
            
            print(f"Instance {instance_name} completed - Runtime: {runtime:.2f}s, Bins: {optimal_bins}")

        except Exception as e:
            print(f"Error in instance {instance_name}: {str(e)}")
            current_bins = best_num_bins if best_num_bins != float('inf') else upper_bound
            result = {
                'Instance': instance_name,
                'Variables': variables_length,
                'Clauses': clauses_length,
                'Runtime': timeit.default_timer() - start,
                'Optimal_Bins': current_bins,
                'Status': 'ERROR'
            }
            
            # Save error result to Excel
            excel_file = 'BPP_INC_C1.xlsx'
            if os.path.exists(excel_file):
                try:
                    existing_df = pd.read_excel(excel_file)
                    instance_exists = instance_name in existing_df['Instance'].tolist() if 'Instance' in existing_df.columns else False
                    
                    if instance_exists:
                        instance_idx = existing_df.index[existing_df['Instance'] == instance_name].tolist()[0]
                        for key, value in result.items():
                            existing_df.at[instance_idx, key] = value
                    else:
                        result_df = pd.DataFrame([result])
                        existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                except:
                    existing_df = pd.DataFrame([result])
            else:
                existing_df = pd.DataFrame([result])
            
            existing_df.to_excel(excel_file, index=False)
            print(f"Error results saved to {excel_file}")
            
            with open(f'results_BPP_INC_C1_{instance_id}.json', 'w') as f:
                json.dump(result, f)