#!/usr/bin/env python3
"""
Export locality matrix from ptx_tracing.py output
"""

import sys
import os
import numpy as np
import json
from PIL import Image

def extract_from_heatmap_data(base_path, kernel_name, grid_dims):
    """Try to load the raw matrix data"""
    # Try to find existing matrix files
    possible_files = [
        f"{base_path}/{kernel_name}_{grid_dims}_matrix.json",
        f"{base_path}/{kernel_name}_{grid_dims}_matrix.npy",
        f"{base_path}/{kernel_name}_{grid_dims}.json",
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"[Found] {filepath}")
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return np.array(data)
            elif filepath.endswith('.npy'):
                return np.load(filepath)
    
    print("[Error] No matrix file found")
    print("Available files:")
    if os.path.exists(base_path):
        for f in os.listdir(base_path):
            print(f"  - {f}")
    
    return None

if __name__ == "__main__":
    # For your vector addition
    matrix = extract_from_heatmap_data(
        "img/vector",
        "_Z9vectorAddPiS_S_i",
        "256-1"
    )
    
    if matrix is not None:
        # Save as JSON for validation
        output_file = "vector_locality_matrix.json"
        with open(output_file, 'w') as f:
            json.dump(matrix.tolist(), f)
        print(f"[Success] Saved matrix to {output_file}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Total sharing: {np.sum(matrix)}")
    else:
        print("[Error] Could not extract matrix")
        print("\nPlease modify ptx_tracing.py to save the matrix.")
        print("Add after generating heatmap:")
        print("  import json")
        print("  with open(f'img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}_matrix.json', 'w') as f:")
        print("      json.dump(kernel_map, f)")
