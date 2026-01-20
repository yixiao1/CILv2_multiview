#!/usr/bin/env python3
"""
Script to check for frame number mismatches between different file types in a dataset.

This script scans a hierarchical dataset structure and verifies that all specified
file prefixes have the same frame numbers for each route.
"""

import os
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import re
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Root folder of the dataset
root_folder = '/data-net/ted/extra_data_diego'
# root_folder = '/data/122-2/Datasets/IL_multiview'

# List of subdatasets to check (folders inside root_folder)
# Set to None or empty list to check all subdatasets
subdataset_list = [
    'ted_carla0914_additionalweathers',
    'ted_carla0914_fps10_dense_town01_3cam_rgb_depth_ss_960x540',
    'ted_carla0914_valid_Town02_busy_clearnoon'
    # Add more subdatasets as needed
]

# List of file prefixes to check (without frame numbers)
# Example: 'rgb_front' will match 'rgb_front000000.png', 'rgb_front000001.png', etc.
prefixes_to_check = [
    'cmd_fix_can_bus',
    # 'rgb_front'
    'scout14ep1',
    'scout15ep2'
    # Add more prefixes as needed
]
# prefixes_to_check = ['conti_front', 'att_mask_ss_hat_conti_front']
# prefixex_to_check = ['sekonix_60', 'att_mask_ss_hat_sekonix_60',]
# prefixex_to_check = ['sekonix_120', 'att_mask_ss_hat_sekonix_120',]

# Number of processes for parallel processing
num_processes = None  # None = use all available CPUs

# Output file for list of files to delete
# The script finds the intersection of frames across all prefixes (lowest common denominator)
# and outputs paths of files that need to be deleted to achieve consistency
output_file = 'files_to_delete_cilvw_conti_front.txt'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_frame_number(filename):
    """
    Extract frame number from filename like 'prefix000123.ext'
    
    Args:
        filename: Name of the file (without path)
    
    Returns:
        int: Frame number, or None if no match
    """
    # Match digits before the file extension
    match = re.search(r'(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None


def get_frames_for_prefix(route_path, prefix):
    """
    Get all frame numbers and file paths for a given prefix in a route directory.
    
    Args:
        route_path: Path to the route directory
        prefix: File prefix to search for
    
    Returns:
        tuple: (sorted list of frame numbers, dict mapping frame_num -> filepath)
    """
    pattern = os.path.join(route_path, f"{prefix}*")
    files = glob(pattern)
    
    frame_to_path = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        frame_num = extract_frame_number(filename)
        if frame_num is not None:
            frame_to_path[frame_num] = filepath
    
    return sorted(frame_to_path.keys()), frame_to_path


def check_route(route_path):
    """
    Check a single route for frame mismatches between different prefixes.
    
    Args:
        route_path: Path to the route directory
    
    Returns:
        dict or None: Mismatch information if found, None otherwise
    """
    # Get frames and file paths for each prefix
    prefix_frames = {}
    prefix_paths = {}
    
    for prefix in prefixes_to_check:
        frames, frame_to_path = get_frames_for_prefix(route_path, prefix)
        if frames:  # Only include if files were found
            prefix_frames[prefix] = set(frames)
            prefix_paths[prefix] = frame_to_path
    
    # Need at least 2 prefixes to compare
    if len(prefix_frames) < 2:
        return None
    
    # Calculate intersection of all frame sets (frames present in ALL prefixes)
    all_frame_sets = list(prefix_frames.values())
    common_frames = set.intersection(*all_frame_sets)
    
    # If all prefixes have the same frames, no mismatch
    if all(frames == common_frames for frames in prefix_frames.values()):
        return None
    
    # Find files to delete (files that are NOT in the common intersection)
    files_to_delete = []
    prefix_info = {}
    
    for prefix in prefix_frames:
        extra_frames = prefix_frames[prefix] - common_frames
        if extra_frames:
            # Get file paths for extra frames
            extra_files = [prefix_paths[prefix][frame] for frame in sorted(extra_frames)]
            files_to_delete.extend(extra_files)
            
            prefix_info[prefix] = {
                'total_frames': len(prefix_frames[prefix]),
                'extra_frames': sorted(extra_frames),
                'extra_files': extra_files,
                'range': (min(prefix_frames[prefix]), max(prefix_frames[prefix]))
            }
    
    # Compare all pairs of prefixes for detailed reporting
    mismatches = []
    prefix_list = list(prefix_frames.keys())
    
    for i in range(len(prefix_list)):
        for j in range(i + 1, len(prefix_list)):
            prefix1, prefix2 = prefix_list[i], prefix_list[j]
            frames1, frames2 = prefix_frames[prefix1], prefix_frames[prefix2]
            
            # Check if frame sets differ
            if frames1 != frames2:
                only_in_1 = frames1 - frames2
                only_in_2 = frames2 - frames1
                
                mismatches.append({
                    'prefix1': prefix1,
                    'prefix2': prefix2,
                    'count1': len(frames1),
                    'count2': len(frames2),
                    'only_in_1': sorted(only_in_1),
                    'only_in_2': sorted(only_in_2),
                    'range1': (min(frames1), max(frames1)) if frames1 else (None, None),
                    'range2': (min(frames2), max(frames2)) if frames2 else (None, None),
                })
    
    return {
        'route_path': route_path,
        'common_frames': sorted(common_frames),
        'prefix_info': prefix_info,
        'files_to_delete': files_to_delete,
        'mismatches': mismatches
    }


def find_all_routes_organized(root_folder, subdataset_filter=None):
    """
    Find all route directories in the dataset, organized by subdataset and weather.
    
    Args:
        root_folder: Root directory of the dataset
        subdataset_filter: List of subdataset names to include, or None for all
    
    Returns:
        dict: Nested dict structure {subdataset: {weather: [route_paths]}}
    """
    organized_routes = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(root_folder):
        print(f"Error: Root folder does not exist: {root_folder}")
        return organized_routes
    
    # Walk through subdatasets
    for subdataset in sorted(os.listdir(root_folder)):
        subdataset_path = os.path.join(root_folder, subdataset)
        
        if not os.path.isdir(subdataset_path):
            continue
        
        # Apply subdataset filter
        if subdataset_filter and subdataset not in subdataset_filter:
            continue
        
        # Walk through weather conditions
        for weather in sorted(os.listdir(subdataset_path)):
            weather_path = os.path.join(subdataset_path, weather)
            
            if not os.path.isdir(weather_path):
                continue
            
            # Find route directories (must contain "route" in name)
            for route in sorted(os.listdir(weather_path)):
                if 'route' in route.lower():
                    route_path = os.path.join(weather_path, route)
                    if os.path.isdir(route_path):
                        organized_routes[subdataset][weather].append(route_path)
    
    return organized_routes


def format_frame_list(frames, max_display=10):
    """
    Format a list of frame numbers for display.
    
    Args:
        frames: List of frame numbers
        max_display: Maximum number of frames to display
    
    Returns:
        str: Formatted string
    """
    if not frames:
        return "none"
    
    if len(frames) <= max_display:
        return str(frames)
    else:
        displayed = frames[:max_display]
        remaining = len(frames) - max_display
        return f"{displayed} ... and {remaining} more"


def parse_route_path(route_path, root_folder):
    """
    Parse route path into subdataset, weather, and route name.
    
    Args:
        route_path: Full path to route
        root_folder: Root folder path
    
    Returns:
        tuple: (subdataset, weather, route_name)
    """
    relative_path = route_path.replace(root_folder, '').strip('/')
    parts = relative_path.split('/')
    
    if len(parts) >= 3:
        # parts[0] = subdataset, parts[1] = weather, parts[2] = route
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        # parts[0] = weather, parts[1] = route (no subdataset)
        return "Default", parts[0], parts[1]
    else:
        return "Unknown", "Unknown", os.path.basename(route_path)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the frame mismatch check."""
    
    print("=" * 80)
    print("Frame Mismatch Checker")
    print("=" * 80)
    print(f"Dataset root: {root_folder}")
    if subdataset_list:
        print(f"Subdatasets:  {', '.join(subdataset_list)}")
    else:
        print(f"Subdatasets:  All")
    print(f"Checking prefixes: {', '.join(prefixes_to_check)}")
    print("=" * 80)
    print()
    
    # Find all routes organized by subdataset and weather
    print("Scanning for routes...")
    organized_routes = find_all_routes_organized(root_folder, subdataset_list)
    
    if not organized_routes:
        print("No routes found! Check that the root folder path and subdataset list are correct.")
        return
    
    # Count total routes
    total_routes = sum(
        len(routes)
        for weather_dict in organized_routes.values()
        for routes in weather_dict.values()
    )
    
    # Count weather conditions
    total_weather_conditions = sum(
        len(weather_dict)
        for weather_dict in organized_routes.values()
    )
    
    print(f"Found {total_routes} routes across {total_weather_conditions} weather condition(s)")
    print()
    
    # Check routes with progress bars
    print("Checking routes for frame mismatches...")
    print()
    
    all_results = []
    all_files_to_delete = []
    processes = num_processes if num_processes else os.cpu_count()
    
    # Outer progress bar for weather conditions
    weather_pbar = tqdm(
        total=total_weather_conditions,
        desc="Weather conditions",
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    for subdataset, weather_dict in organized_routes.items():
        for weather, routes in weather_dict.items():
            weather_pbar.set_description(f"Checking {subdataset}/{weather}")
            
            # Inner progress bar for routes within this weather condition
            with tqdm(
                total=len(routes),
                desc=f"  Routes in {weather}",
                position=1,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            ) as route_pbar:
                
                # Process routes in parallel with progress tracking
                with Pool(processes=processes) as pool:
                    for result in pool.imap_unordered(check_route, routes):
                        if result is not None:
                            all_results.append(result)
                            all_files_to_delete.extend(result['files_to_delete'])
                        route_pbar.update(1)
            
            weather_pbar.update(1)
    
    weather_pbar.close()
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    if not all_results:
        print("✓ No mismatches found! All routes have consistent frame counts.")
        print()
        return
    
    print(f"Found {len(all_results)} route(s) with mismatches")
    print(f"Total files to delete: {len(all_files_to_delete)}\n")
    
    # Write files to delete to output file
    with open(output_file, 'w') as f:
        for filepath in sorted(all_files_to_delete):
            f.write(filepath + '\n')
    
    print(f"✓ Written list of files to delete to: {output_file}")
    print()
    
    # Group results by subdataset and weather for organized output
    grouped_results = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        route_path = result['route_path']
        subdataset, weather, route_name = parse_route_path(route_path, root_folder)
        grouped_results[subdataset][weather].append(result)
    
    # Print detailed results organized by subdataset and weather
    mismatch_num = 1
    for subdataset in sorted(grouped_results.keys()):
        for weather in sorted(grouped_results[subdataset].keys()):
            results_in_weather = grouped_results[subdataset][weather]
            
            print(f"\n{'═' * 80}")
            print(f"SubDataset: {subdataset} | Weather: {weather}")
            print(f"Found {len(results_in_weather)} mismatched route(s)")
            print(f"{'═' * 80}")
            
            for result in results_in_weather:
                route_path = result['route_path']
                _, _, route_name = parse_route_path(route_path, root_folder)
                common_frames = result['common_frames']
                prefix_info = result['prefix_info']
                
                print(f"\n{'─' * 80}")
                print(f"MISMATCH #{mismatch_num}: {route_name}")
                print(f"{'─' * 80}")
                print(f"Full Path: {route_path}")
                print(f"Common frames: {len(common_frames)} frames ({min(common_frames)} to {max(common_frames)})")
                print()
                
                # Show summary for each prefix
                for prefix, info in prefix_info.items():
                    extra_count = len(info['extra_frames'])
                    if extra_count > 0:
                        print(f"  {prefix}:")
                        print(f"    Total: {info['total_frames']} files (range: {info['range'][0]} to {info['range'][1]})")
                        print(f"    Extra: {extra_count} file(s) to delete - frames: {format_frame_list(info['extra_frames'])}")
                
                print()
                
                # Show detailed comparisons
                for mismatch in result['mismatches']:
                    prefix1 = mismatch['prefix1']
                    prefix2 = mismatch['prefix2']
                    count1 = mismatch['count1']
                    count2 = mismatch['count2']
                    range1 = mismatch['range1']
                    range2 = mismatch['range2']
                    only_in_1 = mismatch['only_in_1']
                    only_in_2 = mismatch['only_in_2']
                    
                    print(f"  Comparing: {prefix1} vs {prefix2}")
                    print(f"  ├─ {prefix1:20s}: {count1:5d} files (frames {range1[0]:6d} to {range1[1]:6d})")
                    print(f"  └─ {prefix2:20s}: {count2:5d} files (frames {range2[0]:6d} to {range2[1]:6d})")
                    print()
                    
                    if only_in_1:
                        print(f"     Missing in {prefix2}: {format_frame_list(only_in_1)}")
                    
                    if only_in_2:
                        print(f"     Missing in {prefix1}: {format_frame_list(only_in_2)}")
                    
                    print()
                
                mismatch_num += 1
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total routes checked:       {total_routes}")
    print(f"Routes with mismatches:     {len(all_results)}")
    print(f"Routes OK:                  {total_routes - len(all_results)}")
    print(f"Total files to delete:      {len(all_files_to_delete)}")
    print(f"Output file:                {os.path.abspath(output_file)}")
    print("=" * 80)
    print()
    print("To delete all mismatched files, run:")
    print(f"  while IFS= read -r file; do rm \"$file\"; done < {output_file}")
    print("  # Or to move to a backup directory:")
    print(f"  mkdir -p backup && while IFS= read -r file; do mv \"$file\" backup/; done < {output_file}")
    print()


if __name__ == "__main__":
    main()
