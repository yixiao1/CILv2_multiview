#!/bin/bash

ROOT1="/data-net/ted/extra_data_diego"
ROOT2="/data-net/ted/dataset_ainoa/binary_masks_gaze_predictions"

# Specify the datasets you want to process
DATASETS=("ted_carla0914_additionalweathers" "ted_carla0914_fps10_dense_town01_3cam_rgb_depth_ss_960x540")
# ted_carla0914_additionalweathers:                            258501 frames
# ted_carla0914_fps10_dense_town01_3cam_rgb_depth_ss_960x540:  247514 frames

echo "Processing datasets: ${DATASETS[@]}"

for dataset_name in "${DATASETS[@]}"; do
    echo "Processing ${dataset_name} ..."
    dataset_dir="$ROOT1/$dataset_name"
    
    # Check if dataset exists in ROOT1
    if [[ ! -d "$dataset_dir" ]]; then
        echo "Warning: Dataset '$dataset_name' not found in ROOT1, skipping..."
        continue
    fi
    
    # Check if dataset exists in ROOT2
    if [[ ! -d "$ROOT2/$dataset_name" ]]; then
        echo "Warning: Dataset '$dataset_name' not found in ROOT2, skipping..."
        continue
    fi
    
    echo "Processing dataset: $dataset_name"
    
    for weather_dir in "$dataset_dir"/*; do
        if [[ ! -d "$weather_dir" ]]; then continue; fi
        
        weather=$(basename "$weather_dir")
        echo "  Processing weather: $weather"
        
        for route_dir in "$weather_dir"/*; do
            if [[ ! -d "$route_dir" ]]; then continue; fi
            
            route=$(basename "$route_dir")
            
            # Check if corresponding directory exists in ROOT2
            source_dir="$ROOT2/$dataset_name/$weather/$route"
            if [[ -d "$source_dir" ]]; then
                echo "    Processing route: $route"
                
                # Create symlinks for each gaze prediction file
                gaze_count=0
                for gaze_file in "$source_dir"/gaze_pred*.png; do
                    if [[ -f "$gaze_file" ]]; then
                        filename=$(basename "$gaze_file")
                        target_path="$route_dir/$filename"
                        
                        # Check if symlink already exists
                        if [[ -L "$target_path" ]]; then
                            echo "      Symlink already exists: $filename"
                        else
                            ln -s "$gaze_file" "$target_path"
                            ((gaze_count++))
                        fi
                    fi
                done
                echo "      Created $gaze_count gaze prediction symlinks"
            else
                echo "    Warning: Route '$route' not found in ROOT2/$dataset_name/$weather/"
            fi
        done
    done
    echo "Completed dataset: $dataset_name"
    echo ""
done

echo "All specified datasets processed!"