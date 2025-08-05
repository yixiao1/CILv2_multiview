#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <experiment_path> <video_name> [video_fps]"
    echo ""
    echo "Arguments:"
    echo "  experiment_path: Path to the experiment directory"
    echo "  video_name:      Name prefix for the output videos"
    echo "  video_fps:       Frame rate for output videos (optional, default: extracted from path or 20)"
    exit 1
}

# Check if ffmpeg is installed
check_ffmpeg() {
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is not installed. Please install ffmpeg to continue."
        exit 1
    fi
}

# Extract FPS from experiment path
extract_fps_from_path() {
    local path="$1"
    if [[ $path =~ _([0-9]+)FPS ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "20"
    fi
}

# Main function
main() {
    if [ $# -lt 2 ] || [ $# -gt 3 ]; then
        usage
    fi

    check_ffmpeg

    experiment_path="$1"
    video_name="$2"
    
    if [ $# -eq 3 ]; then
        video_fps="$3"
    else
        video_fps=$(extract_fps_from_path "$experiment_path")
        echo "No FPS specified, using extracted/default FPS: $video_fps"
    fi

    if [ ! -d "$experiment_path" ]; then
        echo "Error: Experiment path '$experiment_path' does not exist."
        exit 1
    fi

    videos_dir="$experiment_path/videos"
    mkdir -p "$videos_dir"
    echo "Created videos directory: $videos_dir"

    processed_count=0
    failed_count=0

    # Find and process all route directories
    echo "Scanning for route directories..."
    
    # Use a different approach to find directories
    route_dirs=()
    while IFS= read -r -d '' dir; do
        route_dirs+=("$dir")
    done < <(find "$experiment_path" -maxdepth 1 -type d -name "*_route[0-9][0-9][0-9][0-9][0-9]" -print0)

    if [ ${#route_dirs[@]} -eq 0 ]; then
        echo "No route directories found in the experiment path."
        echo "Looking for pattern: *_route[0-9][0-9][0-9][0-9][0-9]"
        echo "Available directories:"
        ls -la "$experiment_path"
        exit 1
    fi

    echo "Found ${#route_dirs[@]} route directories to process."

    # Process each route directory
    for route_dir in "${route_dirs[@]}"; do
        route_basename=$(basename "$route_dir")
        echo "Processing: $route_basename"
        
        if [[ $route_basename =~ ^(.+)_(route[0-9]{5})$ ]]; then
            weather="${BASH_REMATCH[1]}"
            route="${BASH_REMATCH[2]}"
        else
            echo "Warning: Skipping directory with unexpected format: $route_basename"
            continue
        fi

        frames_dir="$route_dir/0"
        if [ ! -d "$frames_dir" ]; then
            echo "Warning: Frames directory not found: $frames_dir"
            continue
        fi

        # Get frame files and sort them properly
        mapfile -t frame_files < <(find "$frames_dir" -name "*.jpg" | sort -V)
        
        if [ ${#frame_files[@]} -eq 0 ]; then
            echo "Warning: No .jpg frames found in: $frames_dir"
            continue
        fi

        output_video="$videos_dir/${video_name}_${weather}_${route}_${video_fps}FPS.mp4"
        
        echo "  Processing ${#frame_files[@]} frames -> $(basename "$output_video")"
        
        # Method 1: Try using image sequence input (simpler and often more reliable)
        # First, check if frames are numbered consecutively
        first_frame="${frame_files[0]}"
        if [[ $first_frame =~ ([0-9]{6})\.jpg$ ]]; then
            frame_pattern="${frames_dir}/%06d.jpg"
            
            # Try the image sequence method first
            if ffmpeg -y -framerate "$video_fps" -i "$frame_pattern" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 18 -c:v libx264 -pix_fmt yuv420p -r "$video_fps" "$output_video" 2>/dev/null; then
                echo "  ✓ Successfully created: $(basename "$output_video")"
                ((processed_count++))
                continue
            fi
        fi
        
        # Method 2: If image sequence fails, use concat method
        echo "  Trying concat method..."
        temp_list=$(mktemp)
        
        # Create concat file with proper escaping
        for frame_file in "${frame_files[@]}"; do
            # Escape single quotes in filenames for ffmpeg
            escaped_file=$(printf "%s" "$frame_file" | sed "s/'/'\\\''/g")
            echo "file '$escaped_file'" >> "$temp_list"
        done

        # Set frame duration for concat method
        frame_duration=$(echo "scale=6; 1.0/$video_fps" | bc -l 2>/dev/null || echo "0.05")
        
        if ffmpeg -y -f concat -safe 0 -i "$temp_list" -vsync vfr -r "$video_fps" -c:v libx264 -pix_fmt yuv420p "$output_video" 2>"$temp_list.log"; then
            echo "  ✓ Successfully created: $(basename "$output_video")"
            ((processed_count++))
        else
            echo "  ✗ Failed to create video for: $route_basename"
            echo "  Error log saved to: $temp_list.log"
            cat "$temp_list.log"
            ((failed_count++))
            # Keep the log file for debugging
            mv "$temp_list.log" "$videos_dir/error_${route_basename}.log"
        fi

        rm -f "$temp_list"
    done

    echo ""
    echo "Video generation complete!"
    echo "Successfully processed: $processed_count routes"
    echo "Failed: $failed_count routes"
    echo "Videos saved in: $videos_dir"
    
    if [ $failed_count -gt 0 ]; then
        echo "Check error logs in $videos_dir for failed conversions"
    fi
}

main "$@"