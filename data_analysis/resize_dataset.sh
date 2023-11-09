#!/bin/bash

# This function will be executed in parallel
process_image() {
    file="$1"
    resolution="$2"
    skip_string="$3"
    skip_extension="$4"
    include_string="$5"

    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename_no_ext="${filename%.*}"

    # Skip files based on user options
    if [[ $filename == resized_* ]]; then
        return
    fi

    if [[ ! -z "$skip_string" && $filename == *"$skip_string"* ]]; then
        return
    fi

    if [[ ! -z "$skip_extension" && $file == *"$skip_extension" ]]; then
        return
    fi

    if [[ ! -z "$include_string" && $filename != *"$include_string"* ]]; then
        return
    fi
    
    # Resize image and save with prefix 'resized' in the original subdirectory
    convert "$file" -resize "$resolution" "$(dirname $file)/resized_$filename_no_ext.$extension"
}

# Initialize variables
skip_string=""
skip_extension=""
include_string=""

# Parse options
while getopts ":s:e:i:" opt; do
  case $opt in
    s)
      skip_string=$OPTARG
      ;;
    e)
      skip_extension=$OPTARG
      ;;
    i)
      include_string=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Remove the options from the positional parameters
shift $((OPTIND-1))

# Check if directory path and resolution are provided as arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 [-s skip_string] [-e skip_extension] [-i include_string] <directory_path> <resolution>"
    exit 1
fi

# Assign arguments to variables
dir_path=$1
resolution=$2

# Check if directory exists and is accessible
if [ ! -d "$dir_path" ]; then
    echo "Error: Directory $dir_path does not exist or is not accessible."
    exit 1
fi

# Count total files that will be processed
total_files=$(find "$dir_path" -type f ! -name "$skip_ext" ! -name "*$skip_string*" ! -name "resized_*" \( -iname \*.png -o -iname \*.jpg \) | wc -l)

# Using GNU parallel to process images concurrently
export -f process_image
find "$dir_path" -type f \( -iname \*.png -o -iname \*.jpg \) | tqdm --total $total_files | 
    parallel process_image {} $resolution $skip_string $skip_extension $include_string

echo "All matching images in $dir_path have been resized to $resolution and saved therein."
