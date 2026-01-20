#!/bin/bash
#
# Backup Script - Move files to backup while preserving directory structure
# 
# Usage: ./backup_files.sh <files_to_delete.txt> <root_directory> [backup_directory]
#
# Example:
#   ./backup_files.sh files_to_delete.txt /data-net/ted/extra_data_diego
#   ./backup_files.sh files_to_delete.txt /data-net/ted/extra_data_diego /path/to/custom/backup
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <files_list> <root_directory> [backup_directory]; a /backup subdirectory will be created in root_directory if not specified."
    echo ""
    echo "Example:"
    echo "  $0 files_to_delete.txt /data-net/ted/extra_data_diego"
    echo ""
    exit 1
fi

FILES_LIST="$1"
ROOT_DIR="$2"
BACKUP_DIR="${3:-${ROOT_DIR}/backup}"

# Validate inputs
if [ ! -f "$FILES_LIST" ]; then
    echo "Error: File list not found: $FILES_LIST"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory not found: $ROOT_DIR"
    exit 1
fi

# Count files
TOTAL_FILES=$(wc -l < "$FILES_LIST")

echo "=============================================================================="
echo "Backup Script - Preserving Directory Structure"
echo "=============================================================================="
echo "Files list:       $FILES_LIST"
echo "Root directory:   $ROOT_DIR"
echo "Backup directory: $BACKUP_DIR"
echo "Files to move:    $TOTAL_FILES"
echo "=============================================================================="
echo ""

# Ask for confirmation
read -p "Proceed with backup? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Counter for progress
COUNT=0
MOVED=0
FAILED=0

# Process each file
while IFS= read -r file; do
    COUNT=$((COUNT + 1))
    
    # Calculate target path (preserve structure relative to root)
    target="$BACKUP_DIR/${file#$ROOT_DIR/}"
    
    # Create target directory
    mkdir -p "$(dirname "$target")"
    
    # Move file
    if [ -f "$file" ]; then
        if mv "$file" "$target"; then
            MOVED=$((MOVED + 1))
        else
            echo "Failed to move: $file"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "File not found (skipping): $file"
        FAILED=$((FAILED + 1))
    fi
    
    # Progress indicator
    if [ $((COUNT % 100)) -eq 0 ]; then
        echo "Progress: $COUNT / $TOTAL_FILES files processed..."
    fi
done < "$FILES_LIST"

echo ""
echo "=============================================================================="
echo "Backup Complete"
echo "=============================================================================="
echo "Total files:      $TOTAL_FILES"
echo "Successfully moved: $MOVED"
echo "Failed/Missing:   $FAILED"
echo "Backup location:  $BACKUP_DIR"
echo "=============================================================================="
echo ""
echo "To restore all files, run:"
echo "  cd \"$BACKUP_DIR\" && find . -type f -exec sh -c 'mkdir -p \"\$(dirname \"$ROOT_DIR/\$1\")\" && mv \"\$1\" \"$ROOT_DIR/\$1\"' _ {} \\;"
echo ""