#!/bin/bash
#
# Restore Script - Restore backed up files to their original locations
# 
# Usage: ./restore_files.sh <backup_directory>
#
# Example:
#   ./restore_files.sh /data-net/ted/extra_data_diego/backup
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_directory>"
    echo ""
    echo "Example:"
    echo "  $0 /data-net/ted/extra_data_diego/backup"
    echo ""
    exit 1
fi

BACKUP_DIR="$1"

# Validate inputs
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Count files in backup
TOTAL_FILES=$(find "$BACKUP_DIR" -type f | wc -l)

echo "=============================================================================="
echo "Restore Script - Restoring Files to Original Locations"
echo "=============================================================================="
echo "Backup directory: $BACKUP_DIR"
echo "Files to restore: $TOTAL_FILES"
echo "=============================================================================="
echo ""
echo "Files will be restored to their original absolute paths."
echo ""

# Ask for confirmation
read -p "Proceed with restore? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting restore..."

# Counter for progress
COUNT=0
RESTORED=0
FAILED=0

# Process files - they already have their full absolute paths
while IFS= read -r backup_file; do
    COUNT=$((COUNT + 1))
    
    # The backup file path already contains the full original path structure
    # We just need to move it back to the same location
    original_path="$backup_file"
    original_path="${original_path#$BACKUP_DIR/}"  # Remove backup prefix
    original_path="/${original_path}"  # Add leading slash for absolute path
    
    # Create target directory
    mkdir -p "$(dirname "$original_path")"
    
    # Move file back
    if mv "$backup_file" "$original_path"; then
        RESTORED=$((RESTORED + 1))
    else
        echo "Failed to restore: $backup_file -> $original_path"
        FAILED=$((FAILED + 1))
    fi
    
    # Progress indicator
    if [ $((COUNT % 100)) -eq 0 ]; then
        echo "Progress: $COUNT / $TOTAL_FILES files processed..."
    fi
done < <(find "$BACKUP_DIR" -type f)

echo ""
echo "=============================================================================="
echo "Restore Complete"
echo "=============================================================================="
echo "Total files:          $TOTAL_FILES"
echo "Successfully restored: $RESTORED"
echo "Failed:               $FAILED"
echo "=============================================================================="
echo ""

# Clean up empty directories in backup
if [ $RESTORED -gt 0 ]; then
    echo "Cleaning up empty directories in backup..."
    find "$BACKUP_DIR" -type d -empty -delete 2>/dev/null || true
    echo "Done."
fi

echo ""