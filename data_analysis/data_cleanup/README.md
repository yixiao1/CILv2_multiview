# Frame Mismatch Checker - README

## Purpose

This script checks for frame number mismatches between different file types in a dataset and generates a list of files to delete to achieve consistency across all prefixes.

## How It Works

### The "Lowest Common Denominator" Rule

The script follows a simple principle: **all prefixes must have exactly the same frames**.

For example, if you're checking:
- `cmd_fix_can_bus` (1259 files, frames 0-1258)
- `scout14ep1` (1260 files, frames 0-1259)
- `scout15ep2` (1260 files, frames 0-1259)

The script will:
1. Find the **intersection** of all frame sets: frames 0-1258 (present in ALL prefixes)
2. Identify **extra frames** not in the intersection:
   - `scout14ep1`: frame 1259 is extra
   - `scout15ep2`: frame 1259 is extra
3. Generate the full file paths for deletion:
   ```
   /path/to/route/scout14ep1_001259.png
   /path/to/route/scout15ep2_001259.png
   ```

## Configuration

Edit the script to configure:

```python
# Root folder and subdatasets to check
root_folder = '/data-net/ted/extra_data_diego'
subdataset_list = ['ted_carla0914_additionalweathers']

# File prefixes to compare
prefixes_to_check = [
    'cmd_fix_can_bus',
    'scout14ep1',
    'scout15ep2',
]

# Output file
output_file = 'files_to_delete.txt'
```

## Running the Script

```bash
python check_frame_mismatches.py
```

The script will:
1. Scan the dataset structure
2. Show progress bars for weather conditions and routes
3. Print detailed mismatch information to console
4. Write all file paths to delete to `files_to_delete.txt`

## Output File Format

`files_to_delete.txt` contains one file path per line:

```
/data-net/ted/extra_data_diego/ted_carla0914_additionalweathers/ClearNoon/ClearNoon_route00024/scout14ep1_001259.png
/data-net/ted/extra_data_diego/ted_carla0914_additionalweathers/ClearNoon/ClearNoon_route00024/scout15ep2_001259.png
/data-net/ted/extra_data_diego/ted_carla0914_additionalweathers/ClearSunset/ClearSunset_route00005/rgb_front_004567.jpg
...
```

## Deleting Files

⚠️ **IMPORTANT**: Always backup your data before deleting!

### Option 1: Use the backup script (EASIEST & RECOMMENDED)

I've included helper scripts for easy backup and restore:

```bash
# Make scripts executable
chmod +x backup_files.sh restore_files.sh

# Backup files (preserves directory structure)
./backup_files.sh files_to_delete.txt /data-net/ted/extra_data_diego

# If you need to restore
./restore_files.sh /data-net/ted/extra_data_diego/backup
```

The backup script will:
- Ask for confirmation before proceeding
- Show progress every 100 files
- Preserve the complete directory structure
- Report success/failure statistics

### Option 2: Manual backup with preserved structure

```bash
# Set backup directory (creates backup folder next to your dataset)
BACKUP_DIR="/data-net/ted/extra_data_diego/backup"

# Move files while preserving directory structure
while IFS= read -r file; do
    target="$BACKUP_DIR/${file#/data-net/ted/extra_data_diego/}"
    mkdir -p "$(dirname "$target")"
    mv "$file" "$target"
done < files_to_delete.txt
```

**Example**: 
- Original: `/data-net/ted/extra_data_diego/ted_carla0914_additionalweathers/ClearNight/ClearNight_route00000/scout14ep1_004905.png`
- Backed up to: `/data-net/ted/extra_data_diego/backup/ted_carla0914_additionalweathers/ClearNight/ClearNight_route00000/scout14ep1_004905.png`

### Option 3: Delete files directly (DANGEROUS - no recovery possible)
```bash
while IFS= read -r file; do rm "$file"; done < files_to_delete.txt
```

### Restoring from backup (manual method)

If you need to restore the files manually:

```bash
# Restore all backed up files to their original locations
BACKUP_DIR="/data-net/ted/extra_data_diego/backup"

# Find all files and restore them (they already have absolute paths)
find "$BACKUP_DIR" -type f | while read -r file; do
    original="${file#$BACKUP_DIR/}"
    original="/${original}"
    mkdir -p "$(dirname "$original")"
    mv "$file" "$original"
done
```

### Option 4: Review files first
```bash
# Count how many files will be deleted
wc -l files_to_delete.txt

# Check specific routes
grep "ClearNoon_route00024" files_to_delete.txt

# Dry run (see what would be deleted without deleting)
while IFS= read -r file; do echo "Would delete: $file"; done < files_to_delete.txt

# Check total size of files to be deleted
while IFS= read -r file; do stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null; done < files_to_delete.txt | awk '{sum+=$1} END {print "Total size:", sum/1024/1024/1024, "GB"}'
```

## Console Output

The script provides detailed information:

```
════════════════════════════════════════════════════════════════════════════════
SubDataset: ted_carla0914_additionalweathers | Weather: ClearNoon
Found 5 mismatched route(s)
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
MISMATCH #1: ClearNoon_route00024
────────────────────────────────────────────────────────────────────────────────
Full Path: /path/to/route
Common frames: 1259 frames (0 to 1258)

  scout14ep1:
    Total: 1260 files (range: 0 to 1259)
    Extra: 1 file(s) to delete - frames: [1259]
  
  scout15ep2:
    Total: 1260 files (range: 0 to 1259)
    Extra: 1 file(s) to delete - frames: [1259]

  Comparing: cmd_fix_can_bus vs scout14ep1
  ├─ cmd_fix_can_bus     :  1259 files (frames      0 to   1258)
  └─ scout14ep1          :  1260 files (frames      0 to   1259)

     Missing in cmd_fix_can_bus: [1259]
```

## Summary

At the end, you'll see:

```
════════════════════════════════════════════════════════════════════════════════
SUMMARY
════════════════════════════════════════════════════════════════════════════════
Total routes checked:       450
Routes with mismatches:     23
Routes OK:                  427
Total files to delete:      156
Output file:                /current/directory/files_to_delete.txt
════════════════════════════════════════════════════════════════════════════════
```

## Requirements

```bash
pip install tqdm
```

## Troubleshooting

**Q: No routes found?**  
A: Check that `root_folder` and `subdataset_list` are correct.

**Q: Wrong files being marked for deletion?**  
A: The script finds the intersection of ALL prefixes. If one prefix has fewer frames, those frames will be deleted from all other prefixes.

**Q: How to exclude certain prefixes?**  
A: Simply remove them from the `prefixes_to_check` list in the configuration.