#!/usr/bin/env python3
import sys
import csv
import os

def has_nonzero_lds_bank_conflict(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Use csv.reader to properly parse quoted CSV fields
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, start=1):
                try:
                    # Look for "SQ_LDS_BANK_CONFLICT" in the row
                    if "SQ_LDS_BANK_CONFLICT" in row:
                        idx = row.index("SQ_LDS_BANK_CONFLICT")
                        # Next field should be the value
                        if idx + 1 < len(row):
                            val_str = row[idx + 1].strip()
                            # Try to parse as float (handles 0.00000000e+00, 1.0, etc.)
                            try:
                                val = float(val_str)
                                if val != 0.0:
                                    # Return file name, line number, full line text, and value
                                    f.seek(0)  # Rewind to get raw line
                                    lines = f.readlines()
                                    raw_line = lines[line_num - 1].rstrip('\n\r')
                                    return True, line_num, raw_line, val
                            except ValueError:
                                # Skip malformed numbers (e.g., empty, non-numeric)
                                pass
                except Exception as e:
                    print(f"[Warn] Error parsing line {line_num} in {filepath}: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"[Error] Could not read {filepath}: {e}", file=sys.stderr)
    return False, None, None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_lds_conflict.py <file1> <file2> ...", file=sys.stderr)
        print("Example: python check_lds_conflict.py *.csv", file=sys.stderr)
        sys.exit(1)

    found_any = False
    for filepath in sys.argv[1:]:
        if not os.path.isfile(filepath):
            print(f"[Skip] Not a file: {filepath}", file=sys.stderr)
            continue

        has_conflict, line_num, line_text, value = has_nonzero_lds_bank_conflict(filepath)
        if has_conflict:
            found_any = True
            print(f"{os.path.basename(filepath)}")
            print(f"  Line {line_num}: {line_text}")
            print(f"  → SQ_LDS_BANK_CONFLICT = {value}")
            print()  # blank line for readability

    if not found_any:
        print("✅ All files have SQ_LDS_BANK_CONFLICT = 0")

if __name__ == "__main__":
    main()
