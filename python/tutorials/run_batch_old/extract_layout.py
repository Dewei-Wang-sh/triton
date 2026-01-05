#!/usr/bin/env python3
import sys
import re
import os

# Regex to match target header line (case-sensitive unless re.IGNORECASE added)
HEADER_PATTERN = re.compile(
    r"TritonAMDGPUCoalesceAsyncCopy \(tritonamdgpu-coalesce-async-copy"
)

# Regex to match lines like:
#   #shared = #ttg ...
#   #shared1 = #ttg ...
#   # shared2 = #ttg ...
SHARED_PATTERN = re.compile(
    r"^#\s*shared\d*\s*=\s*#ttg\b"
)

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[Error] Could not read {filepath}: {e}", file=sys.stderr)
        return []

    n = len(lines)
    for i, line in enumerate(lines):
        if HEADER_PATTERN.search(line):
            # Start scanning up to 20 lines after this
            shared_lines = []
            for j in range(i + 1, min(i + 21, n)):
                if SHARED_PATTERN.match(lines[j]):
                    shared_lines.append(lines[j].rstrip('\n\r'))
                    if len(shared_lines) == 2:
                        return shared_lines
            # If we found only 0 or 1, still return what we have (optional: skip if <2)
            return shared_lines
    return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_layout.py <file1> <file2> ...", file=sys.stderr)
        sys.exit(1)

    output_lines = []
    for filepath in sys.argv[1:]:
        if not os.path.isfile(filepath):
            print(f"[Warn] Skipping non-file: {filepath}", file=sys.stderr)
            continue

        shared = process_file(filepath)
        if shared:  # Only output if at least one match (or require 2? see note below)
            output_lines.append(os.path.basename(filepath))
            output_lines.extend(shared)

    # Print result
    for line in output_lines:
        print(line)

if __name__ == "__main__":
    main()
