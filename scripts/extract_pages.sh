#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# The return value of a pipeline is the status of the last command to exit non-zero.
set -euo pipefail

function usage {
  echo "Usage: $0 <input> <output_dir> <dpi>"
  echo ""
  echo "Arguments:"
  echo "  input.pdf    - Path to the source PDF file."
  echo "  output_dir   - Path to the directory where PNG files will be saved."
  echo "  dpi          - Resolution in dpi (e.g., 150, 300)."
  exit 1
}

# --- 1. Input Validation ---

# Check for the correct number of arguments
if [[ $# -le 1 ]]; then
  echo "Error: Incorrect number of arguments." >&2
  usage
fi

INPUT_PDF="$1"
OUTPUT_DIR="$2"
DPI="${3:-150}"
TEMP_PREFIX="p"

if [[ ! -f "$INPUT_PDF" ]]; then
  echo "Error: Input file '$INPUT_PDF' not found or is not a regular file." >&2
  exit 2
fi

if ! [[ "$DPI" =~ ^[0-9]+$ ]] || (( DPI <= 0 )); then
  echo "Error: DPI must be a positive integer, but got '$DPI'." >&2
  exit 3
fi

if ! pdftoppm -png -r "$DPI" "$INPUT_PDF" "$OUTPUT_DIR/$TEMP_PREFIX"; then
  echo "Error: pdftoppm failed to process the PDF." >&2
  exit 5
fi

exit 0