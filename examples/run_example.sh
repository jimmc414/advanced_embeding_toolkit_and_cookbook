#!/bin/bash
# Helper script to run examples with correct PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the example
if [ $# -eq 0 ]; then
    echo "Usage: $0 <example_number or filename>"
    echo "Examples:"
    echo "  $0 01"
    echo "  $0 01_basic_search.py"
    echo "  $0 all   # Run all examples"
    exit 1
fi

if [ "$1" == "all" ]; then
    # Run all examples
    for example in "$SCRIPT_DIR"/*.py; do
        echo ""
        echo "========================================"
        echo "Running: $(basename "$example")"
        echo "========================================"
        python3 "$example" || echo "Failed: $example"
    done
else
    # Run specific example
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        # Number provided, find matching file
        example_file=$(ls "$SCRIPT_DIR"/"$1"_*.py 2>/dev/null | head -1)
    else
        # Filename provided
        example_file="$SCRIPT_DIR/$1"
    fi

    if [ -f "$example_file" ]; then
        python3 "$example_file"
    else
        echo "Error: Example not found: $1"
        echo "Available examples:"
        ls -1 "$SCRIPT_DIR"/*.py | sed 's/.*\//  /'
        exit 1
    fi
fi
