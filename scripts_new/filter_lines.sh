#!/bin/bash

# Check if both filenames are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <line_numbers_file> <data_file>"
    exit 1
fi

line_numbers_file="$1"
data_file="$2"

# Use awk to read line numbers from file and print corresponding lines
awk 'NR==FNR { lines[$1]; next } FNR in lines' "$line_numbers_file" "$data_file"