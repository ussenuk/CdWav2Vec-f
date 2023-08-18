#!/bin/bash

# specify the input file
input_file="ling_sent.txt"

# specify the output file
output_file="output.txt"

# read the input file line by line
while read -r line; do
    # use sed to remove any leading whitespace
    echo "$line" | sed 's/^[ \t]*//' >> "$output_file"
done < "$input_file"