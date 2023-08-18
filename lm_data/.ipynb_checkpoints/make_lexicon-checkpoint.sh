#!/bin/bash

filename="$1"
outputfile="$2"

# Use tr command to replace all non-alphanumeric characters with newline
# Use sort command to sort the words
# Use uniq command to remove duplicates
# Use tr command again to replace newline with space
# Use tr command one more time to replace space with newline
tr -c '[:alnum:]' '\n' < "$filename" | sort | uniq | tr ' ' '\n' > "$outputfile"
