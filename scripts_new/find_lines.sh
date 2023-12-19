# Check if filename and search string are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename> <search_string>"
    exit 1
fi

filename="$1"
search_string="$2"

# Use grep to find lines starting with the specified string and awk to print line numbers
line_numbers=$(grep -n "^$search_string" "$filename" | awk -F: '{print $1}')

# Check if any matching lines are found
if [ -z "$line_numbers" ]; then
    echo "No matching lines found."
else
    echo "$line_numbers"
fi