#!/bin/bash
# Show top contributors by current line count in the codebase

echo
echo "Top Contributors by Current Line Count"
echo "======================================="
echo

git ls-files | grep -E '\.(py|swift|js|ts|tsx|md)$' | xargs -I {} git blame --line-porcelain {} 2>/dev/null | grep "^author " | sort | uniq -c | sort -rn | awk '
BEGIN {total=0}
{count[NR]=$1; author[NR]=$2" "$3" "$4" "$5; total+=$1}
END {
  for(i=1; i<=NR; i++) {
    pct = (count[i]/total)*100
    printf "%6d lines  %5.1f%%  %s\n", count[i], pct, author[i]
  }
}' | head -15
