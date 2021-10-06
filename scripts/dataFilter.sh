#!/bin/bash

find . -name README  -type f -delete
find . -type f -size -5k -delete
find . -name *.unk -type f -delete
find . -type f -not -name "*.jpg" -not -name "*.png" -not -name "*.xml" -not -name "*.pdf" -not -name "*.doc" -not -name "*.csv" -not -name "*.html" -not -name "*.ppt" -not -name "*.xls" -not -name "*.gif" -not -name "*.txt" -not -name "*.ps" -not -name "*.gz" -not -name "*.gif" -delete

find . -type d -empty -delete
rm -rf _2/

for d in */; do
  if [[ $d =~ "_" ]]; then
    IFS='_'
    read -ra type <<<"$d"
    mkdir -p "${type[0]}"
    mv -f "${d}"* "${type[0]}"
  fi
done

find . -type d -empty -delete
