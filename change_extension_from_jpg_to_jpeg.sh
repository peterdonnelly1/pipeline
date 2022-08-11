#!/bin/bash

for f in *.jpg; do
  [[ -f "$f" ]] || continue                                                                                # skip if not regular file
    mv -- "$f" "${f%.jpg}.jpeg" 
done

