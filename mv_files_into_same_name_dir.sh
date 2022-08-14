#!/bin/bash

for f in *.jpeg; do
  [[ -f "$f" ]] || continue                                                                                # skip if not regular file
  #~ echo "${f%.*}"                                                                                        # just so we can see progress
  mkdir "${f%.*}"                                                                                          # strip the file suffix and make directory of that name
  mv "$f" "${f%.*}"                                                                                        # move file into that directory
done
