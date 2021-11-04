#!/bin/bash

find . -type f -print0 | xargs -0 -n 1 -P 8 python ../../../scripts/preprocessing.py
