#!/bin/bash

# Variables
OUTPUT_LIB="../lib/potential_native.so"         # Output library name
SOURCE_FILE="../src/potential_native.cpp"          # Source C++ file

# Compilation command
g++ -shared -o $OUTPUT_LIB $SOURCE_FILE \
    -fPIC \
    -static-libgcc -static-libstdc++ -lm \
    -O3 -march=native \

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Output: $OUTPUT_LIB"
else
    echo "Compilation failed."
    exit 1
fi
