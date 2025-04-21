#!/bin/bash

# Variables
OUTPUT_LIB="../lib/potential_eigen.so"         # Output library name
SOURCE_FILE="../src/potential_eigen.cpp"          # Source C++ file
EIGEN_INCLUDE_PATH="../include" # Path to Eigen include directory

# Compilation command
g++ -shared -o $OUTPUT_LIB $SOURCE_FILE \
    -fPIC \
    -static-libgcc -static-libstdc++ \
    -O3 -march=native \
    -I"$EIGEN_INCLUDE_PATH"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Output: $OUTPUT_LIB"
else
    echo "Compilation failed."
    exit 1
fi
