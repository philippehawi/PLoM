# C++ Shared Library: `cpp_lib`

This project provides a set of C++ shared libraries for computing potential gradients using either Eigen-based or native C++ implementations. The libraries can be used on both Windows and Linux and are tested using Python scripts.

---

## Folder Structure

```
cpp_lib/
├── compilation_scripts/      # Scripts for compiling the libraries
│   ├── compile_eigen.bat     # Windows batch script for Eigen-based library
│   ├── compile_eigen.sh      # Linux shell script for Eigen-based library
│   ├── compile_native.bat    # Windows batch script for native library
│   ├── compile_native.sh     # Linux shell script for native library
├── include/                  # Header files and external libraries
│   └── Eigen/                # Eigen library (header-only)
├── lib/                      # Pre-compiled libraries
│   ├── potential_eigen.dll   # Eigen-based library for Windows
│   ├── potential_eigen.so    # Eigen-based library for Linux
│   ├── potential_native.dll  # Native library for Windows
│   ├── potential_native.so   # Native library for Linux
├── src/                      # C++ source files
│   ├── potential_eigen.cpp   # Eigen-based implementation
│   ├── potential_native.cpp  # Native C++ implementation
├── test_scripts/             # Python scripts for testing the libraries
│   ├── check_potential_eigen.py # Test script for Eigen-based library
│   ├── check_potential_native.py # Test script for native library
```

---

## Libraries

### Eigen-Based Library
- **Path**: `lib/potential_eigen.*`
- **Description**: Implements the potential gradient computation using the Eigen C++ library for matrix operations.
- **Compilation Scripts**:
  - `compilation_scripts/compile_eigen.bat` (Windows)
  - `compilation_scripts/compile_eigen.sh` (Linux)

### Native Library
- **Path**: `lib/potential_native.*`
- **Description**: Implements the potential gradient computation using raw C++ arrays without external dependencies like Eigen.
- **Compilation Scripts**:
  - `compilation_scripts/compile_native.bat` (Windows)
  - `compilation_scripts/compile_native.sh` (Linux)

---

## Compilation Instructions

### Windows
1. Open a Command Prompt.
2. Navigate to the `compilation_scripts/` folder.
3. Run the appropriate batch file:
   - For Eigen-based library:
     ```cmd
     compile_eigen.bat
     ```
   - For native library:
     ```cmd
     compile_native.bat
     ```

### Linux
1. Open a terminal.
2. Navigate to the `compilation_scripts/` folder.
3. Run the appropriate shell script:
   - For Eigen-based library:
     ```bash
     ./compile_eigen.sh
     ```
   - For native library:
     ```bash
     ./compile_native.sh
     ```

---

## Testing Instructions

### Prerequisites
- Python 3.x
- Numpy installed:
  ```bash
  pip install numpy
  ```

### Run Tests
1. Navigate to the `test_scripts/` folder.
2. Run the appropriate test script:
   - For Eigen-based library:
     ```bash
     python check_potential_eigen.py
     ```
   - For native library:
     ```bash
     python check_potential_native.py
     ```

### Expected Output
- The script compares the results of the C++ library with a Python implementation.
- If the results match:
  ```
  Results match
  ```
- If there is a discrepancy:
  ```
  Results do not match
  ```

---

## Notes

- The **Eigen-based library** requires the Eigen library, located in the `include/Eigen/` folder.
- The **native library** does not depend on any external libraries, making it portable.
- Ensure that the Python scripts point to the correct library paths in `lib/`.

---

## License

This project is released under the MIT License. Feel free to use and modify it.