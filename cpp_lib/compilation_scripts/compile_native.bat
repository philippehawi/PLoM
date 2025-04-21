@echo off

:: Define variables
set OUTPUT_LIB=../lib/potential_native.dll
set SOURCE_FILE=../src/potential_native.cpp

:: Compilation command
g++ -shared -o %OUTPUT_LIB% %SOURCE_FILE% -static -static-libgcc -static-libstdc++ -lm -O3 -march=native -flto

:: Check if compilation was successful
if %ERRORLEVEL% equ 0 (
    echo Compilation successful. Output: %OUTPUT_LIB%
) else (
    echo Compilation failed.
    exit /b 1
)

pause