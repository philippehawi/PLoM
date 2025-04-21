@echo off

:: Define variables
set OUTPUT_LIB=../lib/potential_eigen.dll
set SOURCE_FILE=../src/potential_eigen.cpp
set INCLUDE_PATH=../include

:: Compilation command
g++ -shared -o %OUTPUT_LIB% %SOURCE_FILE% -static -static-libgcc -static-libstdc++ -I"%INCLUDE_PATH%" -O3 -march=native -flto

:: Check if compilation was successful
if %ERRORLEVEL% equ 0 (
    echo Compilation successful. Output: %OUTPUT_LIB%
) else (
    echo Compilation failed.
    exit /b 1
)

pause