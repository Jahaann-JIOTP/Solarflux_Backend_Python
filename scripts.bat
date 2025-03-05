@echo off
setlocal enabledelayedexpansion

REM Define the Python executable path (optional, if not set in PATH)
set PYTHON_EXEC=python

REM List of Python scripts to run in sequence
set SCRIPTS=(
    StringHour.py
    StringDay.py
    General_Tags_five.py
    GT_Hour.py
    Day_plant.py
    gmmeterhour.py
    gmmeterday.py
    merge.py
)

REM Iterate over the scripts
for %%S in %SCRIPTS% do (
    echo Running %%S...
    %PYTHON_EXEC% %%S
    if errorlevel 1 (
        echo %%S failed. Exiting batch file.
        exit /b 1
    )
    echo %%S completed successfully.
)

echo All scripts executed successfully.
exit /b 0
