@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =========================================================
rem  SmartScaler - Prepare Environments (Windows 10)
rem  - Creates venvs if missing: ui_venv, tf_venv, venv310
rem  - Robustly bootstraps PIP in each venv (fixes "No module named pip.__main__")
rem  - Installs ALL requirements in each venv
rem  - Exposes SMARTSCALER_*_PY env vars
rem  - No user input needed
rem =========================================================

title SmartScaler - Prepare Environments
set "ROOT=%~dp0"
pushd "%ROOT%"

echo ================================================================
echo   SmartScaler - Prepare Python Environments
echo ================================================================
echo.

rem ----- Prefer Python 3.10, then 3.11, 3.9, else "python" on PATH -----
call :find_python
if errorlevel 1 (
  echo [ERROR] Could not find Python 3.9+ via "py" or "python".
  echo         Please install Python 3.10 and the Windows "py" launcher.
  goto :done_fail
)
echo Using Python launcher: %PY%
echo.

rem ----- VENV paths -----
set "UI_VENV=%ROOT%ui_venv"
set "TF_VENV=%ROOT%tf_venv"
set "VENV310=%ROOT%venv310"

rem ----- Requirements files (must be at project root) -----
set "UI_REQ=%ROOT%requirementsUI.txt"
set "TF_REQ=%ROOT%requirementsTF_VENV.txt"
set "V310_REQ=%ROOT%requirementsVENV310.txt"

rem ===============================
rem [1/3] Ensure virtual envs exist
rem ===============================
echo [1/3] Ensuring virtual environments exist...
call :ensure_venv "%UI_VENV%"  || goto :done_fail
call :ensure_venv "%TF_VENV%"  || goto :done_fail
call :ensure_venv "%VENV310%"  || goto :done_fail
echo.

rem ==============================================
rem [2/3] Install ALL requirements up-front
rem ==============================================
echo [2/3] Installing ALL requirements (UI, TF_VENV, VENV310)...
call :pip_install "%UI_VENV%"  "%UI_REQ%"   "UI"      || goto :done_fail
call :pip_install "%TF_VENV%"  "%TF_REQ%"   "TF_VENV" || goto :done_fail
call :pip_install "%VENV310%"  "%V310_REQ%" "VENV310" || goto :done_fail
echo.

rem ===================================================
rem [3/3] Expose helper env vars for UI/subprocesses
rem ===================================================
set "SMARTSCALER_ROOT=%ROOT%"
set "SMARTSCALER_UI_PY=%UI_VENV%\Scripts\python.exe"
set "SMARTSCALER_TF_PY=%TF_VENV%\Scripts\python.exe"
set "SMARTSCALER_V310_PY=%VENV310%\Scripts\python.exe"

rem Optional: make UI venv the default for any follow-up commands
set "VIRTUAL_ENV=%UI_VENV%"
set "PATH=%UI_VENV%\Scripts;%PATH%"

echo [INFO] Default venv = UI
echo        SMARTSCALER_UI_PY   = %SMARTSCALER_UI_PY%
echo        SMARTSCALER_TF_PY   = %SMARTSCALER_TF_PY%
echo        SMARTSCALER_V310_PY = %SMARTSCALER_V310_PY%
echo.
echo [OK] Environments ready. You can now run your UI or scripts.
goto :done_ok


rem =========================
rem Subroutines
rem =========================

:find_python
  rem Prefer py -3.10, then -3.11, -3.9, else plain python
  set "PY=py -3.10"
  %PY% -V >NUL 2>&1 && exit /b 0

  set "PY=py -3.11"
  %PY% -V >NUL 2>&1 && exit /b 0

  set "PY=py -3.9"
  %PY% -V >NUL 2>&1 && exit /b 0

  set "PY=python"
  %PY% -V >NUL 2>&1 && exit /b 0

  exit /b 1


:ensure_venv
  set "VENV_DIR=%~1"
  if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   - Found venv: %VENV_DIR%
    exit /b 0
  )
  echo   - Creating venv: %VENV_DIR%
  %PY% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo   ! Failed to create venv: %VENV_DIR%
    exit /b 1
  )
  exit /b 0


:bootstrap_pip
  rem Ensures pip works inside the given venv by using ensurepip,
  rem and if needed, repairs with "python -m venv --upgrade-deps".
  set "VENV_DIR=%~1"
  set "TAG=%~2"

  rem 1) quick check
  call "%VENV_DIR%\Scripts\python.exe" -m pip --version >NUL 2>&1
  if not errorlevel 1 exit /b 0

  echo   [%TAG%] Bootstrapping pip via ensurepip...
  call "%VENV_DIR%\Scripts\python.exe" -m ensurepip --upgrade
  if errorlevel 1 (
    echo   [%TAG%] WARN: ensurepip failed or unavailable. Trying venv repair...
  )

  rem 2) re-check
  call "%VENV_DIR%\Scripts\python.exe" -m pip --version >NUL 2>&1
  if not errorlevel 1 exit /b 0

  rem 3) attempt repair with --upgrade-deps (Python 3.9+)
  echo   [%TAG%] Repairing venv tools with: %PY% -m venv --upgrade-deps "%VENV_DIR%"
  %PY% -m venv --upgrade-deps "%VENV_DIR%" >NUL 2>&1

  rem final check
  call "%VENV_DIR%\Scripts\python.exe" -m pip --version >NUL 2>&1
  if errorlevel 1 (
    echo   [%TAG%] ERROR: pip is still not available after repair.
    exit /b 1
  )
  exit /b 0


:pip_install
  set "VENV_DIR=%~1"
  set "REQ_FILE=%~2"
  set "TAG=%~3"

  if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [%TAG%] ERROR: Missing venv at %VENV_DIR%
    exit /b 1
  )

  rem Ensure pip is healthy first
  call :bootstrap_pip "%VENV_DIR%" "%TAG%"
  if errorlevel 1 exit /b 1

  echo   [%TAG%] Upgrading pip/setuptools/wheel...
  call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
  if errorlevel 1 (
    echo   [%TAG%] ERROR: Failed to upgrade pip/setuptools/wheel.
    exit /b 1
  )

  if exist "%REQ_FILE%" (
    echo   [%TAG%] Installing requirements from: %REQ_FILE%
    call "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
      echo   [%TAG%] ERROR: Failed to install requirements from %REQ_FILE%
      exit /b 1
    )
  ) else (
    echo   [%TAG%] WARN: Requirements file not found: %REQ_FILE% (skipping)
  )
  exit /b 0


rem =========================
rem End / Exit points
rem =========================
:done_ok
echo.
echo [DONE] SmartScaler environment setup completed successfully.
echo.
popd
endlocal
exit /b 0

:done_fail
echo.
echo ------------------------------------------------
echo   One or more steps failed. Please review logs.
echo ------------------------------------------------
echo.
popd
endlocal
exit /b 1
