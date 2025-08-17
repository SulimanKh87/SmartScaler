@echo off
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >NUL

REM Resolve project root (folder of this .bat)
set "BASE_DIR=%~dp0"
pushd "%BASE_DIR%"

REM Ensure a dedicated UI venv
if not exist "ui_venv\Scripts\python.exe" (
  py -3 -m venv ui_venv
)

REM Upgrade pip & install UI deps only
call "ui_venv\Scripts\python.exe" -m pip install --upgrade pip
call "ui_venv\Scripts\pip.exe" install -r requirementsUI.txt

REM Open the browser first, then start the server
start "" http://127.0.0.1:5502

REM Run the Flask app
set "FLASK_APP=app.py"
REM Force UTF-8 to avoid Windows cp1252 emoji issues
set "PYTHONUTF8=1"
call "ui_venv\Scripts\python.exe" app.py --host 127.0.0.1 --port 5502

popd
endlocal


REM ---- Prepare TensorFlow venv (tf_venv) for upscale/compare ----
if not exist "tf_venv\Scripts\python.exe" (
  echo Creating tf_venv (Python 3.9 recommended for TF 2.10)...
  py -3.9 -m venv tf_venv
)

call "tf_venv\Scripts\python.exe" -m pip install --upgrade pip
if exist "requirementsTF_VENV.txt" (
  call "tf_venv\Scripts\pip.exe" install -r requirementsTF_VENV.txt
) else (
  echo WARNING: requirementsTF_VENV.txt not found. Skipping TF deps install.
)

REM You may override the TF interpreter with SMARTSCALER_TF_PY or config.json
