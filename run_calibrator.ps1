$ErrorActionPreference = "Stop"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$bundledPython = "C:\Users\belad\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { $bundledPython }
& $python "$PSScriptRoot\calibrator_gui.py"
