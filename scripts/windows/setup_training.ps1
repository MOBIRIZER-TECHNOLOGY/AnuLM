param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped: $Executable exited with $LASTEXITCODE" }
}
Invoke-Checked -Executable $PythonExecutable -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3,14), "Use Python 3.14 for this tested environment"')
$taskVenv = Join-Path $taskRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $taskVenv 'Scripts/python.exe'))) {
    Invoke-Checked -Executable $PythonExecutable -Arguments @('-m', 'venv', $taskVenv)
}
$taskPython = Join-Path $taskVenv 'Scripts/python.exe'
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'pip', 'install', 'torch==2.14.0', 'torchvision==0.29.0', '--index-url', 'https://download.pytorch.org/whl/cu130')
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'pip', 'install', '-r', 'scripts/requirements-india.txt')
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'pip', 'install', '--no-deps', '-e', '.')
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'pip', 'check')
Invoke-Checked -Executable $taskPython -Arguments @('-m', 'playwright', 'install', 'chromium')
Invoke-Checked -Executable $taskPython -Arguments @('scripts/check_training_machine.py')
Write-Output 'Setup passed. Read START_TRAINING.md before launching the download/training pipeline.'
