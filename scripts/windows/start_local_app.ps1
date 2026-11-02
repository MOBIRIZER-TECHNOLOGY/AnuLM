param(
    [ValidateSet('demo', 'evaluation')][string]$App = 'demo',
    [string]$PythonExecutable,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $PythonExecutable) {
    $taskCandidates = @((Join-Path $taskRoot '.venv/Scripts/python.exe'),
                        (Join-Path $taskRoot '../validation-env/Scripts/python.exe'))
    $PythonExecutable = $taskCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $PythonExecutable) { throw 'Run the setup in START_TRAINING.md first, or supply -PythonExecutable.' }
$taskPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
$taskModule = if ($App -eq 'demo') { 'anulm.apps.omni' } else { 'anulm.apps.evaluate' }
$taskPort = if ($App -eq 'demo') { 7863 } else { 7864 }
$taskUrl = "http://127.0.0.1:$taskPort"
$taskLogs = Join-Path $taskRoot 'logs/local-apps'
New-Item -ItemType Directory -Force -Path $taskLogs | Out-Null
$taskRecordPath = Join-Path $taskLogs "$App.json"
$taskProcess = $null
if (Test-Path -LiteralPath $taskRecordPath) {
    $taskRecord = Get-Content -LiteralPath $taskRecordPath -Raw | ConvertFrom-Json
    $taskOwner = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskRecord.pid)" -ErrorAction SilentlyContinue
    if ($taskOwner -and $taskOwner.CommandLine -like "* -m $taskModule *") {
        $taskProcess = Get-Process -Id $taskRecord.pid
    }
}
if (-not $taskProcess) {
    if (Get-NetTCPConnection -LocalPort $taskPort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $taskPort is already occupied. No existing process was stopped."
    }
    $env:GRADIO_ANALYTICS_ENABLED = 'False'
    $env:PYTHONIOENCODING = 'utf-8'
    Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue
    $taskArguments = @('-u', '-m', $taskModule, '--host', '127.0.0.1', '--port', [string]$taskPort)
    if ($App -eq 'evaluation') {
        if (-not $Checkpoint) {
            $taskFinal = Join-Path $taskRoot 'runs/india-v1/final-step-60000.pt'
            if (Test-Path -LiteralPath $taskFinal) { $Checkpoint = $taskFinal }
        }
        if ($Checkpoint) { $taskArguments += @('--ckpt', ('"{0}"' -f (Resolve-Path -LiteralPath $Checkpoint).Path)) }
    } else {
        $taskManifest = Join-Path $taskRoot 'runs/india-v1/post-training/approved-demo-manifest.json'
        if (Test-Path -LiteralPath $taskManifest) { $taskArguments += @('--validation-manifest', ('"{0}"' -f $taskManifest)) }
    }
    $taskProcess = Start-Process -FilePath $taskPython `
        -ArgumentList $taskArguments `
        -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $taskLogs "$App.log") `
        -RedirectStandardError (Join-Path $taskLogs "$App-error.log")
    @{pid=$taskProcess.Id; module=$taskModule; url=$taskUrl; started=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $taskRecordPath
}
for ($taskAttempt = 0; $taskAttempt -lt 40; $taskAttempt++) {
    try {
        $taskResponse = Invoke-WebRequest -Uri "$taskUrl/config" -UseBasicParsing -TimeoutSec 2
        if ($taskResponse.StatusCode -eq 200) {
            Write-Output "$App is ready: $taskUrl"
            exit 0
        }
    } catch {
        if (-not (Get-Process -Id $taskProcess.Id -ErrorAction SilentlyContinue)) {
            throw "App exited. See $taskLogs/$App-error.log"
        }
    }
    Start-Sleep -Milliseconds 250
}
throw "App is still starting. Check $taskLogs/$App-error.log and open $taskUrl when ready."
