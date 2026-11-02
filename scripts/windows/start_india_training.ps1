param([Parameter(Mandatory = $true)][string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
$taskLogs = Join-Path $taskRoot 'logs/india-v1'
New-Item -ItemType Directory -Force -Path $taskLogs | Out-Null
$taskLock = Join-Path $taskLogs 'RUNNING.json'
if (Test-Path -LiteralPath $taskLock) {
    $taskOwner = Get-Content -LiteralPath $taskLock -Raw | ConvertFrom-Json
    if (Get-Process -Id $taskOwner.pid -ErrorAction SilentlyContinue) {
        Write-Output 'The India training orchestrator is already running.'
        exit 0
    }
}
$taskArguments = @('-u', '-m', 'anulm.training.india_run')
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'data/india-v1/DOWNLOADS.json'))) {
    $taskDownloadRecord = Join-Path $taskLogs 'download-process.json'
    $taskDownloadId = $null
    if (Test-Path -LiteralPath $taskDownloadRecord) {
        $taskPrior = Get-Content -LiteralPath $taskDownloadRecord -Raw | ConvertFrom-Json
        $taskExisting = Get-Process -Id $taskPrior.pid -ErrorAction SilentlyContinue
        if ($taskExisting -and $taskExisting.ProcessName -eq 'python') {
            $taskDownloadId = $taskExisting.Id
        }
    }
    if (-not $taskDownloadId) {
        $taskDownload = Start-Process -FilePath $taskPython -ArgumentList @('-u', '-m', 'anulm.data.india_corpus', 'fetch') `
            -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $taskLogs 'download.log') `
            -RedirectStandardError (Join-Path $taskLogs 'download-error.log')
        $taskDownloadId = $taskDownload.Id
        @{pid=$taskDownloadId;purpose='Pinned full India corpus download';started=(Get-Date).ToString('o')} |
            ConvertTo-Json | Set-Content -LiteralPath $taskDownloadRecord
    }
    $taskArguments += @('--download-pid', [string]$taskDownloadId)
}
$taskRun = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'orchestrator.log') `
    -RedirectStandardError (Join-Path $taskLogs 'orchestrator-error.log')
@{pid=$taskRun.Id;purpose='India download/prepare/tonight-training pipeline';started=(Get-Date).ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLogs 'orchestrator-process.json')
Write-Output "India pipeline launched. Status and logs: $taskLogs"
