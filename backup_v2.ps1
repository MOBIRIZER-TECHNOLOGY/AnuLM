# Keep dated copies of the base-v2 resume point, one per ~3 hours, newest 3 kept.
# Registered as the scheduled task anulm_v2_backup, firing every 30 min; most
# firings do nothing. train.py already rewrites ckpt_base_v2.pt.last every
# 10,000 steps (~53 min); this guards against that one file being lost or bad.
#
# Timing matters on Windows: train.py saves with os.replace, which fails if
# another process has the target open -- so a copy that overlaps a save would
# crash training. Copies are made only 1-30 min after a save, when the next
# save is at least ~20 min away (a copy takes well under a minute).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$log = "v2_backup.log"
$src = "ckpt_base_v2.pt.last"
$dir = "backups\v2"
function Note($m) { Add-Content $log ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $m) }

if (-not (Test-Path $src)) { exit 0 }
$age = ((Get-Date) - (Get-Item $src).LastWriteTime).TotalMinutes
if ($age -lt 1 -or $age -gt 30) { exit 0 }                  # outside the safe window
New-Item -ItemType Directory -Force $dir | Out-Null
$newest = Get-ChildItem $dir -Filter "*.last" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($newest -and ((Get-Date) - $newest.LastWriteTime).TotalMinutes -lt 170) { exit 0 }   # one per ~3 h

$step = "unknown"
$eval = Select-String -Path "v2_train_run.log" -Pattern "eval @\s+(\d+)" | Select-Object -Last 1
if ($eval) { $step = [int]$eval.Matches[0].Groups[1].Value + 1 }
$name = "ckpt_base_v2_step{0}_{1}.pt.last" -f $step, (Get-Date -Format "yyyyMMdd-HHmm")
try {
    Copy-Item $src (Join-Path $dir $name)
    if (Test-Path "ckpt_base_v2.pt") { Copy-Item "ckpt_base_v2.pt" (Join-Path $dir ($name -replace "\.pt\.last$", ".best.pt")) }
    Note "backed up step $step -> $name"
} catch { Note "backup failed: $($_.Exception.Message)"; exit 0 }

# Milestones for the "watch it learn" page: the first backup at or past each
# of these steps also keeps its best checkpoint (1.6 GB) in milestones\,
# which is never pruned -- the rolling copies below are.
$milestones = 700000, 1000000, 1500000, 2000000, 2200000, 2500000, 2750000
if ($step -ne "unknown") {
    $mdir = Join-Path $dir "milestones"
    New-Item -ItemType Directory -Force $mdir | Out-Null
    foreach ($m in $milestones) {
        $target = Join-Path $mdir ("ckpt_base_v2_milestone{0}.pt" -f $m)
        if ($step -ge $m -and -not (Test-Path $target) -and (Test-Path "ckpt_base_v2.pt")) {
            Copy-Item "ckpt_base_v2.pt" $target
            Note "kept milestone $m (best checkpoint as of step $step)"
            break
        }
    }
}

# Keep the newest 3 backup sets; older ones are removed (these are copies, the
# live resume point is untouched).
$sets = Get-ChildItem $dir -Filter "*.last" | Sort-Object LastWriteTime -Descending
foreach ($old in ($sets | Select-Object -Skip 3)) {
    Remove-Item $old.FullName
    $best = $old.FullName -replace "\.pt\.last$", ".best.pt"
    if (Test-Path $best) { Remove-Item $best }
    Note "removed old backup $($old.Name)"
}
