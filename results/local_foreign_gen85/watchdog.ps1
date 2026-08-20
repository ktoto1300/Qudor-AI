$ErrorActionPreference = 'Continue'
$py = 'C:\Python314\python.exe'
$repo = 'C:\Users\kamil\Desktop\Qudor'
$out = Join-Path $repo 'results\local_foreign_gen85'
$log = Join-Path $out 'run.log'
$staleFile = Join-Path $out 'summary.json'
$wlog = Join-Path $out 'watchdog.log'
$target = 10060
$argsArena = @('tools\overnight_arena.py', '--net', 'checkpoints\gen85_best.pt',
    '--profile', 'cpu-laptop', '--only', 'berlioz,marcobt15,dimi,gorisanson,sigma',
    '--hours', '8', '--batch', '10', '--outdir', 'results\local_foreign_gen85')

function Log([string]$m) {
    Add-Content -LiteralPath $wlog -Value ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m)
}

Log "watchdog started; watching pid $target"
while ($true) {
    $alive = Get-Process -Id $target -ErrorAction SilentlyContinue
    if ($alive) {
        if (-not (Test-Path -LiteralPath $staleFile)) {
            Start-Sleep -Seconds 30
            continue
        }
        $stale = (Get-Date) - (Get-Item -LiteralPath $staleFile).LastWriteTime
        if ($stale.TotalMinutes -gt 15) {
            Log ("WEDGED: summary.json stale {0} min; killing {1}" -f [int]$stale.TotalMinutes, $target)
            Stop-Process -Id $target -Force -ErrorAction SilentlyContinue
        } else {
            Start-Sleep -Seconds 30
            continue
        }
    }
    Log 'arena not running; starting new instance'
    & $py @argsArena *>> $log
    Log ("arena instance exited code {0}; restarting in 20s" -f $LASTEXITCODE)
    Start-Sleep -Seconds 20
}