# schedule_task.ps1
# Registers a Windows Task Scheduler task to run the Ocean Debris daily pipeline
# at 02:00 AM every day.
#
# Usage (run as Administrator):
#   powershell -ExecutionPolicy Bypass -File .\schedule_task.ps1
#
# To remove the task later:
#   Unregister-ScheduledTask -TaskName "OceanDebrisDailyUpdate" -Confirm:$false

$TaskName    = "OceanDebrisDailyUpdate"
$Description = "Ocean Debris Detector: daily data refresh + 90-day trajectory simulation"

# Adjust these paths to match your environment
$PythonExe   = (Get-Command python).Source
$BackendDir  = Join-Path $PSScriptRoot "backend"
$ScriptArg   = "-m src.scheduler.daily_pipeline --max-particles 5000"
$LogFile     = Join-Path $BackendDir "logs\daily_pipeline.log"

# Ensure log directory exists
$LogDir = Split-Path $LogFile
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $ScriptArg `
    -WorkingDirectory $BackendDir

$Trigger = New-ScheduledTaskTrigger -Daily -At "02:00AM"

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30)

# Run as current user (no password required for interactive sessions)
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest

try {
    # Remove if already registered
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

    Register-ScheduledTask `
        -TaskName    $TaskName `
        -Description $Description `
        -Action      $Action `
        -Trigger     $Trigger `
        -Settings    $Settings `
        -Principal   $Principal `
        -Force | Out-Null

    Write-Host "✓ Task '$TaskName' registered successfully." -ForegroundColor Green
    Write-Host "  Runs: daily at 02:00 AM"
    Write-Host "  Python: $PythonExe"
    Write-Host "  Working dir: $BackendDir"
    Write-Host ""
    Write-Host "  To run immediately:  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  To view status:      Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  To remove:           Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "✗ Failed to register task: $_" -ForegroundColor Red
    Write-Host "  Try running this script as Administrator." -ForegroundColor Yellow
    exit 1
}
