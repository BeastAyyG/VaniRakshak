# VaniRakshak Android Build & Deploy Script
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=== 1. Building VaniRakshak Debug APK ===" -ForegroundColor Cyan
& ".\gradlew.bat" :app:assembleDebug

$apkPath = "$scriptDir\app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path $apkPath)) {
    Write-Error "APK not found at expected location: $apkPath"
    exit 1
}
Write-Host "APK Built Successfully: $apkPath" -ForegroundColor Green

Write-Host "`n=== 2. Checking ADB Handset Connection ===" -ForegroundColor Cyan
$adbPaths = @(
    "C:\Users\saumya\Downloads\platform-tools-latest-windows\platform-tools\adb.exe",
    "C:\Users\saumya\AppData\Local\Android\Sdk\platform-tools\adb.exe"
)
$adbPath = $adbPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $adbPath) {
    Write-Error "ADB executable not found. Ensure Android SDK platform-tools is installed."
    exit 1
}

function Check-AdbDevices {
    param([string]$path)
    $output = & $path devices -l
    return $output
}

$devicesOutput = Check-AdbDevices -path $adbPath
Write-Host ($devicesOutput -join "`n")

$hasDevice = $devicesOutput | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of devices" }
$isUnauthorized = $devicesOutput | Where-Object { $_ -match "\bunauthorized\b" }

# If device not visible, attempt auto-fix if elevated or guide user
if (-not $hasDevice) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if ($isAdmin) {
        Write-Host "`n[Administrator Detected] Automatically repairing OnePlus ADB interface GUIDs..." -ForegroundColor Yellow
        powershell -ExecutionPolicy Bypass -File "C:\Users\saumya\fix_adb_driver.ps1"
        Start-Sleep -Seconds 2
        $devicesOutput = Check-AdbDevices -path $adbPath
        $hasDevice = $devicesOutput | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of devices" }
        $isUnauthorized = $devicesOutput | Where-Object { $_ -match "\bunauthorized\b" }
    } else {
        # Check if OnePlus is plugged into USB
        $oneplusPnp = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -like "*VID_22D9*" }
        if ($oneplusPnp) {
            Write-Host "`n[Auto-Fix] Requesting Administrator permission to register OnePlus ADB driver..." -ForegroundColor Yellow
            try {
                $elevated = Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File C:\Users\saumya\fix_adb_driver.ps1 -NoPause" -Verb RunAs -PassThru -Wait
                Start-Sleep -Seconds 2
                $devicesOutput = Check-AdbDevices -path $adbPath
                $hasDevice = $devicesOutput | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of devices" }
                $isUnauthorized = $devicesOutput | Where-Object { $_ -match "\bunauthorized\b" }
            } catch {
                Write-Host "==========================================================================" -ForegroundColor Yellow
                Write-Host "  OnePlus Handset Detected on USB, but Windows ADB Driver Needs Registering" -ForegroundColor Yellow
                Write-Host "==========================================================================" -ForegroundColor Yellow
                Write-Host "To fix:"
                Write-Host "  1. Right-click 'Fix_OnePlus_ADB.bat' on your Desktop -> 'Run as administrator'"
                Write-Host "  OR"
                Write-Host "  2. Open PowerShell as Administrator and re-run .\build_and_install.ps1"
                Write-Host "==========================================================================" -ForegroundColor Yellow
            }
        }
    }
}

if ($isUnauthorized) {
    Write-Host "`n==========================================================================" -ForegroundColor Magenta
    Write-Host "  Phone is Connected but UNAUTHORIZED" -ForegroundColor Magenta
    Write-Host "==========================================================================" -ForegroundColor Magenta
    Write-Host "Please unlock your OnePlus phone screen."
    Write-Host "A popup titled 'Allow USB debugging?' should appear."
    Write-Host "Check 'Always allow from this computer' and tap 'Allow'."
    Write-Host "==========================================================================" -ForegroundColor Magenta
    exit 1
}

if (-not $hasDevice) {
    Write-Warning "Handset not ready in ADB. Ensure USB debugging is ON, driver is registered, and phone screen is unlocked."
    exit 1
}

Write-Host "`nHandset Connected!" -ForegroundColor Green

Write-Host "`n=== 3. Bridging Backend Port (adb reverse tcp:8000 tcp:8000) ===" -ForegroundColor Cyan
& $adbPath reverse tcp:8000 tcp:8000

Write-Host "`n=== 4. Installing Debug APK to Handset ===" -ForegroundColor Cyan
& $adbPath install -r $apkPath

Write-Host "`n=== 5. Launching VaniRakshak Application ===" -ForegroundColor Cyan
& $adbPath shell am start -n com.vericall.temp/com.vericall.SplashActivity

Write-Host "`n=== VaniRakshak is running on your phone! ===" -ForegroundColor Green
