# ===============================================
# PowerShell Diagnostic Script for FastAPI Pipeline
# ===============================================
# Purpose: run pipeline, detect 500, fetch traceback from Docker logs
# Usage:   .\\test_pipeline.ps1 -Ticker META
# ===============================================

param (
    [string]$Ticker = "META",
    [string]$Container = "stockpredictionproject-api-1"
)

Write-Host "`n=== Running pipeline for $Ticker ===" -ForegroundColor Cyan

# --- 1. Send request ---
$body = @{ ticker = $Ticker; force = $true } | ConvertTo-Json
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:8000/pipeline" -Method POST -Body $body -ContentType "application/json" -ErrorAction Stop
    Write-Host "✅ Pipeline completed successfully!" -ForegroundColor Green
    Write-Host ($resp | ConvertTo-Json -Depth 4)
    exit 0
}
catch {
    Write-Host "⚠️  Pipeline failed. Checking container logs..." -ForegroundColor Yellow
}

# --- 2. Try to pull container logs ---
try {
    $logs = docker logs $Container --tail 200 2>&1
    $errorLines = $logs | Select-String -Pattern "\[run_predict\] Failed"
    if ($errorLines) {
        Write-Host "`n--- Recent run_predict errors from Docker ---" -ForegroundColor Red
        $errorLines.Context.PreContext + $errorLines.Line + $errorLines.Context.PostContext
    }
    else {
        Write-Host "No [run_predict] errors found in Docker logs. Showing last lines:" -ForegroundColor Yellow
        $logs | Select-String -Pattern "ERROR|Traceback|Exception" -Context 3
    }
}
catch {
    Write-Host "❌ Unable to access Docker logs. Make sure container '$Container' is running." -ForegroundColor Red
}

# --- 3. Try to read log file from container ---
try {
    Write-Host "`n--- Checking for predict_error.log file ---" -ForegroundColor Cyan
    $filePath = "/output/${Ticker}_predict_error.log"
    $cmd = "cat $filePath"
    $output = docker exec $Container bash -c $cmd 2>&1
    if ($output -match "Traceback") {
        Write-Host $output -ForegroundColor White
    }
    else {
        Write-Host "No traceback file found at $filePath" -ForegroundColor DarkYellow
    }
}
catch {
    Write-Host "❌ Failed to fetch error log from container." -ForegroundColor Red
}
