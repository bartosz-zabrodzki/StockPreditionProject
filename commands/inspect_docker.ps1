param(
    [string]$ContainerName = "stock_predictor_debug",
    [string]$ImageName = "stock-predictor-full"
)

Write-Host "`n[1] Starting temporary Docker container..." -ForegroundColor Cyan
docker run -dit --name $ContainerName -v "${PWD}:/app" $ImageName bash

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start container. Ensure Docker Desktop is running and image exists." -ForegroundColor Red
    exit 1
}

Write-Host "`n[2] Working directory inside container:" -ForegroundColor Yellow
docker exec -it $ContainerName bash -c "pwd"

Write-Host "`n[3] Project structure inside container (/app):" -ForegroundColor Yellow
docker exec -it $ContainerName bash -c "ls -R /app"

Write-Host "`n[4] Folder size summary:" -ForegroundColor Yellow
docker exec -it $ContainerName bash -c "du -h --max-depth=1 /app"

Write-Host "`n[5] Opening interactive shell (type 'exit' to quit)..." -ForegroundColor Cyan
docker exec -it $ContainerName bash

Write-Host "`n[6] Cleaning up container..." -ForegroundColor Cyan
docker rm -f $ContainerName | Out-Null

Write-Host "`n[✔] Inspection complete — container removed." -ForegroundColor Green