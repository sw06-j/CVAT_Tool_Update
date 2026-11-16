$ErrorActionPreference = "SilentlyContinue"
$cvatDir = "e:\작업\열화상 데이터 라벨링\OSS_CVAT"
$dockerExe = "C:\Users\Jeongsiwoo\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"

# 1. Start Docker Desktop if daemon is not ready
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $dockerExe) {
        Start-Process $dockerExe
    }
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 2
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
}

# 2. Start CVAT Docker containers
Set-Location $cvatDir
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.override.yml up -d

# 3. Start auto_trainer.py if port 5000 is not already listening
$p5000 = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if (-not $p5000) {
    Start-Process -FilePath "python" -ArgumentList "auto_trainer.py" -WorkingDirectory $cvatDir -WindowStyle Minimized -RedirectStandardOutput "$cvatDir\auto_trainer_out.log" -RedirectStandardError "$cvatDir\auto_trainer_err.log"
}

# 4. Wait up to 40s for both CVAT (8080) and AI Engine (5000) to respond 200
$c8080 = "pending"
$c5000 = "pending"
for ($i = 0; $i -lt 20; $i++) {
    try { $c8080 = (Invoke-WebRequest -Uri "http://localhost:8080/api/server/about" -UseBasicParsing -TimeoutSec 3).StatusCode } catch { $c8080 = $_.Exception.Message }
    try { $c5000 = (Invoke-WebRequest -Uri "http://localhost:5000/status" -UseBasicParsing -TimeoutSec 3).StatusCode } catch { $c5000 = $_.Exception.Message }
    if ($c8080 -eq 200 -and $c5000 -eq 200) { break }
    Start-Sleep -Seconds 2
}
Write-Output "CVAT(8080): $c8080 | AI_Engine(5000): $c5000"
if ($c5000 -ne 200 -and (Test-Path "$cvatDir\auto_trainer_err.log")) {
    Get-Content "$cvatDir\auto_trainer_err.log" -Tail 15
}
