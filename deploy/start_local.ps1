<#
.SYNOPSIS
  Arranca la webapp ECG en esta PC (Windows 11) con waitress y el bundle versionado.

.DESCRIPTION
  Por defecto escucha solo en 127.0.0.1 (nadie mas puede conectarse) y sirve en CPU:
  la inferencia tarda 43-93 ms por modelo en CPU y 5,5 ms en GPU, pero la peticion completa tarda ~2,4 s
  casi toda en el PDF y los graficos, asi que la GPU no acelera nada y queda libre para entrenar.

  -Port     Puerto (5002 por defecto).
  -Threads  Hilos de waitress (4 por defecto).
  -Lan      Escucha en toda la red local (0.0.0.0): telefonos u otras PCs de la misma red.
  -Tunnel   Ademas abre un tunel rapido de Cloudflare: URL https publica y temporal, sin cuenta.
            Configura waitress para confiar en el tunel (cuenta un proxy y lee X-Forwarded-For) y obliga
            a escuchar solo en 127.0.0.1; asi cada visitante tiene su propio limite de solicitudes.
  -UseGpu   Infiere en la GPU si hay (ECG_DEVICE=auto).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\start_local.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\start_local.ps1 -Tunnel
#>
[CmdletBinding()]
param(
    [int]$Port = 5002,
    [int]$Threads = 4,
    [switch]$Lan,
    [switch]$Tunnel,
    [switch]$UseGpu
)

$ErrorActionPreference = "Stop"
if ($Lan -and $Tunnel) {
    throw "-Lan y -Tunnel son incompatibles: con tunel la app debe escuchar solo en 127.0.0.1."
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$cloudflared = $null
if ($Tunnel) {
    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared) {
        throw "No se encontro cloudflared. Instalelo con: winget install --id Cloudflare.cloudflared  (y abra una terminal nueva)."
    }
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "No se encontro Python en el PATH. Instale Python 3.10 o superior y vuelva a intentarlo."
}
$ErrorActionPreference = "Continue"
& $python.Source -c "import waitress, torch, flask" 2>&1 | Out-Null
$depsOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $depsOk) {
    throw "Faltan dependencias. Ejecute: pip install -r requirements.txt"
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "El puerto $Port ya esta en uso. Use -Port con otro numero (o cierre la otra instancia)."
}

$bindHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
if ($UseGpu) { $env:ECG_DEVICE = "auto" } elseif (-not $env:ECG_DEVICE) { $env:ECG_DEVICE = "cpu" }
# waitress guarda el cuerpo de la peticion antes de que Flask compruebe su tamano; su tope por defecto es 1 GiB.
# Se limita al mismo valor que ECG_MAX_UPLOAD_MB (5 por defecto) mas 1 MB de holgura para las cabeceras multipart.
$maxUploadMb = 5
if ($env:ECG_MAX_UPLOAD_MB -match '^[0-9]+$' -and [int]$env:ECG_MAX_UPLOAD_MB -ge 1) { $maxUploadMb = [int]$env:ECG_MAX_UPLOAD_MB }
$maxBody = ($maxUploadMb + 1) * 1MB
$serveArgs = @("-m", "waitress", "--host=$bindHost", "--port=$Port", "--threads=$Threads", "--max-request-body-size=$maxBody")
if ($Tunnel) {
    # waitress borra X-Forwarded-For salvo que confie en el proxy: sin esto todos los visitantes del tunel
    # compartirian una sola IP (127.0.0.1) y un visitante agotaria el limite de todos.
    $serveArgs += @("--trusted-proxy=127.0.0.1", "--trusted-proxy-count=1", "--trusted-proxy-headers=x-forwarded-for")
}
$serveArgs += "webapp.wsgi:app"                    # waitress exige la aplicacion como ultimo argumento

Write-Host "ECG webapp: http://127.0.0.1:$Port   (dispositivo: $($env:ECG_DEVICE), escucha en $bindHost)"
if ($Lan) {
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
        ForEach-Object { Write-Host "En la red local: http://$($_.IPAddress):$Port" }
    Write-Host "Windows puede pedir permiso en el Firewall la primera vez: permita solo redes privadas."
}

if (-not $Tunnel) {
    & $python.Source @serveArgs
    exit $LASTEXITCODE
}

$server = Start-Process -FilePath $python.Source -ArgumentList $serveArgs -PassThru -NoNewWindow
try {
    $ready = $false
    for ($i = 0; $i -lt 180 -and -not $ready; $i++) {
        if ($server.HasExited) { throw "waitress termino antes de tiempo (codigo $($server.ExitCode))." }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null
            $ready = $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) { throw "La app no respondio en 180 s." }
    Write-Host "Abriendo el tunel. La URL https://....trycloudflare.com aparece abajo y cambia en cada ejecucion."
    Write-Host "Cualquiera con esa URL puede usar la demo mientras este script siga abierto. Ctrl+C para cerrar."
    & $cloudflared.Source tunnel --url "http://127.0.0.1:$Port"
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
