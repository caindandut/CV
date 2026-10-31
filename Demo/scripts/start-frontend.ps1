$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Join-Path (Split-Path $PSScriptRoot -Parent) 'frontend')
& npm.cmd run dev
