$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m flybrain.arena @args
if ($LASTEXITCODE -ne 0) { throw "Arena exited with code $LASTEXITCODE" }
