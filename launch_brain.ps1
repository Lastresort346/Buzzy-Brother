$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m flybrain.arena --brain @args
if ($LASTEXITCODE -ne 0) { throw "Neural arena exited with code $LASTEXITCODE" }
