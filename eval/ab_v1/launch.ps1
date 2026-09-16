# ASCII only for Windows PowerShell 5.1.
param(
    [ValidateSet('all','check','smoke','run','export')][string]$Action = 'all',
    [string]$Distro = ''
)
$ErrorActionPreference = 'Stop'
try {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'WSL is missing. Use the same WSL2 Ubuntu environment used for training.'
    }
    $installed = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ })
    if ($Distro -eq '') {
        if ($installed -contains 'Ubuntu-24.04') { $Distro = 'Ubuntu-24.04' }
        elseif ($installed -contains 'Ubuntu') { $Distro = 'Ubuntu' }
        else { throw 'No Ubuntu found. Open the WSL distribution used for training and run bash eval/ab_v1/start.sh all.' }
    }
    if ($installed -notcontains $Distro) { throw "Distribution not found: $Distro" }
    $listing = (& wsl.exe --list --verbose 2>$null | Out-String) -replace "`0", ''
    if ($listing -notmatch ('(?m)^\s*\*?\s*' + [regex]::Escape($Distro) + '\s+\S+\s+2\s*$')) {
        throw "WSL2 required: $Distro"
    }
    $linuxRepo = ((& wsl.exe -d $Distro --exec wslpath -a $repo) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxRepo.StartsWith('/')) { throw 'Could not translate repository path.' }
    Write-Host "Repository: $repo"
    Write-Host 'Inference only: no training, package install, paid API, or weight upload.'
    & wsl.exe -d $Distro --cd $linuxRepo --exec bash eval/ab_v1/start.sh $Action
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'See START_EVAL_AB.txt. No training was started.'
    exit 1
}
