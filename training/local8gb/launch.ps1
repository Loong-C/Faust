# ASCII-only script for Windows PowerShell 5.1; UTF-8 data is read by Linux Python.
param(
    [ValidateSet('all','check','doctor','prepare','smoke','train','generate')][string]$Action = 'all',
    [ValidateSet('standard','lean')][string]$Profile = 'standard',
    [string]$Distro = ''
)
$ErrorActionPreference = 'Stop'
try {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'WSL is missing. In Administrator PowerShell: wsl --install -d Ubuntu-24.04 ; then reboot and create a Linux username/password.'
    }
    $installed = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ })
    if ($Distro -eq '') {
        if ($installed -contains 'Ubuntu-24.04') { $Distro = 'Ubuntu-24.04' }
        elseif ($installed -contains 'Ubuntu') { $Distro = 'Ubuntu' }
        else { throw 'No Ubuntu found. In Administrator PowerShell: wsl --install -d Ubuntu-24.04 ; reboot if requested, open Ubuntu and create your user.' }
    }
    if ($installed -notcontains $Distro) { throw "Distribution not found: $Distro" }
    $listing = (& wsl.exe --list --verbose 2>$null | Out-String) -replace "`0", ''
    if ($listing -notmatch ('(?m)^\s*\*?\s*' + [regex]::Escape($Distro) + '\s+\S+\s+2\s*$')) {
        throw "WSL2 required. Run: wsl --set-version $Distro 2"
    }
    $linuxRepo = ((& wsl.exe -d $Distro --exec wslpath -a $repo) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxRepo.StartsWith('/')) { throw 'Could not translate repository path.' }
    Write-Host "Repository: $repo"
    Write-Host "Local GPU via WSL2 ($Distro). No server rental. Action=$Action Profile=$Profile"
    & wsl.exe -d $Distro --cd $linuxRepo --exec bash training/local8gb/start.sh $Action $Profile
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'See START_LOCAL_8GB.txt in the repository. No training was started if setup failed.'
    exit 1
}
