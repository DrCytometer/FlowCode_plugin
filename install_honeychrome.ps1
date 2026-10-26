# Install or update Honeychrome and the FlowCode plugin on Windows 10/11 (x64).
#
# Open PowerShell (Start menu > type "PowerShell") and run:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.ps1 | iex"
#
# The same command installs on the first run and updates on every later run.
# After the first run, the "Update Honeychrome" icon on the Desktop runs it.
#
# What it does:
#   1. Installs Git, Python 3.12, CMake and the Visual Studio C++ Build Tools
#      with winget if they are missing (Windows asks for permission to make
#      changes; the Build Tools download is several GB).
#   2. Installs vcpkg and builds Armadillo with OpenBLAS for the AutoSpectral
#      kernels (30-60 minutes the first time only).
#   3. Downloads Honeychrome (branch below) and the FlowCode plugin into
#      %USERPROFILE%\Honeychrome, or updates them to the latest version.
#   4. Creates a Python environment in %USERPROFILE%\Honeychrome\venv and
#      installs Honeychrome's requirements when they have changed.
#   5. Compiles the speed-up kernels (SOM, AutoSpectral optimisation,
#      noise-model GLS, Fix My Unmix) and checks that each one loads.
#   6. Copies the FlowCode plugin files into %USERPROFILE%\Experiments\plugins.
#   7. Puts "Honeychrome" and "Update Honeychrome" icons on the Desktop.
#
# Everything is logged to %USERPROFILE%\Honeychrome\install.log.
#
# Settings can be overridden for one run by setting environment variables
# first, for example:  $env:HONEYCHROME_BRANCH = 'main'
# $env:HONEYCHROME_FULL = '1' forces the requirements and kernels to be reinstalled.

function Get-Setting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

$HoneychromeRepo   = Get-Setting 'HONEYCHROME_REPO'   'https://github.com/Cytkit/honeychrome.git'
$HoneychromeBranch = Get-Setting 'HONEYCHROME_BRANCH' 'autospectral'
$FlowcodeRepo      = Get-Setting 'FLOWCODE_REPO'      'https://github.com/DrCytometer/FlowCode_plugin.git'
$FlowcodeBranch    = Get-Setting 'FLOWCODE_BRANCH'    'main'
$ScriptUrl         = Get-Setting 'SCRIPT_URL'         'https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.ps1'
$Full              = (Get-Setting 'HONEYCHROME_FULL' '0') -eq '1'

$InstallDir  = Join-Path $HOME 'Honeychrome'
$AppDir      = Join-Path $InstallDir 'honeychrome'
$PluginSrc   = Join-Path $InstallDir 'FlowCode_plugin'
$Venv        = Join-Path $InstallDir 'venv'
$VPy         = Join-Path $Venv 'Scripts\python.exe'
$VcpkgDir    = Join-Path $InstallDir 'vcpkg'
$Toolchain   = Join-Path $InstallDir 'vcpkg_toolchain.cmake'
$PluginsDir  = Join-Path $HOME 'Experiments\plugins'
$Log         = Join-Path $InstallDir 'install.log'
$ReqStamp    = Join-Path $InstallDir '.requirements.sha256'
$KernelStamp = Join-Path $InstallDir '.kernels.sha256'

# Static libraries linked against the dynamic C runtime, matching Python extensions.
$VcpkgTriplet = 'x64-windows-static-md'
# Plugin repository files that are not part of the plugin itself.
$PluginSkip = @('flowcode_rescue_assess.py')

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-Log([string]$Text) {
    Add-Content -Path $Log -Value $Text -Encoding UTF8
}

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
    Write-Log "==> $Text"
}

function Write-Note([string]$Text) {
    Write-Host "    $Text"
    Write-Log "    $Text"
}

# Run a program, show and log its output, and stop if it fails.
function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$WorkDir = $null) {
    Write-Log ">> $Exe $($Arguments -join ' ')"
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    if ($WorkDir) { Push-Location $WorkDir }
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Write-Log $line
        }
        $code = $LASTEXITCODE
    }
    finally {
        if ($WorkDir) { Pop-Location }
        $ErrorActionPreference = $saved
    }
    if ($code -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $code." }
}

# Run a program quietly and return its trimmed output.
function Get-Output([string]$Exe, [string[]]$Arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $out = & $Exe @Arguments 2>$null } finally { $ErrorActionPreference = $saved }
    if ($LASTEXITCODE -ne 0) { return $null }
    return ("$out").Trim()
}

function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Install-WingetPackage([string]$Id, [string]$Label, [string[]]$Extra = @()) {
    $null = Get-Output 'winget' @('list', '--id', $Id, '--exact', '--accept-source-agreements')
    if ($LASTEXITCODE -eq 0) {
        Write-Note "$Label already installed"
        return
    }
    Write-Note "Installing $Label"
    Invoke-Checked 'winget' (@('install', '--id', $Id, '--exact', '--silent',
        '--accept-source-agreements', '--accept-package-agreements') + $Extra)
    Update-PathFromRegistry
}

function Find-VcTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    return Get-Output $vswhere @('-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
}

function Find-Python312 {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $py = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($py) {
        $exe = Get-Output $py.Source @('-3.12', '-c', 'import sys; print(sys.executable)')
        if ($exe -and (Test-Path $exe)) { return $exe }
    }
    return $null
}

function Get-TextHash([string[]]$Files, [string]$Prefix = '') {
    $sb = New-Object System.Text.StringBuilder
    $null = $sb.Append($Prefix)
    foreach ($f in $Files) { $null = $sb.Append([IO.File]::ReadAllText($f)) }
    $sha = [Security.Cryptography.SHA256]::Create()
    $bytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($sb.ToString()))
    return ([BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

function Read-Stamp([string]$Path) {
    if (Test-Path $Path) { return (Get-Content $Path -Raw).Trim() }
    return ''
}

# Clone a repository, or bring an existing clone to the tip of the branch.
# Local edits to tracked files are stashed (kept in git stash), never lost.
function Sync-Repo([string]$Url, [string]$Branch, [string]$Dir, [string]$Label) {
    if (Test-Path (Join-Path $Dir '.git')) {
        $before = Get-Output 'git' @('-C', $Dir, 'rev-parse', '--short', 'HEAD')
        Invoke-Checked 'git' @('-C', $Dir, 'fetch', '--quiet', 'origin', "+refs/heads/${Branch}:refs/remotes/origin/$Branch")
        $dirty = Get-Output 'git' @('-C', $Dir, 'status', '--porcelain', '--untracked-files=no')
        if ($dirty) {
            $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
            Invoke-Checked 'git' @('-C', $Dir, 'stash', 'push', '--quiet', '-m', "Saved by the Honeychrome installer $stamp")
            Write-Note "Local changes to $Label were saved with 'git stash' before updating."
        }
        Invoke-Checked 'git' @('-C', $Dir, 'checkout', '--quiet', '--force', '-B', $Branch, "origin/$Branch")
        $after = Get-Output 'git' @('-C', $Dir, 'rev-parse', '--short', 'HEAD')
        if ($before -eq $after) { Write-Note "$Label is up to date ($Branch @ $after)" }
        else { Write-Note "$Label updated: $before -> $after ($Branch)" }
    }
    else {
        Invoke-Checked 'git' @('clone', '--quiet', '--branch', $Branch, $Url, $Dir)
        $head = Get-Output 'git' @('-C', $Dir, 'rev-parse', '--short', 'HEAD')
        Write-Note "$Label downloaded ($Branch @ $head)"
    }
}

function New-Shortcut([string]$Path, [string]$Target, [string]$Arguments, [string]$WorkDir, [string]$Icon) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($Path)
    $link.TargetPath = $Target
    $link.Arguments = $Arguments
    $link.WorkingDirectory = $WorkDir
    if ($Icon -and (Test-Path $Icon)) { $link.IconLocation = $Icon }
    $link.Save()
}

function Install-Honeychrome {
    $forceFull = $Full

    # ---------------------------------------------------------------------
    # Checks before anything is changed
    # ---------------------------------------------------------------------
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        throw "This installer supports 64-bit Intel/AMD Windows only (this PC reports $env:PROCESSOR_ARCHITECTURE)."
    }
    if (-not (Get-Command 'winget' -ErrorAction SilentlyContinue)) {
        throw "winget was not found. Install 'App Installer' from the Microsoft Store, then run this again."
    }
    $running = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
        Where-Object { $_.CommandLine -match 'honeychrome[.]main|Scripts\\honeychrome' }
    if ($running) { throw 'Honeychrome is running. Quit Honeychrome, then run this again.' }

    # ---------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------
    Write-Step 'Installing tools with winget (Git, Python 3.12, CMake)'
    Install-WingetPackage 'Git.Git' 'Git'
    Install-WingetPackage 'Python.Python.3.12' 'Python 3.12' @('--scope', 'user')
    Install-WingetPackage 'Kitware.CMake' 'CMake'
    Update-PathFromRegistry

    Write-Step 'Checking the Visual Studio C++ Build Tools'
    if (-not (Find-VcTools)) {
        Write-Note 'Installing the C++ Build Tools. This downloads several GB and can take 20-40 minutes.'
        Install-WingetPackage 'Microsoft.VisualStudio.2022.BuildTools' 'Visual Studio 2022 Build Tools' @('--override',
            '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
        if (-not (Find-VcTools)) {
            throw ("Visual Studio Build Tools are installed without the C++ tools. Open 'Visual Studio Installer', " +
                "click Modify on Build Tools 2022, tick 'Desktop development with C++', install, then run this again.")
        }
    }
    Write-Note "C++ tools found in $(Find-VcTools)"

    $py = Find-Python312
    if (-not $py) { throw 'Python 3.12 was not found after installation. Restart the computer and run this again.' }
    Write-Note "Python: $py"
    $cmakeCmd = Get-Command 'cmake' -ErrorAction SilentlyContinue
    if (-not $cmakeCmd) { throw 'CMake was not found after installation. Restart the computer and run this again.' }

    # ---------------------------------------------------------------------
    # Armadillo and OpenBLAS through vcpkg
    # ---------------------------------------------------------------------
    Write-Step 'Preparing Armadillo and OpenBLAS (vcpkg)'
    if (-not (Test-Path (Join-Path $VcpkgDir '.git'))) {
        Invoke-Checked 'git' @('clone', '--quiet', 'https://github.com/microsoft/vcpkg.git', $VcpkgDir)
    }
    $vcpkg = Join-Path $VcpkgDir 'vcpkg.exe'
    if (-not (Test-Path $vcpkg)) {
        Invoke-Checked (Join-Path $VcpkgDir 'bootstrap-vcpkg.bat') @('-disableMetrics')
    }
    Write-Note 'Building Armadillo the first time takes 30-60 minutes; later runs reuse it.'
    Invoke-Checked $vcpkg @('install', "armadillo:$VcpkgTriplet")
    $vcpkgCmake = (Join-Path $VcpkgDir 'scripts\buildsystems\vcpkg.cmake') -replace '\\', '/'
    Set-Content -Path $Toolchain -Encoding ASCII -Value @(
        "set(VCPKG_TARGET_TRIPLET $VcpkgTriplet CACHE STRING `"`")",
        "include(`"$vcpkgCmake`")"
    )

    # ---------------------------------------------------------------------
    # Source code
    # ---------------------------------------------------------------------
    Write-Step 'Downloading Honeychrome and the FlowCode plugin'
    Sync-Repo $HoneychromeRepo $HoneychromeBranch $AppDir 'Honeychrome'
    Sync-Repo $FlowcodeRepo $FlowcodeBranch $PluginSrc 'FlowCode plugin'

    # ---------------------------------------------------------------------
    # Python environment and requirements
    # ---------------------------------------------------------------------
    Write-Step 'Preparing the Python environment'
    if (Test-Path $VPy) {
        $ver = Get-Output $VPy @('-c', 'import sys; print("%d.%d" % sys.version_info[:2])')
        if ($ver -ne '3.12') {
            Write-Note 'The existing environment is not Python 3.12; rebuilding it.'
            Remove-Item -Recurse -Force $Venv
        }
    }
    if (-not (Test-Path $VPy)) {
        Invoke-Checked $py @('-m', 'venv', $Venv)
        $forceFull = $true
        Write-Note "Created $Venv"
    }

    $reqFiles = @((Join-Path $AppDir 'requirements.txt'), (Join-Path $AppDir 'pyproject.toml'))
    $reqHash = Get-TextHash $reqFiles
    if ($forceFull -or ($reqHash -ne (Read-Stamp $ReqStamp))) {
        Write-Step "Installing Honeychrome's Python requirements (10-20 minutes the first time)"
        $numpyPin = (Select-String -Path $reqFiles[0] -Pattern '^numpy==' | Select-Object -First 1).Line.Trim()
        Invoke-Checked $VPy @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel') $AppDir
        Invoke-Checked $VPy @('-m', 'pip', 'install', $numpyPin) $AppDir
        Invoke-Checked $VPy @('install_annoy.py') $AppDir
        Invoke-Checked $VPy @('-m', 'pip', 'install', '-r', 'requirements.txt') $AppDir
        Invoke-Checked $VPy @('install_opentsne.py') $AppDir
        Invoke-Checked $VPy @('-m', 'pip', 'install', '-e', '.') $AppDir
        Set-Content -Path $ReqStamp -Value $reqHash
    }
    else {
        Write-Note 'Requirements unchanged since the last run'
        Invoke-Checked $VPy @('-m', 'pip', 'install', '--quiet', '--no-deps', '-e', '.') $AppDir
    }

    # ---------------------------------------------------------------------
    # Compiled kernels
    # ---------------------------------------------------------------------
    $kdir = Join-Path $AppDir 'src\honeychrome\bundled_plugins'
    $tracked = (Get-Output 'git' @('-C', $kdir, 'ls-files', '--', '*.c', '*.cpp', '*.hpp', 'CMakeLists.txt', 'build_*kernel*.py')) -split '\s+' |
        Where-Object { $_ } | ForEach-Object { Join-Path $kdir $_ }
    $pyVersion = Get-Output $VPy @('--version')
    $kernelHash = Get-TextHash $tracked "$pyVersion $VcpkgTriplet"
    if ($forceFull -or ($kernelHash -ne (Read-Stamp $KernelStamp))) {
        Write-Step 'Compiling the speed-up kernels'
        Get-ChildItem -Path $kdir -Force | Where-Object {
            $_.Name -eq 'build' -or $_.Name -like '_som_kernel.*' -or
            $_.Name -like '_autospectral_opt_kernel*.pyd' -or $_.Name -like '_gls_m2p_kernel*.pyd'
        } | Remove-Item -Recurse -Force
        $env:HONEYCHROME_OPENMP = '1'
        $env:CMAKE_TOOLCHAIN_FILE = $Toolchain
        try {
            Invoke-Checked $VPy @('build_som_kernel.py') $kdir
            Invoke-Checked $VPy @('build_autospectral_opt_kernel.py') $kdir
            Invoke-Checked $VPy @('build_fix_my_unmix_kernel.py') $kdir
        }
        finally {
            Remove-Item Env:CMAKE_TOOLCHAIN_FILE -ErrorAction SilentlyContinue
        }
        Set-Content -Path $KernelStamp -Value $kernelHash
    }
    else {
        Write-Note 'Kernel sources unchanged since the last build'
    }

    Write-Step 'Checking that Honeychrome and every kernel load'
    $check = @'
import sys
sys.path.insert(0, '.')
import PySide6, honeychrome
from som_kernel_wrapper import SOM_KERNEL_AVAILABLE
from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE
from gls_m2p_kernel_wrapper import GLS_M2P_KERNEL_AVAILABLE
try:
    import _fmu_kernel
    fmu = hasattr(_fmu_kernel.lib, 'fmu_anchored_batch')
except ImportError:
    fmu = False
status = {
    'SOM (clustering)': SOM_KERNEL_AVAILABLE,
    'AutoSpectral optimisation': AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
    'Noise-model GLS': GLS_M2P_KERNEL_AVAILABLE,
    'Fix My Unmix': fmu,
}
for name, ok in status.items():
    print(f"    {'OK     ' if ok else 'MISSING'} {name}")
sys.exit(0 if all(status.values()) else 1)
'@
    $checkFile = Join-Path $InstallDir 'check_kernels.py'
    Set-Content -Path $checkFile -Value $check -Encoding UTF8
    try { Invoke-Checked $VPy @($checkFile) $kdir }
    catch {
        Remove-Item $KernelStamp -ErrorAction SilentlyContinue
        throw "A kernel did not load. Run the installer again with `$env:HONEYCHROME_FULL = '1', and send the log if it still fails."
    }

    # ---------------------------------------------------------------------
    # FlowCode plugin
    # ---------------------------------------------------------------------
    Write-Step "Installing the FlowCode plugin into $PluginsDir"
    New-Item -ItemType Directory -Force -Path $PluginsDir | Out-Null
    $files = Get-ChildItem -Path $PluginSrc -Filter '*.py' -File |
        Where-Object { ($PluginSkip -notcontains $_.Name) -and ($_.Name -notlike '*_test.py') }
    if (-not $files) { throw "No plugin files were found in $PluginSrc." }
    foreach ($f in $files) { Copy-Item -Force $f.FullName (Join-Path $PluginsDir $f.Name) }
    Write-Note "$(@($files).Count) plugin files copied"

    # ---------------------------------------------------------------------
    # Desktop shortcuts
    # ---------------------------------------------------------------------
    Write-Step 'Creating the Desktop icons'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $icon = Join-Path $AppDir 'src\honeychrome\view_components\assets\cytkit_web_logo.ico'
    New-Shortcut (Join-Path $desktop 'Honeychrome.lnk') $VPy '-m honeychrome.main' $AppDir $icon
    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    New-Shortcut (Join-Path $desktop 'Update Honeychrome.lnk') $psExe `
        "-NoProfile -NoExit -ExecutionPolicy Bypass -Command `"irm $ScriptUrl | iex`"" $InstallDir $null
    Write-Note "Double-click 'Honeychrome' on the Desktop to start the app."

    $appHead = Get-Output 'git' @('-C', $AppDir, 'rev-parse', '--short', 'HEAD')
    $pluginHead = Get-Output 'git' @('-C', $PluginSrc, 'rev-parse', '--short', 'HEAD')
    Write-Host ''
    Write-Host "Done. Honeychrome $HoneychromeBranch @ $appHead, FlowCode plugin @ $pluginHead" -ForegroundColor Green
    Write-Host ''
    Write-Host 'First time only: start Honeychrome, open Application Configuration, tick'
    Write-Host '"Enable FlowCode" and accept. The FlowCode tab then appears.'
    Write-Host ''
    Write-Host 'After an update: restart Honeychrome so the new code is loaded.'
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Log ''
Write-Log "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Honeychrome installer ====="
try {
    Install-Honeychrome
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Host ''
    Write-Host "Installation stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "The full log is in $Log. Send it to Oliver if you need help."
}