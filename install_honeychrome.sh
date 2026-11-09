#!/bin/bash
# Install or update Honeychrome and the FlowCode plugin on macOS.
#
# Open Terminal and run:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.sh)"
#
# The same command installs on the first run and updates on every later run.
# After the first run, the "Update Honeychrome" icon on the Desktop runs it.
#
# What it does:
#   1. Installs Apple's command line developer tools and Homebrew if missing
#      (an administrator password is asked for once).
#   2. Installs Git, Python 3.12, CMake, Armadillo and libomp with Homebrew.
#   3. Downloads Honeychrome (branch below) and the FlowCode plugin into
#      ~/Honeychrome, or updates them to the latest version.
#   4. Creates a Python environment in ~/Honeychrome/venv and installs
#      Honeychrome's requirements when they have changed.
#   5. Compiles the speed-up kernels (SOM, AutoSpectral optimisation,
#      noise-model GLS, Fix My Unmix) and checks that each one loads.
#   6. Checks that every Honeychrome module, function and setting the
#      FlowCode plugin uses is present, and stops before copying it if not.
#   7. Copies the FlowCode plugin files (Python modules and the data tables
#      read beside them) into ~/Experiments/plugins.
#   8. Puts "Honeychrome" and "Update Honeychrome" icons on the Desktop.
#
# Everything is logged to ~/Honeychrome/install.log.
#
# Settings can be overridden for one run, for example:
#   HONEYCHROME_BRANCH=main /bin/bash -c "$(curl -fsSL <script url>)"
#   HONEYCHROME_FULL=1 forces the requirements and kernels to be reinstalled.

set -Eeuo pipefail

HONEYCHROME_REPO="${HONEYCHROME_REPO:-https://github.com/Cytkit/honeychrome.git}"
HONEYCHROME_BRANCH="${HONEYCHROME_BRANCH:-autospectral}"
FLOWCODE_REPO="${FLOWCODE_REPO:-https://github.com/DrCytometer/FlowCode_plugin.git}"
FLOWCODE_BRANCH="${FLOWCODE_BRANCH:-main}"
SCRIPT_URL="${SCRIPT_URL:-https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.sh}"
FULL="${HONEYCHROME_FULL:-0}"

PYTHON_FORMULA="python@3.12"
BREW_FORMULAE=(git "$PYTHON_FORMULA" cmake armadillo libomp)

INSTALL_DIR="$HOME/Honeychrome"
APP_DIR="$INSTALL_DIR/honeychrome"
PLUGIN_SRC="$INSTALL_DIR/FlowCode_plugin"
VENV="$INSTALL_DIR/venv"
VPY="$VENV/bin/python"
PLUGINS_DIR="$HOME/Experiments/plugins"
LOG="$INSTALL_DIR/install.log"
REQ_STAMP="$INSTALL_DIR/.requirements.sha256"
KERNEL_STAMP="$INSTALL_DIR/.kernels.sha256"

# Plugin repository files that are not part of the plugin itself.
PLUGIN_SKIP=(flowcode_rescue_assess.py)

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() {
    printf '\n\033[1;31mInstallation stopped: %s\033[0m\n' "$*" >&2
    printf 'The full log is in %s. Send it to Oliver if you need help.\n' "$LOG" >&2
    exit 1
}
trap 'fail "a command failed (line $LINENO). See the messages above."' ERR

# ---------------------------------------------------------------------------
# Checks before anything is changed
# ---------------------------------------------------------------------------

[[ "$(uname -s)" == "Darwin" ]] || { echo "This script is for macOS. On Windows use install_honeychrome.ps1."; exit 1; }

mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG") 2>&1
printf '\n===== %s  Honeychrome installer =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

# Only Python processes are checked: this script's own command line contains
# the app's module name too.
if ps -axo comm=,args= | awk '$1 ~ /[Pp]ython/ && /honeychrome[.]main|bin\/honeychrome( |$)/ { found = 1 } END { exit !found }'; then
    fail "Honeychrome is running. Quit Honeychrome, then run this again."
fi

# ---------------------------------------------------------------------------
# Command line tools and Homebrew
# ---------------------------------------------------------------------------

step "Checking Apple's command line developer tools"
if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install >/dev/null 2>&1 || true
    note "A window has opened asking to install the command line developer tools."
    note "Click Install and accept the licence. This script waits until it finishes."
    until xcode-select -p >/dev/null 2>&1; do sleep 15; done
fi
note "Found at $(xcode-select -p)"

find_brew() {
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        [[ -x "$b" ]] && { echo "$b"; return 0; }
    done
    return 1
}

step "Checking Homebrew"
if ! BREW="$(find_brew)"; then
    if ! dseditgroup -o checkmember -m "$USER" admin >/dev/null 2>&1; then
        fail "Homebrew needs an administrator account and $USER is not one. Ask your IT team for admin rights, or ask them to install Homebrew (https://brew.sh)."
    fi
    note "Homebrew is not installed. Your Mac login password is needed once."
    note "(Nothing appears on screen while you type the password; press Return when done.)"
    sudo -v
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    BREW="$(find_brew)" || fail "Homebrew did not install."
fi
eval "$("$BREW" shellenv)"
if ! grep -qs "brew shellenv" "$HOME/.zprofile"; then
    printf '\neval "$(%s shellenv)"\n' "$BREW" >> "$HOME/.zprofile"
fi
note "Using $BREW"

step "Installing tools with Homebrew (git, Python 3.12, CMake, Armadillo, libomp)"
for f in "${BREW_FORMULAE[@]}"; do
    if brew list --formula "$f" >/dev/null 2>&1; then
        note "$f already installed"
    else
        brew install "$f"
    fi
done
PY="$(brew --prefix "$PYTHON_FORMULA")/bin/python3.12"
[[ -x "$PY" ]] || fail "Python 3.12 was not found at $PY."
note "Python: $("$PY" --version)"

# ---------------------------------------------------------------------------
# Source code
# ---------------------------------------------------------------------------

# Clone a repository, or bring an existing clone to the tip of the branch.
# Local edits to tracked files are stashed (kept in git stash), never lost.
sync_repo() {
    local url="$1" branch="$2" dir="$3" label="$4" before after
    if [[ -d "$dir/.git" ]]; then
        before="$(git -C "$dir" rev-parse --short HEAD)"
        git -C "$dir" fetch --quiet origin "+refs/heads/$branch:refs/remotes/origin/$branch"
        if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
            git -C "$dir" stash push --quiet -m "Saved by the Honeychrome installer $(date '+%Y-%m-%d %H:%M')"
            note "Local changes to $label were saved with 'git stash' before updating."
        fi
        git -C "$dir" checkout --quiet --force -B "$branch" "origin/$branch"
        after="$(git -C "$dir" rev-parse --short HEAD)"
        if [[ "$before" == "$after" ]]; then
            note "$label is up to date ($branch @ $after)"
        else
            note "$label updated: $before -> $after ($branch)"
        fi
    else
        git clone --quiet --branch "$branch" "$url" "$dir"
        note "$label downloaded ($branch @ $(git -C "$dir" rev-parse --short HEAD))"
    fi
}

step "Downloading Honeychrome and the FlowCode plugin"
sync_repo "$HONEYCHROME_REPO" "$HONEYCHROME_BRANCH" "$APP_DIR" "Honeychrome"
sync_repo "$FLOWCODE_REPO" "$FLOWCODE_BRANCH" "$PLUGIN_SRC" "FlowCode plugin"

# ---------------------------------------------------------------------------
# Python environment and requirements
# ---------------------------------------------------------------------------

step "Preparing the Python environment"
if [[ -x "$VPY" ]] && ! "$VPY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    note "The existing environment is not Python 3.12; rebuilding it."
    rm -rf "$VENV"
    FULL=1
fi
if [[ ! -x "$VPY" ]]; then
    "$PY" -m venv "$VENV"
    FULL=1
    note "Created $VENV"
fi

cd "$APP_DIR"
req_hash="$(cat requirements.txt pyproject.toml | shasum -a 256 | cut -d' ' -f1)"
if [[ "$FULL" == "1" || "$req_hash" != "$(cat "$REQ_STAMP" 2>/dev/null || true)" ]]; then
    step "Installing Honeychrome's Python requirements (10-20 minutes the first time)"
    "$VPY" -m pip install --upgrade pip setuptools wheel
    "$VPY" -m pip install "$(grep -i '^numpy==' requirements.txt)"
    "$VPY" install_annoy.py
    "$VPY" -m pip install -r requirements.txt
    "$VPY" install_opentsne.py
    "$VPY" -m pip install -e .
    echo "$req_hash" > "$REQ_STAMP"
else
    note "Requirements unchanged since the last run"
    "$VPY" -m pip install --quiet --no-deps -e .
fi

# ---------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------

KDIR="$APP_DIR/src/honeychrome/bundled_plugins"
cd "$KDIR"
kernel_hash="$( { "$VPY" --version; git ls-files -z -- '*.c' '*.cpp' '*.hpp' CMakeLists.txt 'build_*kernel*.py' | xargs -0 cat; } | shasum -a 256 | cut -d' ' -f1)"
if [[ "$FULL" == "1" || "$kernel_hash" != "$(cat "$KERNEL_STAMP" 2>/dev/null || true)" ]]; then
    step "Compiling the speed-up kernels"
    # Built files are removed first: overwriting a loaded binary in place on
    # macOS invalidates its signature and the OS then kills the importing app.
    rm -rf build _som_kernel.* _autospectral_opt_kernel*.so _gls_m2p_kernel*.so
    HONEYCHROME_OPENMP=1 "$VPY" build_som_kernel.py
    HONEYCHROME_OPENMP=1 "$VPY" build_autospectral_opt_kernel.py
    "$VPY" build_fix_my_unmix_kernel.py
    echo "$kernel_hash" > "$KERNEL_STAMP"
else
    note "Kernel sources unchanged since the last build"
fi

step "Checking that Honeychrome and every kernel load"
"$VPY" - <<'PYCHECK' || { rm -f "$KERNEL_STAMP"; fail "a kernel did not load. Run the installer again with HONEYCHROME_FULL=1, and send the log if it still fails."; }
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
    'SOM (clustering, phenotype shift)': SOM_KERNEL_AVAILABLE,
    'AutoSpectral optimisation': AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
    'Noise-model GLS': GLS_M2P_KERNEL_AVAILABLE,
    'Fix My Unmix': fmu,
}
for name, ok in status.items():
    print(f"    {'OK     ' if ok else 'MISSING'} {name}")
sys.exit(0 if all(status.values()) else 1)
PYCHECK

# ---------------------------------------------------------------------------
# FlowCode plugin
# ---------------------------------------------------------------------------

# Every Honeychrome module, function and setting the plugin imports must be
# present; otherwise the plugin is newer than the Honeychrome branch and the
# files already in the plugins folder are left as they are.
step "Checking that the FlowCode plugin fits this Honeychrome"
cd "$APP_DIR"
"$VPY" - "$PLUGIN_SRC" "${PLUGIN_SKIP[@]}" <<'PYCHECK' || fail "the FlowCode plugin needs a newer Honeychrome than the $HONEYCHROME_BRANCH branch provides (see the MISSING lines above). No plugin files were copied, so a previously installed plugin stays as it was. Send the log to Oliver."
import ast
import importlib
import os
import sys
import types
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
src = Path(sys.argv[1])
skip_names = set(sys.argv[2:])


def guarded(tree):
    """Nodes inside try blocks: optional imports the plugin can do without."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for part in node.body + node.orelse + node.handlers + node.finalbody:
                ids.update(id(sub) for sub in ast.walk(part))
    return ids


imports, attributes = {}, {}
for path in sorted(src.glob('*.py')):
    if path.name in skip_names or path.name.endswith('_test.py'):
        continue
    tree = ast.parse(path.read_text(encoding='utf-8'))
    skip = guarded(tree)
    aliases = {}
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').split('.')[0] == 'honeychrome':
            for a in node.names:
                imports.setdefault((node.module, a.name), path.name)
                aliases[a.asname or a.name] = (node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] == 'honeychrome':
                    imports.setdefault((a.name, None), path.name)
                    if a.asname:
                        aliases[a.asname] = (a.name, None)
    for node in ast.walk(tree):
        if (id(node) not in skip and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id in aliases):
            attributes.setdefault((aliases[node.value.id], node.attr), path.name)


def resolve(module, name):
    obj = importlib.import_module(module)
    if name is None:
        return obj
    if hasattr(obj, name):
        return getattr(obj, name)
    return importlib.import_module(f'{module}.{name}')


found, missing = {}, []
for (module, name), where in sorted(imports.items(), key=lambda kv: (kv[0][0], kv[0][1] or '')):
    try:
        found[(module, name)] = resolve(module, name)
    except Exception as exc:
        label = module if name is None else f'{module}.{name}'
        missing.append(f'{label} (used in {where}; {type(exc).__name__}: {exc})')
for (target, attr), where in sorted(attributes.items(), key=lambda kv: (kv[0][0][0], kv[0][0][1] or '', kv[0][1])):
    obj = found.get(target)
    if isinstance(obj, types.ModuleType) and not hasattr(obj, attr):
        missing.append(f'{obj.__name__}.{attr} (used in {where})')
for line in missing:
    print(f'    MISSING {line}')
print(f'    {len(imports) + len(attributes)} Honeychrome names used by the plugin, {len(missing)} missing')
sys.exit(1 if missing else 0)
PYCHECK

step "Installing the FlowCode plugin into $PLUGINS_DIR"
mkdir -p "$PLUGINS_DIR"
copied=0
for f in "$PLUGIN_SRC"/*.py "$PLUGIN_SRC"/*.csv "$PLUGIN_SRC"/*.csv.gz; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f")"
    [[ " ${PLUGIN_SKIP[*]} " == *" $name "* || "$name" == *_test.py ]] && continue
    cp -f "$f" "$PLUGINS_DIR/$name"
    copied=$((copied + 1))
done
[[ "$copied" -gt 0 ]] || fail "no plugin files were found in $PLUGIN_SRC."
note "$copied plugin files copied"

# ---------------------------------------------------------------------------
# Desktop launchers
# ---------------------------------------------------------------------------

step "Creating the Desktop icons"
cat > "$INSTALL_DIR/Honeychrome.command" <<EOF
#!/bin/bash
# Start Honeychrome. Keep this window open while you use the app;
# messages here help diagnose problems.
cd "$APP_DIR" || exit 1
exec "$VPY" -m honeychrome.main
EOF
cat > "$INSTALL_DIR/Update Honeychrome.command" <<EOF
#!/bin/bash
# Update Honeychrome and the FlowCode plugin to the latest versions.
/bin/bash -c "\$(curl -fsSL $SCRIPT_URL)"
echo
read -r -n 1 -p "Press any key to close this window."
EOF
chmod +x "$INSTALL_DIR/Honeychrome.command" "$INSTALL_DIR/Update Honeychrome.command"
cp -f "$INSTALL_DIR/Honeychrome.command" "$INSTALL_DIR/Update Honeychrome.command" "$HOME/Desktop/"
note "Double-click 'Honeychrome' on the Desktop to start the app."

trap - ERR
printf '\n\033[1;32mDone.\033[0m Honeychrome %s @ %s, FlowCode plugin @ %s\n' \
    "$HONEYCHROME_BRANCH" "$(git -C "$APP_DIR" rev-parse --short HEAD)" "$(git -C "$PLUGIN_SRC" rev-parse --short HEAD)"
cat <<'EOF'

First time only: start Honeychrome, open Application Configuration, tick
"Enable FlowCode" and accept. The FlowCode tab then appears.

After an update: restart Honeychrome so the new code is loaded.
EOF