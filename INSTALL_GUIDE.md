# Installing and updating Honeychrome with the FlowCode plugin

This guide is for people who have not used the Terminal or Python before. One command installs everything; the same command (or a Desktop icon) updates it later.

You need:

- A Mac (Apple Silicon or Intel, macOS 13 or newer) or a 64-bit Windows 10/11 PC.
- An administrator account on that computer. On a university-managed computer, ask IT first: the installer adds developer tools (Homebrew on Mac, Visual Studio Build Tools on Windows).
- About 10 GB of free disk space on Windows and 5 GB on Mac.
- A good internet connection and time: 30-60 minutes on a Mac, 1-2 hours on Windows the first time. Updates take a few minutes. Plug the laptop in and stop it from going to sleep.

Everything is installed in a folder called `Honeychrome` in your home folder. Your experiments live in `Experiments`, also in your home folder. The installer never touches your experiments.

---

## 1. First installation

### Mac

1. Open **Terminal**: press Cmd + Space, type `Terminal`, press Return.
2. Copy this whole line, paste it into the Terminal window (Cmd + V) and press Return:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.sh)"
   ```

3. Answer the prompts that appear:
   - A window may ask to install the **command line developer tools**. Click **Install**, then **Agree**. The script waits until it finishes.
   - Terminal may ask for your **password**. This is your Mac login password. Nothing appears while you type it; that is normal. Press Return when done.
4. Wait until you see **Done.** in green. Leave the window open until then. If you see red text saying **Installation stopped**, go to section 4.

### Windows

1. Open **PowerShell**: click Start, type `PowerShell`, click **Windows PowerShell**. (You do not need "Run as administrator".)
2. Copy this whole line, paste it into the PowerShell window (right-click pastes) and press Enter:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.ps1 | iex"
   ```

3. Windows will ask several times **"Do you want to allow this app to make changes to your device?"** Click **Yes** each time.
4. The Visual Studio Build Tools and the Armadillo library take a long time the first time. This is expected.
5. Wait until you see **Done.** in green. If you see red text saying **Installation stopped**, go to section 4.

---

## 2. Starting Honeychrome and turning on the FlowCode plugin

1. Double-click **Honeychrome** on the Desktop. A Terminal (Mac) or black console (Windows) window opens first, then the app. Keep that window open while you work: closing it closes the app, and the messages in it help when something goes wrong.
2. **First time only:** open **Application Configuration**, tick **Enable FlowCode** and accept. Honeychrome reloads and a **FlowCode** tab appears.

---

## 3. Updating

When Oliver says there is an update:

1. Quit Honeychrome.
2. Double-click **Update Honeychrome** on the Desktop.
3. Wait for **Done.**, then start Honeychrome again.

If the Desktop icon is missing, run the installation command from section 1 again. It is the same command for installing and updating.

The update downloads the newest Honeychrome (the `autospectral` branch) and the newest FlowCode plugin, reinstalls Python packages only if they changed, and recompiles the speed-up kernels only if their code changed.

Do not edit files inside the `Honeychrome` folder. An update puts every file back to the published version (your edits are put aside with `git stash`, not deleted, but they will no longer be active).

---

## 4. When something goes wrong

The installer writes everything it does to a log file:

- Mac: `~/Honeychrome/install.log` (in Finder: Go > Home > Honeychrome)
- Windows: `C:\Users\<you>\Honeychrome\install.log`

The app writes its own log to `Experiments/honeychrome.log` in your home folder.

Before asking for help, note which step failed (the last blue `==>` line), and have the end of the relevant log ready. The Honeychrome student Project in Claude can read a pasted log and suggest what to do.

| Message | What to do |
|---|---|
| `Honeychrome is running` | Quit Honeychrome (and close its Terminal/console window), then run the update again. |
| `... is not one` (admin account, Mac) | Your account cannot install Homebrew. Ask IT for administrator rights or for Homebrew to be installed. |
| `winget was not found` (Windows) | Install **App Installer** from the Microsoft Store, then run the command again. |
| `Python 3.12 was not found` or `CMake was not found` (Windows) | Restart the computer and run the command again. |
| `Visual Studio Build Tools are installed without the C++ tools` | Open **Visual Studio Installer**, click **Modify** on Build Tools 2022, tick **Desktop development with C++**, install, and run the command again. |
| `a kernel did not load` | Run the installer once more with a full rebuild (below). If it fails again, send the log. |
| No FlowCode tab after enabling it | Look for `ImportError` in `Experiments/honeychrome.log`, then run the update again. |

A full rebuild reinstalls every Python package and recompiles every kernel:

- Mac (Terminal):

  ```bash
  HONEYCHROME_FULL=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.sh)"
  ```

- Windows (PowerShell):

  ```powershell
  $env:HONEYCHROME_FULL = '1'; irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.ps1 | iex
  ```

---

## 5. What gets installed where

| Location | Contents |
|---|---|
| `~/Honeychrome/honeychrome` | Honeychrome source code (`autospectral` branch) |
| `~/Honeychrome/FlowCode_plugin` | FlowCode plugin source code |
| `~/Honeychrome/venv` | The Python environment Honeychrome runs in |
| `~/Honeychrome/vcpkg` | Windows only: the C++ library builder used for the kernels |
| `~/Experiments/plugins` | The copy of the plugin that Honeychrome loads |
| `~/Experiments` | Your experiments and the app log |
| Mac: Homebrew (`/opt/homebrew`) | Git, Python 3.12, CMake, Armadillo, libomp |
| Windows: Program Files | Git, CMake, Visual Studio 2022 Build Tools; Python 3.12 in your user folder |

`~` means your home folder (`/Users/<you>` on a Mac, `C:\Users\<you>` on Windows).

To remove Honeychrome, delete the `Honeychrome` folder and the two Desktop icons. Keep `Experiments` if you want your data.
