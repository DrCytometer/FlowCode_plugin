# Installing and updating Honeychrome with the FlowCode plugin

This guide is for people who have not used the Terminal or Python before. One command installs everything; the same command (or a Desktop icon) updates it later.

You need:

- A Mac (Apple Silicon or Intel, macOS 13 or newer) or a 64-bit Windows 10/11 PC.
- An administrator account on that computer. On a university-managed computer, ask IT first: the installer adds developer tools (Homebrew on Mac, Visual Studio Build Tools on Windows).
- About 10 GB of free disk space on Windows and 5 GB on Mac, plus room for your experiments: the FlowCode Analysis page keeps working copies of the samples in each experiment's `cache` folder (tens of MB per sample).
- A good internet connection and time: 30-60 minutes on a Mac, 1-2 hours on Windows the first time. Updates take a few minutes. Plug the laptop in and stop it from going to sleep.

Everything is installed in a folder called `Honeychrome` in your home folder. Your experiments live in `Experiments`, also in your home folder. The installer never touches your experiments.

---

## 1. First installation

### Mac

1. Open **Terminal**: press Cmd + Space, type `Terminal`, press Return.
2. Copy this whole line, paste it into the Terminal window (Cmd + V) and press Return:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.sh)"
   ```

3. Answer the prompts that appear:
   - A window may ask to install the **command line developer tools**. Click **Install**, then **Agree**. The script waits until it finishes.
   - Terminal may ask for your **password**. This is your Mac login password. Nothing appears while you type it; that is normal. Press Return when done.
4. Wait until you see **Done.** in green. Leave the window open until then. If you see red text saying **Installation stopped**, go to section 5.

### Windows

1. Open **PowerShell**: click Start, type `PowerShell`, click **Windows PowerShell**. (You do not need "Run as administrator".)
2. Copy this whole line, paste it into the PowerShell window (right-click pastes) and press Enter:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.ps1 | iex"
   ```

3. Windows will ask several times **"Do you want to allow this app to make changes to your device?"** Click **Yes** each time.
4. The Visual Studio Build Tools and the Armadillo library take a long time the first time. This is expected.
5. Wait until you see **Done.** in green. If you see red text saying **Installation stopped**, go to section 5.

---

## 2. Starting Honeychrome and turning on the FlowCode plugin

1. Double-click **Honeychrome** on the Desktop. A Terminal (Mac) or black console (Windows) window opens first, then the app. Keep that window open while you work: closing it closes the app, and the messages in it help when something goes wrong.
2. **First time only:** open **Application Configuration**, tick **Enable FlowCode** and accept. Honeychrome reloads and a **FlowCode** tab appears.

---

## 3. Cohorts acquired over several days

If your mice were acquired on several days, each day with its own single-stained controls, unstained samples and FRET backbone, process each day on its own and bring the days together in the Analysis:

1. Make one Honeychrome experiment per day (for example `Cohort1_Day1` and `Cohort1_Day2` in `Experiments`).
2. In each, set up the spectral model and AutoSpectral with that day's controls and run FlowCode sections 1 to 5, using the same tags, the same combinations CSV (or the same libraries) and the same panel. Export the samples in section 5.
3. Open the experiment you will analyse in (any of the days). In **FlowCode › 6. Analysis › Data**, set **Source** to **Exported files**, then under **Other experiments** click **Add folder…** and choose each other day's export folder (`FlowCode Unmixed` inside that experiment's folder, unless you chose another one in section 5).
4. Check the sample table: every file is listed with its status. A file from another day is used only when it has the same combinations and every channel of this experiment.
5. Draw the unmixed gates in this experiment so they suit every day: the other days' files are gated with its gates.
6. Set groups and pairing as usual. When the days differ, an **acquisition date** covariate is added and adjusted for automatically.

Keep the other days' export folders where they are: the analysis finds them by their location. If you change a day's FlowCode settings, export that day again. Spread your groups across days when you plan the experiment: if one group was acquired only on one day and the other group only on another, the statistics cannot tell the day from the group.

---

## 4. Updating

When Oliver says there is an update:

1. Quit Honeychrome.
2. Double-click **Update Honeychrome** on the Desktop.
3. Wait for **Done.**, then start Honeychrome again.

If the Desktop icon is missing, run the installation command from section 1 again. It is the same command for installing and updating.

The update downloads the newest Honeychrome (the `autospectral` branch) and the newest FlowCode plugin, reinstalls Python packages only if they changed, recompiles the speed-up kernels only if their code changed, and checks that the plugin fits that Honeychrome before copying it.

Do not edit files inside the `Honeychrome` folder. An update puts every file back to the published version (your edits are put aside with `git stash`, not deleted, but they will no longer be active).

---

## 5. When something goes wrong

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
| `the FlowCode plugin needs a newer Honeychrome` | The published plugin is ahead of the published Honeychrome. Nothing was broken: your previous plugin was kept. Send the log to Oliver and run the update again when he says. |
| No FlowCode tab after enabling it | Look for `ImportError` in `Experiments/honeychrome.log`, then run the update again. |
| An Analysis note starting `Pathways:` that says `No such file or directory` | The plugin's data tables were not copied. Run the update again. |
| `has no attribute` in `Experiments/honeychrome.log` when running statistics | The plugin is newer than Honeychrome. Run the update again; if the message stays, send the log. |
| A file from another day (Analysis › Other experiments) is listed but not used | Its status says why: it was exported with a different combinations table, or lacks a channel of this experiment. Use the same combinations and panel in every day's experiment and export it again. |
| Phenotype shift says the spillover correction was skipped | The correction needs **Source: Live unmix**. With exported files (including other days' files) the shift runs without it. |

A full rebuild reinstalls every Python package and recompiles every kernel:

- Mac (Terminal):

  ```bash
  HONEYCHROME_FULL=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.sh)"
  ```

- Windows (PowerShell):

  ```powershell
  $env:HONEYCHROME_FULL = '1'; irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install_honeychrome.ps1 | iex
  ```

---

## 6. What gets installed where

| Location | Contents |
|---|---|
| `~/Honeychrome/honeychrome` | Honeychrome source code (`autospectral` branch) |
| `~/Honeychrome/FlowCode_plugin` | FlowCode plugin source code |
| `~/Honeychrome/venv` | The Python environment Honeychrome runs in |
| `~/Honeychrome/vcpkg` | Windows only: the C++ library builder used for the kernels |
| `~/Experiments/plugins` | The copy of the plugin that Honeychrome loads: its Python files and data tables (gene groups, the Gene Ontology pathways and MGI gene names). A `flowcode_phenotypes.csv` left there by an older version is no longer used |
| `~/Experiments` | Your experiments and the app log. The FlowCode Analysis keeps its working copies in `cache/flowcode_unmixed` and `cache/flowcode_analysis` inside each experiment; do not delete the rest of `cache`, which holds the FlowCode characterisation |
| Mac: Homebrew (`/opt/homebrew`) | Git, Python 3.12, CMake, Armadillo, libomp |
| Windows: Program Files | Git, CMake, Visual Studio 2022 Build Tools; Python 3.12 in your user folder |

`~` means your home folder (`/Users/<you>` on a Mac, `C:\Users\<you>` on Windows).

To remove Honeychrome, delete the `Honeychrome` folder and the two Desktop icons. Keep `Experiments` if you want your data.
