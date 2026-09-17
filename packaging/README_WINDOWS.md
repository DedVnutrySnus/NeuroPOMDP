# NeuroPOMDP Windows Build

This folder contains the Windows packaging files for the standalone NeuroPOMDP desktop application.

## Build

Run:

```bat
packaging\build_windows.bat
```

or from the repository root:

```bat
build_windows.bat
```

The build script will:

1. create a local build virtual environment if needed,
2. install project requirements,
3. install PyInstaller,
4. build a folder-based Windows bundle.

## Output

The executable is generated at:

```text
dist\NeuroPOMDP\NeuroPOMDP.exe
```

The surrounding `dist\NeuroPOMDP\` folder contains the bundled runtime files required to run the app on a machine without Python installed separately.

## Run

Double-click:

```text
dist\NeuroPOMDP\NeuroPOMDP.exe
```

The launcher starts a local Streamlit server on `127.0.0.1`, waits for it to become ready, and opens the default browser automatically.

For development, you can also run the launcher directly from source:

```bash
python -m neuropomdp.launcher
```

## Troubleshooting

If startup fails:

1. Confirm Windows Defender or another security tool is not blocking the executable.
2. Check whether another NeuroPOMDP instance is already running.
3. Rebuild after deleting `build\` and `dist\` if a previous build was interrupted.
4. Run `dist\NeuroPOMDP\NeuroPOMDP.exe` from `cmd.exe` to capture any launcher error message.

If the browser does not open automatically, the local dashboard is usually still available at the localhost URL printed by the launcher.

## Release ZIP

To create a release archive, zip the entire `dist\NeuroPOMDP\` directory:

```text
NeuroPOMDP.zip
  └── NeuroPOMDP\
      ├── NeuroPOMDP.exe
      └── bundled runtime files
```

Distribute the ZIP, not the individual executable.
