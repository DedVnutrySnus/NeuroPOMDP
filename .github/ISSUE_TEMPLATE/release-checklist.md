# Release checklist

## Pre-release

- [ ] Confirm tests pass in the project venv
- [ ] Confirm packaging build succeeds
- [ ] Confirm packaged EXE runs in diagnostic mode
- [ ] Confirm output archive is generated
- [ ] Confirm release notes are updated
- [ ] Confirm README usage instructions are current

## Windows package

- [ ] Extract archive to a clean directory
- [ ] Launch `NeuroPOMDP.exe`
- [ ] Verify dashboard starts
- [ ] Verify local browser opens without runtime import errors

## GitHub release publication

- [ ] Add tag `v0.1.0`
- [ ] Attach archive asset
- [ ] Attach SHA256 checksum
- [ ] Publish release notes
- [ ] Confirm the README links to the latest release
