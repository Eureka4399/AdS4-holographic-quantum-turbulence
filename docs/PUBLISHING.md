# Publishing on GitHub

The upload archive contains the MATLAB source, English documentation, third-party attribution, runtime checks, five compact MAT files, and the data inventory. Checkpoints, large processed datasets, generated outputs, and the private source backup are excluded.

## Upload the repository

1. Extract `holographic-quantum-turbulence-github.zip` and open the inner `holographic-quantum-turbulence` directory.
2. Create an empty GitHub repository. The archive already includes `README.md` and `.gitignore`, so use the supplied files rather than generating replacements.
3. Open the repository's upload page: use **uploading an existing file** for an empty repository, or **Add file > Upload files** for an existing one.
4. Select all files and subdirectories inside the extracted directory and drag them from the file manager onto the upload area. The file chooser selects files; drag and drop also accepts folders and preserves their paths.
5. Confirm that the upload list includes `.gitignore`, `.gitattributes`, and `.editorconfig`, and that paths such as `src/numerics/FD.m` are retained. Enter `Initial release` as the commit message and submit the upload.

After uploading, `README.md`, `setup_project.m`, `src/`, `scripts/`, and `data/` should appear directly at the repository root.

See [GitHub's upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository) for the browser workflow.

## Data included in the repository

The three stationary profiles and two compact processed results total approximately 1.65 MB. The other five source MAT files are represented by filenames, sizes, and checksums in the data manifest. Simulation checkpoints and large processed results are intentionally omitted; use the simulation and analysis workflows to generate the required data.

## Build an upload archive

With Python 3.9 or newer and Git installed:

```text
python tools/package_repository.py
```

The archive is written to `outputs/holographic-quantum-turbulence-github.zip`. It excludes Git history, large data, checkpoints, output files, and the private migration backup. Extract the archive before uploading the repository files through GitHub's browser interface.
