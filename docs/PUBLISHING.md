# Publishing on GitHub

The upload contains the MATLAB source, English documentation, third-party attribution, five compact MAT files, and the data inventory. Checkpoints, large processed datasets, generated outputs, and the local source backup are excluded by `.gitignore`.

## Upload the repository

Create an empty GitHub repository, then use its actual URL from the repository root:

```text
git add .
git status --short
git commit -m "Organize holographic quantum turbulence code and data"
git remote add origin <your-repository-url>
git push -u origin main
```

If creating a fresh copy from the upload ZIP, initialize it first with `git init -b main`. The packaging tool includes only files eligible for Git, and the included MAT files together are approximately 1.64 MB.

After creating the public repository, add its URL, author names, and associated paper identifier to the citation section in `README.md`. Choose the project code and data licenses with the rights holders, keeping the upstream differentiation routines separately attributed.

## Publish large data separately

The large processed MAT files remain available locally, and their checksums and sizes are included in the manifest. A separate research-data archive can distribute them while keeping the GitHub checkout small. Link the archive or DOI from `data/README.md` when it is published.

GitHub rejects ordinary Git files above 100 MiB. The largest local dataset is approximately 9.25 GB; some other files are approximately 6.14–6.16 GB, so they also exceed the documented LFS per-file limits. See [ordinary Git file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## Build an upload archive

With Python 3.9 or newer and Git installed:

```text
python tools/package_repository.py
```

The archive is written to `outputs/holographic-quantum-turbulence-github.zip`. It excludes Git history, large data, checkpoints, output files, and the private migration backup. Extract the archive before uploading the repository files through GitHub's browser interface.
