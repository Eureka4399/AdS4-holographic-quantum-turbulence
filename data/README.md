# Data organization

The data manifest records **10 source MAT files**, totaling **22,431,964,862 bytes** (22.43 GB; 20.89 GiB). Five compact files are included in the repository. Included data keep their original filenames and binary contents.

| Directory | Content | Git distribution |
| --- | --- | --- |
| `reference/` | Stationary radial profiles used to initialize simulations | Three compact profiles included |
| `processed/` | Boundary time series, time-averaged spectra, and vortex statistics | `average2.mat` and `structure_factor.mat` included |
| `raw/` | Per-time-step or periodic simulation checkpoints | Excluded; generate as needed |
| `metadata/` | Inventory with sizes, file formats, and SHA-256 checksums | Included |

## Included files

| File | Size in bytes | Purpose |
| --- | ---: | --- |
| `reference/mu=5,nz=30,vy=3.8_20.mat` | 3,968 | Stationary input for `mu=5`, drive index `3.8` |
| `reference/mu=6,nz=30,vy=7_20.mat` | 1,376 | Stationary input for the default driven configuration |
| `reference/mu=8,nz=30,vy=13.5_20.mat` | 3,968 | Stationary input for `mu=8`, drive index `13.5` |
| `processed/average2.mat` | 1,590,996 | Compact averaged diagnostics |
| `processed/structure_factor.mat` | 49,389 | Compact vortex structure-factor results |

The `vy` token in stationary filenames is a drive index: the scripts use the boundary parameter `vy = -2*pi*kk/20`. It is not the physical value of `vy` itself.

## Large processed datasets

The remaining five MAT files are excluded from the repository and represented by their inventory entries. The recorded filenames and checksums identify these source datasets. The repository does not use LFS; generate the processed results through the corresponding simulation and analysis workflows.

GitHub blocks ordinary Git files larger than 100 MiB, and LFS limits depend on the account plan. See [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## Checkpoints

Checkpoint files were intentionally removed from the shared collection and are not part of the upload. Recreate the run needed for an analysis, and store its checkpoint folder under `raw/`. See [WORKFLOWS.md](../docs/WORKFLOWS.md) for the simulation entry points.

## File verification

The inventory tools read file bytes; they do not load MATLAB arrays or execute simulations. With Python 3.9 or newer:

```text
python tools/data_inventory.py --verify --included-only
```

To check the complete local collection, use `--verify` without `--included-only`. Use `--refresh` only after intentionally adding or replacing datasets; it writes a new manifest.
