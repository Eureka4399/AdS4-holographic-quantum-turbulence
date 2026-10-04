# Holographic Quantum Turbulence

MATLAB code for simulating and analyzing quantum turbulence in an AdS4 holographic superfluid. The repository includes stationary radial profiles, driven-flow and vortex-lattice time evolution, vortex diagnostics, energy spectra, structure functions, and figure-generation scripts.

Checkpoint data are intentionally excluded from the repository. Generate the required checkpoints with the simulation scripts before running the corresponding analysis sections. Compact stationary inputs and selected processed results are included.

## Repository layout

```text
.
├── setup_project.m          # MATLAB path setup
├── hqt_*_path.m             # Data and output path helpers
├── hqt_paths.m              # Absolute project directories
├── src/
│   ├── numerics/            # Finite differences and Fourier filtering
│   ├── solvers/             # Stationary Newton solver
│   ├── diagnostics/         # Vortices, clustering, and structure functions
│   └── utilities/           # Optional GPU compatibility setup
├── scripts/
│   ├── simulation/          # Time-evolution scripts
│   ├── analysis/            # Checkpoint analysis and paper figures
│   └── legacy/              # Earlier stationary/thermodynamics notebooks
├── examples/                # Standalone exploratory example
├── third_party/dmsuite/      # Chebyshev and Fourier differentiation matrices
├── data/
│   ├── reference/           # Compact stationary profiles included in Git
│   ├── processed/           # Processed arrays; two compact results in Git
│   ├── raw/                 # Generated checkpoints, excluded from Git
│   └── metadata/            # Data inventory, sizes, and SHA-256 checksums
├── outputs/                 # Generated figures, movies, and logs
├── docs/                    # Usage, data conventions, and publishing guide
├── tests/                   # Short runtime checks with saving disabled
└── tools/                   # File inventory and repository packaging
```

## Requirements

- MATLAB. R2025a is the reference environment for the modern graphics calls such as `theme`.
- Parallel Computing Toolbox and a MATLAB-supported GPU for the GPU workflows. Several analysis routines also use `gather`.
- Optimization Toolbox for sections using `lsqcurvefit`.
- Statistics and Machine Learning Toolbox for sections using `nlinfit` and `statset`.

The Chebyshev and Fourier differentiation routines are bundled with their original author credits. See [third-party attribution](THIRD_PARTY.md).

## Getting started

Open MATLAB in the repository root and initialize the project:

```matlab
paths = setup_project();
```

To run the checkpoint-producing driven-flow configuration:

```matlab
tevol_driven_cluster
```

The default parameters are `mu = 6`, drive index `kk = 7`, `nx = ny = 301`, `nz = 20`, and `dt = 0.02`. Checkpoints are saved under `data/raw/` at intervals of one simulation-time unit. Edit the script parameters to select a different configuration.

For a small standalone plotting example:

```matlab
structure_function_vs_energy_spectrum
```

The analysis files are MATLAB notebooks organized with `%%` sections. Run their initialization section first, then the sections needed for the selected dataset. Match the run-folder name, grid, domain length, and time range to the simulation. Their initialization enables GPU compatibility before reading saved GPU arrays. See [the workflow guide](docs/WORKFLOWS.md) for entry points and parameter conventions.

For short runtime checks with all result saving and export disabled, see [tests/README.md](tests/README.md).

## Data

The repository includes three stationary profiles, `average2.mat`, and `structure_factor.mat`, totaling approximately 1.65 MB. The other processed arrays are excluded from Git to keep the repository small. All ten source MAT files are recorded in the [data manifest](data/metadata/manifest.csv).

See [the data guide](data/README.md) for storage policy and [the data dictionary](docs/DATA_DICTIONARY.md) for field names, array ordering, and checkpoint conventions. Checkpoint omission is intentional; these files are intermediate simulation outputs.

## Using and extending the code

Code comments and documentation are written in English. Existing scientific formulas, parameter defaults, differentiation routines, and analysis sections are retained. File paths are resolved relative to the repository so the code can be used outside the original author's machine.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution conventions and [the publishing guide](docs/PUBLISHING.md) for GitHub upload instructions.

## Citation and attribution

When using this code in research, cite *Holographic Quantum Turbulence* and identify the release tag or commit hash used in your calculations. Attribution and the literature reference for the bundled differentiation routines are provided in [THIRD_PARTY.md](THIRD_PARTY.md).
