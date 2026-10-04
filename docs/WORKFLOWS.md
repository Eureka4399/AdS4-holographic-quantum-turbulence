# Workflow guide

## Initialize the MATLAB session

Open the repository root and call:

```matlab
paths = setup_project();
```

The setup adds reusable functions, simulation scripts, analysis scripts, the example, and the differentiation suite to the MATLAB path. It does not select a GPU or run a calculation. Historical notebooks under `scripts/legacy/` are opened or run by their file paths.

Data and output helpers locate files relative to the repository, independently of MATLAB's current working directory:

```matlab
inputFile = hqt_data_path('reference', 'mu=6,nz=30,vy=7_20.mat');
figureFile = hqt_output_path('figures', 'spectrum.pdf');
```

## Stationary states

`src/solvers/NewtonRaphson_vy.m` solves stationary radial profiles. Its inputs are the radial grid size, horizon and endpoint parameters, three profile guesses, and the boundary flow parameter. The stationary notebook is:

```text
scripts/legacy/Statinary_state_with_v.m
```

The first section performs continuation from a stationary seed, rescales the temporal potential to the target chemical potential, calls the Newton solver, and saves `mu`, `rhoi`, `Z1`, `Z2`, and `Vy1`. Set its seed, target `mu`, and drive parameter for the profile to be generated. Later sections contain the associated thermodynamic and radial-profile calculations; the thermodynamic sections use `free_energy_1` from that calculation. Their profile tables are located under `data/processed/thermodynamics/`.

Stationary filenames use the convention:

```text
mu=<chemical potential>,nz=<radial points>,vy=<drive index>_20.mat
```

The scripts set `vy = -2*pi*drive_index/20`. The three included driven profiles use `(mu, drive index)` values of `(5,3.8)`, `(6,7)`, and `(8,13.5)`. Generate the zero-flow profile before selecting the lattice workflow.

## Time evolution

The time-evolution scripts use Fourier differentiation in periodic `x` and `y`, Chebyshev differentiation in `z`, a fixed fourth-order Runge–Kutta step, and Fourier filtering. The radial grid runs from the horizon at index `1` to the AdS boundary at index `end`.

| Script | Defaults | Outputs |
| --- | --- | --- |
| `tevol_driven.m` | `mu=6`, drive index `7`; `161×161×10`; `dt=0.02`; final time `2000` | Live plots and GIF; optional checkpoint block |
| `tevol_driven_cluster.m` | `mu=6`, drive index `7`; `301×301×20`; `dt=0.02`; final time `2000` | Checkpoints every 50 steps, plots, and progress log |
| `tevol_lattice.m` | `mu=6`, zero drive; `401×401×20`; `dt=0.02`; final time `3000` | Checkpoints every 50 steps, plots, and progress log |

Both checkpoint-producing scripts save `Z1`, `Z2`, `Z3`, `Z4`, and `rhoi` under a run folder in `data/raw/`. The lattice initialization uses an `8×8` arrangement with alternating winding `+6` and `-6`, Gaussian profile width `3`, coordinate perturbation amplitude `0.3`, and random seed `1`.

The driven initialization uses relative Gaussian noise with amplitude `1e-2`. To reproduce a particular noise realization, record and set the MATLAB random seed before running it.

Run-folder suffixes such as `_small`, `_large`, `_large_2`, and `_lattice` label individual research configurations. They are not automatic aliases: the simulation and analysis scripts must use the same folder name and grid settings.

## Checkpoint analysis

Files under `scripts/analysis/` are section-based research notebooks. Their first sections build or infer the grid, select a run folder, and read checkpoints. Subsequent sections compute the desired diagnostics or export figures.

| Script | Purpose |
| --- | --- |
| `movie_show.m` | Driven-flow fields, spectra, structure functions, and animation |
| `movie_show_large.m` | Large-domain diagnostics, temporal averaging, and vortex structure factors |
| `movie_show_decay.m` | Decay snapshots, vortex counts, spectra, and figures |
| `movie_show_lattice.m` | Lattice evolution, vortex counts, spectra, and figures |
| `movie_show_pair.m` | Vortex-pair positions, separations, and annihilation-time fits |
| `plots_cluster.m` | Processed boundary-field time series and cluster/structure-factor figures |
| `plots_for_paper.m` | Paper figures and supplementary movies from several runs |

For each analysis, set the checkpoint folder, dimensions, domain lengths, sampling interval, and time range to the matching run. Run the initialization section before sections that depend on its workspace variables. Enable the relevant commented export or saving blocks when a new output is needed.

`plots_cluster.m` writes `NofV`, `O`, `vx`, and `vy` to processed boundary-field files. The averaging sections of `movie_show_large.m` produce compact structure-function, energy-spectrum, and vortex-structure-factor arrays. Checkpoint histories are intentionally omitted from the repository; recreate the selected history for these calculations.

The earlier `scripts/legacy/statistics.m` contains checkpoint-based spectral diagnostics. It is kept outside the default MATLAB path because of its generic filename.

## Record a run

For a new dataset, record the script and repository revision, `mu`, drive index and boundary parameter, grid dimensions, domain lengths, radial endpoints, time step, final time, checkpoint interval, initialization parameters, random seed, and MATLAB/GPU environment. Match these records to the filenames in the data manifest.

The numerical kernels and formulas are retained from the research code; the repository organization standardizes locations, function help, and documentation.
