# Data dictionary

This guide describes the variables produced and consumed by the simulation and analysis scripts. Dataset-specific settings are selected in the corresponding MATLAB sections.

## Coordinates and units

- Simulation fields use `ndgrid` ordering: `[nx, ny, nz]`, with the first dimension representing `x` and the second representing `y`.
- Radial Chebyshev points descend from the horizon at index `1` to the boundary at index `nz`; the default physical interval is `[0,1]`.
- `mu` is the boundary chemical potential in the simulation's dimensionless normalization.
- `kk` is the filename drive index; the scripts use `vy = -2*pi*kk/20` for the boundary flow parameter.
- Spatial lengths, time, and velocities use the dimensionless conventions of the model. Preserve those conventions when comparing runs.

## Stationary reference files

The stationary-state notebook writes:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `mu` | Scalar | Boundary chemical potential |
| `Z1` | `nz×1` | Stationary scalar profile |
| `Z2` | `nz×1` | Stationary temporal gauge-potential profile |
| `Vy1` | `nz×1` | Stationary spatial gauge profile passed to the initialization |
| `rhoi` | `nz×1` | Radial charge profile computed as `-Lz*Z2` |

Time-evolution initialization interpolates profiles from `nz0=30` to its chosen radial grid using a spline, then extends them across the periodic spatial domain.

## Checkpoint files

Filenames follow `check_point_t=<time>.mat` inside a run folder. `tevol_driven_cluster.m` and `tevol_lattice.m` save:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `Z1` | `nx×ny×nz` | Complex bulk scalar field |
| `Z2` | `nx×ny×nz` | Temporal gauge field reconstructed from the radial constraint |
| `Z3` | `nx×ny×nz` | Spatial gauge field in the x direction |
| `Z4` | `nx×ny×nz` | Spatial gauge field in the y direction |
| `rhoi` | `nx×ny×nz` | Charge-density integration data replicated along z |

The boundary order parameter is extracted as `func_Lz(Z1)(:,:,end)` in mathematical notation; in MATLAB the derivative is first stored in an intermediate variable. Time comes from the filename, while the grid and run parameters are set by the script. Arrays can be saved as GPU-backed MATLAB objects, depending on the workflow.

## Processed boundary-field files

The `driven_*.mat` files are produced by the checkpoint-processing sections. The corresponding save block in `plots_cluster.m` writes:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `NofV` | `1×nt` | Total vortex count per sampled frame |
| `O` | `nx×ny×nt` | Complex boundary order parameter |
| `vx`, `vy` | `nx×ny×nt` | Boundary velocity observables used by the analysis |

Here `nt` is the number of sampled checkpoints. The default extraction in `plots_cluster.m` samples every ten simulation-time units. The filename token `v=` is the drive index and should be distinguished from the saved array named `vy`.

## Averaged diagnostics

The averaging save blocks use the following names:

| Variable | Meaning |
| --- | --- |
| `kplot` | Energy-spectrum wavenumbers |
| `Ek`, `Ekc`, `Eki` | Total, compressible, and incompressible energy spectra |
| `S_Lx`, `S_Tx`, `S_Ly`, `S_Ty` | Longitudinal/transverse absolute increment moments along x/y |
| `k_bin` | Radial vortex-structure-factor bin centers |
| `Sc_1d`, `Sn_1d` | Charge and vortex-number structure factors |
| `Sc_1d_tol`, `Sn_1d_tol` | Accumulated structure-factor samples in the averaging sections |
| `vortices` | Vortex positions and charges associated with the selected field |
| `spacing_info` | Vortex density and spacing estimates |

Structure-function arrays have separation in rows and requested moment order in columns. Separations are initially in grid cells and are converted to physical lengths in plotting sections. The helper `structure_functions` computes absolute moments even for odd orders.

The structure-factor helper normalizes squared FFT magnitudes by the total number of detected vortices. `spacing_info` contains `rho`, `d_density`, `d_nnd`, `k_density`, and `k_nnd`.

## Vortex coordinates

The helpers retain the coordinate conventions used in the research scripts:

| Function | Input/output coordinate convention |
| --- | --- |
| `find_vortices` | Input field `[nx,ny]`; outputs fractional matrix indices `[row,column]` |
| `compute_vortex_structure_factor` | Input field `[Ny,Nx]`; returns physical `[x,y,charge]` |
| `find_vortex_dpc_pbc` | Reads input columns as `[y,x,charge]`; returns `[x,y]` density peaks |
| `track_vortices` | Input `[frame,particle,coordinate]`; coordinates, `L`, and `max_jump` use the same units |

Make the coordinate conversion explicitly when passing data between helpers. `countN_C` detects vortices on periodic plaquettes but computes its nearest-neighbor sign correlation using Euclidean matrix-index distances.

Vortex-pair files written by `movie_show_pair.m` contain `vortices` and `antivortices` with shape `[frame,particle,coordinate]`. Missing positions are represented by `NaN`; their conversion to physical lengths is applied by the analysis sections.

## Manifest fields

`data/metadata/manifest.json` and `manifest.csv` record relative path, original filename, byte count, SHA-256 checksum, MAT format, and distribution policy. The JSON also preserves the MAT-file header. `distribution=git` identifies the five compact files included with the code; `separate_data_archive` identifies the other local processed arrays.
