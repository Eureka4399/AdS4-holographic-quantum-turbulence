# Short runtime checks

`hqt_smoke_test.m` checks MATLAB parsing, repository paths, the numerical helpers, stationary Newton iterations, three short evolution runs, analysis initialization, and selected paper-figure sections using the included compact results.

From the repository root:

```matlab
addpath('tests');
hqt_smoke_test
```

The test requires the same GPU and toolboxes as the selected workflows. It enables GPU compatibility before loading archived `gpuArray` objects.

## No saved data

Simulation scripts are read into memory and evaluated with a `33×33×10` grid and a final time of `0.08`. The production files retain their original grid and time settings. `save`, `imwrite`, `exportgraphics`, `saveas`, `diary`, and directory creation in these in-memory copies are redirected to a discard function. Window capture is replaced by an in-memory frame.

The zero-flow stationary seed is generated in RAM. Omitted analysis checkpoints are represented by synthetic periodic fields in RAM; they are used to exercise the analysis entry points and spatial derivative setup. Paper figure sections additionally read the real `average2.mat` data. Large processed time series are never loaded.

The suite compares data/output filenames, byte counts, and modification times before and after execution. Results are printed to the MATLAB console; no report, checkpoint, figure, movie, or progress log is written.
