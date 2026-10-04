# Contributing

Keep source comments, function help, and documentation in English. Include the physical meaning, dimensions, units or normalization, and coordinate ordering when documenting a scientific quantity.

- Put reusable numerical routines in `src/` and experiment entry points in `scripts/`.
- Use `hqt_data_path` and `hqt_output_path` for input and output locations.
- Keep checkpoints, large arrays, movies, and temporary files out of Git.
- Preserve third-party attribution when editing or replacing upstream routines.
- Document changes to equations, boundary conditions, normalization, filters, or default parameters so they can be distinguished from maintenance changes.
- For a scientific change, report the relevant comparison with the existing calculation, along with the grid, time step, run parameters, and random seed.

The file inventory and packaging tools can be used independently of MATLAB. Update the data manifest when datasets are intentionally changed.
