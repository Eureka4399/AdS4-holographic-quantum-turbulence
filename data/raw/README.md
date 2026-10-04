# Simulation checkpoints

This directory holds generated checkpoint folders. Checkpoints are deliberately excluded from Git. The time-evolution scripts create their run directories here when needed.

Checkpoint filenames follow `check_point_t=<time>.mat`. The default saved fields are `Z1`, `Z2`, `Z3`, `Z4`, and `rhoi`. Match the analysis folder name and time range to the run that produced them.
