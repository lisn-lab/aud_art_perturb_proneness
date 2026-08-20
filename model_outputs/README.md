# Model outputs

`statistical_modelling_brms` contains the saved R model objects loaded by the
statistical reports. `computational_modelling_hgf` contains the saved HGF chains,
diagnostics, fitted input rows, parameter estimates, trajectories, and
trajectory provenance.

These files are retained in the public package because the analysis reports use
them directly. Removing them would change the reproduction task into a full,
computationally expensive model refit.
