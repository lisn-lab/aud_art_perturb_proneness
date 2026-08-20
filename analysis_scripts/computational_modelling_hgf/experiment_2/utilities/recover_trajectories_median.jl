script_dir = @__DIR__
experiment_dir = normpath(joinpath(script_dir, ".."))
project_dir = normpath(joinpath(experiment_dir, ".."))
import Pkg
Pkg.activate(project_dir)
(VERSION.major == 1 && VERSION.minor == 12) || error(
    "This project requires Julia 1.12.x; the current runtime is $(VERSION).",
)

# Recovery-only fallback if run_model.jl saved valid chains and parameter
# estimates but its integrated trajectory-extraction stage was interrupted.
ENV["HGF_EXPERIMENT"] = "2"
include(joinpath(project_dir, "utilities", "recover_observed_trajectories.jl"))
