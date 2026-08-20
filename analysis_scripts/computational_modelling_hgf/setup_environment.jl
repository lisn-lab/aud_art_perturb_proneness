# =============================================================================
# Prepare the pinned HGF environment on a new computer
# =============================================================================
#
# Run once after copying the project to macOS or Linux:
#
#     julia +1.12 setup_environment.jl
#
# Pkg.instantiate installs the exact package versions recorded in Manifest.toml
# and selects binary artifacts for the current operating system. It does not
# update or re-resolve the dependency graph.
# =============================================================================

project_dir = @__DIR__
import Pkg
Pkg.activate(project_dir)

(VERSION.major == 1 && VERSION.minor == 12) || error(
    "This project requires Julia 1.12.x; the current runtime is $(VERSION).",
)

println("Preparing HGF environment")
println("  Julia: $(VERSION)")
println("  kernel: $(Sys.KERNEL)")
println("  architecture: $(Sys.ARCH)")
println("  project: $(project_dir)")

Pkg.instantiate()
Pkg.precompile()

println()
println("Direct project dependencies")
Pkg.status(; mode = Pkg.PKGMODE_PROJECT)
println()
println("Environment preparation completed successfully")
