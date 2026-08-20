# =============================================================================
# Output protection, environment capture, and provenance
# =============================================================================

using Pkg, SHA


"""Return the material output paths produced by one complete fit."""
function canonical_output_paths(output_dir::AbstractString)
    output_names = [
        "fitting_results.h5",
        "diagnostics.txt",
        "diagnostics_table.csv",
        "parameter_estimates_df.csv",
        "estimated_state_trajectories.csv",
        "estimated_state_trajectories.provenance.toml",
    ]
    return joinpath.(output_dir, output_names)
end


"""Stop before sampling rather than mixing files from different fits."""
function assert_no_existing_outputs(output_dir::AbstractString)
    existing_outputs = filter(isfile, canonical_output_paths(output_dir))
    isempty(existing_outputs) || error(
        "Refusing to overwrite an existing HGF run. Archive these files first: $(join(existing_outputs, ", "))",
    )
    return nothing
end


"""Return a SHA-256 content fingerprint for a file."""
file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end


"""
    write_run_environment(; output_dir, project_dir, intended_chains) -> String

Write the Julia, operating-system, hardware, project, manifest, and direct
package versions used by this run. The manifest remains the complete dependency
record; this file is the compact Julia analogue of an R sessionInfo() report.
"""
function write_run_environment(;
    output_dir::AbstractString,
    project_dir::AbstractString,
    intended_chains::Int,
)
    project_path = joinpath(project_dir, "Project.toml")
    manifest_path = joinpath(project_dir, "Manifest.toml")
    isfile(project_path) || error("Project.toml not found: $(project_path)")
    isfile(manifest_path) || error("Manifest.toml not found: $(manifest_path)")

    direct_packages = sort(
        [
            package.name => string(package.version)
            for package in values(Pkg.dependencies())
            if package.is_direct_dep && !isnothing(package.version)
        ];
        by = first,
    )

    environment_path = joinpath(output_dir, "run_environment.toml")
    open(environment_path, "w") do io
        println(io, "julia_version = $(repr(string(VERSION)))")
        println(io, "kernel = $(repr(string(Sys.KERNEL)))")
        println(io, "architecture = $(repr(string(Sys.ARCH)))")
        println(io, "machine = $(repr(Sys.MACHINE))")
        println(io, "cpu_threads = $(Sys.CPU_THREADS)")
        println(io, "julia_threads = $(Threads.nthreads())")
        println(io, "intended_chains = $(intended_chains)")
        println(io, "project_file = $(repr(project_path))")
        println(io, "project_sha256 = $(repr(file_sha256(project_path)))")
        println(io, "manifest_file = $(repr(manifest_path))")
        println(io, "manifest_sha256 = $(repr(file_sha256(manifest_path)))")
        println(io)
        println(io, "[direct_packages]")
        for (package_name, package_version) in direct_packages
            println(io, "$(repr(package_name)) = $(repr(package_version))")
        end
    end

    println("Run environment written to $(environment_path)")
    return environment_path
end


"""Write hashes and extraction dimensions linking trajectories to their inputs."""
function write_trajectory_provenance(;
    output_dir::AbstractString,
    project_dir::AbstractString,
    environment_path::AbstractString,
    input_path::AbstractString,
    chain_path::AbstractString,
    estimates_path::AbstractString,
    trajectory_path::AbstractString,
    samples_per_chain::Int,
    n_chains::Int,
    draws_per_cell::Int,
    n_cells::Int,
    n_rows::Int,
)
    project_path = joinpath(project_dir, "Project.toml")
    manifest_path = joinpath(project_dir, "Manifest.toml")
    provenance_path =
        joinpath(output_dir, "estimated_state_trajectories.provenance.toml")

    open(provenance_path, "w") do io
        println(io, "experiment = 2")
        println(io, "julia_version = $(repr(string(VERSION)))")
        println(io, "extraction = \"in-run observed-response replay in fitted-likelihood order\"")
        println(io, "summary = \"posterior median across all retained draws\"")
        println(io, "samples_per_chain = $(samples_per_chain)")
        println(io, "chains = $(n_chains)")
        println(io, "draws_per_cell = $(draws_per_cell)")
        println(io, "cells = $(n_cells)")
        println(io, "rows = $(n_rows)")
        println(io, "project_sha256 = $(repr(file_sha256(project_path)))")
        println(io, "manifest_sha256 = $(repr(file_sha256(manifest_path)))")
        println(io, "environment_sha256 = $(repr(file_sha256(environment_path)))")
        println(io, "input_sha256 = $(repr(file_sha256(input_path)))")
        println(io, "chains_sha256 = $(repr(file_sha256(chain_path)))")
        println(io, "parameter_estimates_sha256 = $(repr(file_sha256(estimates_path)))")
        println(io, "output_sha256 = $(repr(file_sha256(trajectory_path)))")
    end

    return provenance_path
end
