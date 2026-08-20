# =============================================================================
# Output protection and provenance
# =============================================================================
#
# R mental model
# --------------
# These are comparable to small R utilities that check paths before `saveRDS()`
# or `write.csv()` and record `digest::sha256` hashes after writing files.
#
# A SHA-256 hash is a content fingerprint. If any byte changes, the hash changes.
# The provenance record therefore links the trajectories to the exact data,
# chains, and parameter table that produced them.
# =============================================================================

using SHA


"""
    canonical_output_paths(output_dir) -> Vector{String}

Return a vector of the complete canonical output paths expected from one run.
`Vector{String}` is the Julia equivalent of an R character vector.
"""
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


"""
    assert_no_existing_outputs(output_dir) -> Nothing

Stop before fitting if any canonical output already exists. This prevents an old
trajectory CSV from being mistaken for output from a newer fit that stopped at
the diagnostics stage. `Nothing` is comparable to `invisible(NULL)` in R.
"""
function assert_no_existing_outputs(output_dir::AbstractString)
    existing_outputs = filter(isfile, canonical_output_paths(output_dir))
    isempty(existing_outputs) || error(
        "Refusing to overwrite an existing HGF run. Archive these files first: $(join(existing_outputs, ", "))",
    )
    return nothing
end


"""
    file_sha256(path) -> String

Read a file as bytes and return its SHA-256 content fingerprint as text.
"""
file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end


"""
    write_trajectory_provenance(; ...) -> String

Write a TOML text file recording the extraction method, array dimensions, and
SHA-256 hashes of all files participating in trajectory extraction. Returns the
provenance file path.
"""
function write_trajectory_provenance(;
    output_dir::AbstractString,
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
    provenance_path =
        joinpath(output_dir, "estimated_state_trajectories.provenance.toml")

    open(provenance_path, "w") do io
        println(io, "experiment = 1")
        println(io, "julia_version = \"$(VERSION)\"")
        println(io, "extraction = \"in-run observed-response replay in fitted-likelihood order\"")
        println(io, "summary = \"posterior median across all retained draws\"")
        println(io, "samples_per_chain = $(samples_per_chain)")
        println(io, "chains = $(n_chains)")
        println(io, "draws_per_cell = $(draws_per_cell)")
        println(io, "cells = $(n_cells)")
        println(io, "rows = $(n_rows)")
        println(io, "input_sha256 = \"$(file_sha256(input_path))\"")
        println(io, "chains_sha256 = \"$(file_sha256(chain_path))\"")
        println(io, "parameter_estimates_sha256 = \"$(file_sha256(estimates_path))\"")
        println(io, "output_sha256 = \"$(file_sha256(trajectory_path))\"")
    end

    return provenance_path
end
