# =============================================================================
# Diagnostics for the Experiment 2 HGF fit
# =============================================================================

using CSV, DataFrames, Dates, MCMCChains, Printf, Statistics


"""
    count_divergences(posterior_chains::Chains) -> Int

Count post-warm-up NUTS transitions marked as divergent.
"""
function count_divergences(posterior_chains::Chains)
    return sum(divergence_mask(posterior_chains))
end


"""Return the post-warm-up divergence indicator as iteration x chain."""
function divergence_mask(posterior_chains::Chains)
    has_internal_section =
        hasproperty(posterior_chains, :name_map) &&
        (:internals in keys(posterior_chains.name_map))
    has_internal_section || error(
        "The fitted chain has no NUTS internals, so divergences cannot be checked.",
    )

    internals = MCMCChains.get_sections(posterior_chains, :internals)
    internal_names = Symbol.(MCMCChains.names(internals))
    divergence_name = if :numerical_error in internal_names
        :numerical_error
    elseif :divergent in internal_names
        :divergent
    else
        error("The fitted chain contains no recognised divergence indicator.")
    end
    return reshape(
        Array(internals[:, [divergence_name], :]) .> 0,
        size(posterior_chains, 1),
        size(posterior_chains, 3),
    )
end


"""Return one sampler-internal quantity as iteration x chain."""
function sampler_internal(posterior_chains::Chains, name::Symbol)
    internals = MCMCChains.get_sections(posterior_chains, :internals)
    name in Symbol.(MCMCChains.names(internals)) || error(
        "The fitted chain has no sampler internal named $(name).",
    )
    return reshape(
        Array(internals[:, [name], :]),
        size(posterior_chains, 1),
        size(posterior_chains, 3),
    )
end


"""
    check_fit_diagnostics(posterior_chains::Chains; output_dir,
                          max_divergence_fraction=0.0) -> NamedTuple

Write rank-normalised Rhat, bulk and tail effective sample sizes, energy mixing,
and divergence reports. By default, any divergence fails the gate. A nonzero
`max_divergence_fraction` permits an explicitly documented "accepted with
warning" result when all other diagnostics pass.
"""
function check_fit_diagnostics(
    posterior_chains::Chains;
    output_dir::AbstractString,
    max_divergence_fraction::Real = 0.0,
)
    0 <= max_divergence_fraction <= 1 || error(
        "max_divergence_fraction must be between 0 and 1.",
    )
    summary_table = MCMCChains.summarystats(posterior_chains)

    diagnostics = DataFrame(
        parameter = string.(summary_table.nt.parameters),
        mean = summary_table.nt.mean,
        std = summary_table.nt.std,
        ess_bulk = summary_table.nt.ess_bulk,
        ess_tail = summary_table.nt.ess_tail,
        rhat = summary_table.nt.rhat,
    )
    diagnostics.rhat_warn =
        .!isfinite.(diagnostics.rhat) .| (diagnostics.rhat .> 1.01)
    diagnostics.ess_bulk_warn =
        .!isfinite.(diagnostics.ess_bulk) .| (diagnostics.ess_bulk .< 400)
    diagnostics.ess_tail_warn =
        .!isfinite.(diagnostics.ess_tail) .| (diagnostics.ess_tail .< 400)
    diagnostics.ess_warn =
        diagnostics.ess_bulk_warn .| diagnostics.ess_tail_warn

    divergence_flags = divergence_mask(posterior_chains)
    divergences = sum(divergence_flags)
    total_transitions = length(divergence_flags)
    divergence_fraction = divergences / total_transitions
    divergences_by_chain = vec(sum(divergence_flags; dims = 1))

    energy = sampler_internal(posterior_chains, :hamiltonian_energy)
    ebfmi = [
        mean(diff(energy[:, chain]).^2) / var(energy[:, chain])
        for chain in axes(energy, 2)
    ]
    tree_depth = sampler_internal(posterior_chains, :tree_depth)
    max_tree_depth = maximum(tree_depth)
    max_energy_error = sampler_internal(
        posterior_chains,
        :max_hamiltonian_energy_error,
    )

    table_path = joinpath(output_dir, "diagnostics_table.csv")
    report_path = joinpath(output_dir, "diagnostics.txt")
    CSV.write(table_path, diagnostics)

    n_to_show = min(10, nrow(diagnostics))
    worst_rhat = sort(diagnostics, :rhat, rev = true)[1:n_to_show, :]
    worst_bulk_ess = sort(diagnostics, :ess_bulk)[1:n_to_show, :]
    worst_tail_ess = sort(diagnostics, :ess_tail)[1:n_to_show, :]

    convergence_passed =
        !any(diagnostics.rhat_warn) && !any(diagnostics.ess_warn)
    divergence_tolerance_met =
        divergence_fraction <= max_divergence_fraction
    clean_pass = convergence_passed && divergences == 0
    accepted_with_warning = convergence_passed &&
                            divergences > 0 &&
                            divergence_tolerance_met
    passed = clean_pass || accepted_with_warning
    status = clean_pass ? "clean pass" :
             accepted_with_warning ? "accepted with divergence warning" :
             "failed"

    open(report_path, "w") do io
        println(io, "Experiment 2 HGF fitting diagnostics")
        println(io, "Generated: $(now())")
        println(io, "Chain dimensions: $(size(posterior_chains))")
        println(io, "  iterations x sampled quantities x chains")
        println(io)
        @printf(io, "Rhat warnings: %d\n", sum(diagnostics.rhat_warn))
        @printf(io, "Bulk ESS warnings: %d\n", sum(diagnostics.ess_bulk_warn))
        @printf(io, "Tail ESS warnings: %d\n", sum(diagnostics.ess_tail_warn))
        @printf(io, "Divergent transitions: %d / %d (%.3f%%)\n",
                divergences, total_transitions, 100 * divergence_fraction)
        @printf(io, "Divergences by chain: %s\n", join(divergences_by_chain, ", "))
        @printf(io, "Permitted divergence fraction: %.3f%%\n",
                100 * max_divergence_fraction)
        @printf(io, "Diagnostic gate status: %s\n", status)
        @printf(io, "E-BFMI by chain: %s\n",
                join([@sprintf("%.3f", value) for value in ebfmi], ", "))
        @printf(io, "Maximum observed tree depth: %.0f\n", max_tree_depth)
        @printf(io, "Maximum trajectory energy error: %s\n",
                string(maximum(max_energy_error)))
        println(io)
        println(io, "Ten highest Rhat values")
        show(io, worst_rhat; allcols = true, allrows = true, summary = false)
        println(io)
        println(io)
        println(io, "Ten lowest bulk ESS values")
        show(io, worst_bulk_ess; allcols = true, allrows = true, summary = false)
        println(io)
        println(io)
        println(io, "Ten lowest tail ESS values")
        show(io, worst_tail_ess; allcols = true, allrows = true, summary = false)
        println(io)
        println(io)
        println(io, "Full table: $(table_path)")
    end

    println("Diagnostics written to $(report_path) and $(table_path)")
    println("  Rhat warnings: $(sum(diagnostics.rhat_warn))")
    println("  bulk ESS warnings: $(sum(diagnostics.ess_bulk_warn))")
    println("  tail ESS warnings: $(sum(diagnostics.ess_tail_warn))")
    println("  divergences: $(divergences) / $(total_transitions) ($(round(100 * divergence_fraction; digits = 3))%)")
    println("  gate status: $(status)")

    return (
        passed = passed,
        clean_pass = clean_pass,
        accepted_with_warning = accepted_with_warning,
        status = status,
        rhat_warnings = sum(diagnostics.rhat_warn),
        ess_bulk_warnings = sum(diagnostics.ess_bulk_warn),
        ess_tail_warnings = sum(diagnostics.ess_tail_warn),
        divergences = divergences,
        total_transitions = total_transitions,
        divergence_fraction = divergence_fraction,
        max_divergence_fraction = Float64(max_divergence_fraction),
        divergences_by_chain = divergences_by_chain,
        max_rhat = maximum(diagnostics.rhat),
        min_bulk_ess = minimum(diagnostics.ess_bulk),
        min_tail_ess = minimum(diagnostics.ess_tail),
        ebfmi = ebfmi,
        max_tree_depth = max_tree_depth,
        max_energy_error = maximum(max_energy_error),
    )
end
