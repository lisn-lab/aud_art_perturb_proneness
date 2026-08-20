# =============================================================================
# Diagnostics for an HGF fit
# =============================================================================
#
# R mental model
# --------------
# `chains` is an MCMCChains.Chains object. Its main values behave like a named
# three-dimensional array:
#
#     retained iteration x sampled quantity x chain
#
# This is comparable to an `mcmc.list` plus sampler metadata in R. Besides the
# parameter draws, the object has named sections containing NUTS bookkeeping.
#
# Julia syntax used below
# -----------------------
# `function ... end`   defines a function without running it.
# `x; y=...`           puts optional keyword arguments after the semicolon.
# `!condition`         means logical NOT, like `!condition` in R.
# `x in collection`    asks whether collection contains x, like `%in%` in R.
# `x[:, names, :]`     selects all iterations, named columns, and all chains.
# `f.(x)`              applies f element by element, like vectorised R code.
# `return value`       ends the function and sends value back to its caller.
# =============================================================================

using CSV, DataFrames, Dates, MCMCChains, Printf


"""
    count_divergences(chains::Chains) -> Int

Count post-warm-up NUTS transitions that were marked as divergent.

Input
-----
`chains` is the fitted `MCMCChains.Chains` object. Its main dimensions are
iteration x sampled quantity x chain. NUTS diagnostic values are stored in a
separate named section called `:internals`.

Output
------
One integer: the number of divergent transitions across all retained iterations
and chains.

A divergence is not disagreement between chains. It means NUTS could not
numerically follow the posterior geometry accurately during a particular
transition. Rhat assesses agreement/mixing across chains; divergence assesses
the numerical reliability of individual HMC/NUTS transitions.
"""
function count_divergences(chains::Chains)
    # `hasproperty(object, name)` checks whether an object exposes a named field.
    # It does not load the chain or convert it into a matrix. `name_map` is the
    # table of named sections stored inside this already-loaded Chains object.
    has_internal_section =
        hasproperty(chains, :name_map) && (:internals in keys(chains.name_map))
    has_internal_section || error(
        "The fitted chain has no NUTS internals, so divergences cannot be checked.",
    )

    # `get_sections` returns another Chains object containing only the sampler's
    # internal bookkeeping rather than the fitted parameter draws.
    internals = MCMCChains.get_sections(chains, :internals)

    # `names` obtains the labelled internal columns. `Symbol.(...)` converts all
    # labels to Julia Symbols, comparable to standardising R column names.
    internal_names = Symbol.(MCMCChains.names(internals))

    # Turing 0.34.1, fixed in this project's Project.toml, records divergences as
    # `numerical_error`. The second label supports another compatible convention.
    for divergence_name in (:numerical_error, :divergent)
        if divergence_name in internal_names
            # Select every iteration, the one divergence column, and every chain.
            # Convert the labelled Chains slice to an ordinary 3D Array, mark
            # positive values as `true`, then sum `true` values as 1s.
            divergence_array = Array(internals[:, [divergence_name], :])
            return sum(divergence_array .> 0)
        end
    end

    error("The fitted chain contains no recognised divergence indicator.")
end


"""
    check_fit_diagnostics(chains::Chains; output_dir) -> NamedTuple

Calculate convergence diagnostics, write their reports, and return a compact
pass/fail result.

Input
-----
`chains` is the in-memory `Chains` object returned by `sample()`.
`output_dir` is a path string identifying where reports should be written.

Outputs
-------
The function writes two files:

- `diagnostics_table.csv`: a DataFrame-like table with one row per sampled
  quantity and columns for mean, SD, bulk ESS, Rhat, and warning flags.
- `diagnostics.txt`: a shorter report for human inspection.

It returns a NamedTuple, analogous to an R named list, containing `passed`, the
number of Rhat warnings, the number of ESS warnings, and the divergence count.
"""
function check_fit_diagnostics(chains::Chains; output_dir::AbstractString)
    # `summarize` calculates the posterior mean and SD of each sampled quantity.
    summary_table = MCMCChains.summarize(chains)

    # `ess_rhat` calculates bulk effective sample size and split Rhat. Rhat uses
    # between-chain and within-chain variation; ESS estimates the information in
    # autocorrelated draws as an equivalent number of independent draws.
    convergence = MCMCChains.ess_rhat(chains)

    # Construct a DataFrame, directly comparable to `data.frame(...)` in R.
    diagnostics = DataFrame(
        parameter = string.(summary_table.nt.parameters),
        mean = summary_table.nt.mean,
        std = summary_table.nt.std,
        ess_bulk = convergence.nt.ess,
        rhat = convergence.nt.rhat,
    )

    # The dot before each operator makes the comparison element-wise. Any
    # non-finite diagnostic is treated as a failure rather than silently ignored.
    diagnostics.rhat_warn =
        .!isfinite.(diagnostics.rhat) .| (diagnostics.rhat .> 1.01)
    diagnostics.ess_warn =
        .!isfinite.(diagnostics.ess_bulk) .| (diagnostics.ess_bulk .< 400)
    divergences = count_divergences(chains)

    table_path = joinpath(output_dir, "diagnostics_table.csv")
    report_path = joinpath(output_dir, "diagnostics.txt")
    CSV.write(table_path, diagnostics)

    # Sort copies of the table so the report shows the ten quantities requiring
    # the closest inspection. `rev=true` means descending order.
    n_to_show = min(10, nrow(diagnostics))
    worst_rhat = sort(diagnostics, :rhat, rev = true)[1:n_to_show, :]
    worst_ess = sort(diagnostics, :ess_bulk)[1:n_to_show, :]

    # `open(path, "w") do io ... end` is similar to opening an R connection,
    # writing through it, and guaranteeing that it closes when the block ends.
    open(report_path, "w") do io
        println(io, "Experiment 1 HGF fitting diagnostics")
        println(io, "Generated: $(now())")
        println(io, "Chain dimensions: $(size(chains))")
        println(io, "  iterations x sampled quantities x chains")
        println(io)
        @printf(io, "Rhat warnings: %d\n", sum(diagnostics.rhat_warn))
        @printf(io, "Bulk ESS warnings: %d\n", sum(diagnostics.ess_warn))
        @printf(io, "Divergent transitions: %d\n", divergences)
        println(io)
        println(io, "Ten highest Rhat values")
        show(io, worst_rhat; allcols = true, allrows = true, summary = false)
        println(io)
        println(io)
        println(io, "Ten lowest bulk ESS values")
        show(io, worst_ess; allcols = true, allrows = true, summary = false)
        println(io)
        println(io)
        println(io, "Full table: $(table_path)")
    end

    passed = !any(diagnostics.rhat_warn) &&
             !any(diagnostics.ess_warn) &&
             divergences == 0

    println("Diagnostics written to $(report_path) and $(table_path)")
    println("  Rhat warnings: $(sum(diagnostics.rhat_warn))")
    println("  ESS warnings: $(sum(diagnostics.ess_warn))")
    println("  divergences: $(divergences)")

    return (
        passed = passed,
        rhat_warnings = sum(diagnostics.rhat_warn),
        ess_warnings = sum(diagnostics.ess_warn),
        divergences = divergences,
    )
end
