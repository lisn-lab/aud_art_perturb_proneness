# Statistical modelling

Each experiment has one R Markdown analysis report:

- `experiment_1/analysis.Rmd`
- `experiment_2/analysis.Rmd`

Start at the project root and follow the `renv::restore()` instructions in the
root `README.md`. The reports load the included `brms` model objects, regenerate
the exported tables and manuscript figures, write an HTML report beside each
R Markdown file, and record `sessionInfo()` at the end.

The fitting chunks are retained for transparency but have `eval=FALSE` because
refitting is computationally expensive and is not required to reproduce the
reported summaries from the saved models.
