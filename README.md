# General Descent for Multiobjective Optimization

This folder contains the Julia code used for the analytical experiments in
the paper *General descent methods for unconstrained multiobjective
optimization via dynamic scalarization*.

The code has two entry points:

- `run_main.jl`: one short, editable example;
- `run_campaign.jl`: the complete 33-problem protocol.

## Installation

Julia 1.12.6 was used for the reported campaign. From this folder, run:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Short example

Edit the six values at the top of `run_main.jl`, then run:

```bash
julia --project=. run_main.jl
```

The available methods are `:GD`, `:SD`, and `:SDNM`.

## Paper campaign

The complete protocol is stored in `paper_campaign.toml`. Check it without
starting the solvers:

```bash
julia --project=. run_campaign.jl --dry-run
```

Run a small end-to-end test:

```bash
julia --project=. run_campaign.jl --smoke
julia --project=. analyze_campaign.jl results/smoke
```

Run the complete serial campaign:

```bash
julia --project=. run_campaign.jl
julia --project=. analyze_campaign.jl
```

Completed problem--block--method cases are skipped when the command is run
again. A full campaign is long; independent subsets can be selected without
changing the configuration file:

```bash
julia --project=. run_campaign.jl --problems=AP2,AP3 --methods=GD,SD --blocks=1
```

Use a different output folder with `--output=results/my_run`.

## Experimental design

Each of the five blocks uses 300 trajectories per method. GD uses six initial
weights at each of 50 starting points. SD and SD-NM use 300 starting points.
All methods have a safeguard of 20,000 iterations. Objective scaling is off.
The complete numerical parameters are written explicitly in
`experiment_tools.jl` and `paper_campaign.toml`.


## Output and analysis

Every completed case contains:

- `archive.csv`: decision and objective vectors;
- `summary.toml`: counts and separated solver, quality-check, archive, and
  complete times.

`analyze_campaign.jl` creates block-level and mean/standard-deviation tables
for ND, time, ND/s, Gamma, and hypervolume where applicable. Hypervolume can
be expensive for large archives; use `--skip-hv` to postpone it.

## Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The tests cover all three methods, the persistent archive, and representative five- and ten-objective problems.
