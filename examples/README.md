# Numerical experiments

This directory contains all source code required to reproduce the numerical experiments presented in the paper.
To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/).

Each subdirectory contains the necessary `Project.toml` and `Manifest.toml` files used to create
the numerical results described in the article.
Navigate in a terminal to the particular example folder of interest.
Then, you need to start Julia in this directory and follow the instructions
described below.
```bash
cd path/to/example
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=.
```
Once instantiated that particular example set can be run.

# Convergence test

TODO
```julia
include("elixir_shallowwater_multilayer_convergence_sc_subcell.jl");
```

# Well-balancedness test

TODO

# Circular dam break

TODO: Mention that 02 and 03 redefine `calc_fluxfhat!`

# Dam break past an oblique object

TODO
