# Well-balanced subcell limiting for discontinuous Galerkin discretizations of the shallow-water equations

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19913123.svg)](https://doi.org/10.5281/zenodo.19913123)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{rueda2026subcell,
  title={Well-balanced subcell limiting for discontinuous {G}alerkin discretizations of the shallow-water equations},
  author={Rueda-Ram\'{i}rez, Andr\'{e}s M and Ersing, Patrick and Winters, Andrew R and Gassner, Gregor J},
  year={2026},
  month={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{rueda2026numericalRepro,
  title={Reproducibility repository for
         "{W}ell-balanced subcell limiting for discontinuous {G}alerkin discretizations of the shallow-water equations"},
  author={Rueda-Ram\'{i}rez, Andr\'{e}s M and Ersing, Patrick and Winters, Andrew R and Gassner, Gregor J},
  year={2026},
  howpublished={\url{https://github.com/amrueda/paper_2026_well_balanced_subcell_swe}},
  doi={10.5281/zenodo.19913123}
}
```

## Abstract

High-order discontinuous Galerkin (DG) methods equipped with subcell finite-volume (FV) limiters provide an efficient framework for the simulation of nonlinear hyperbolic balance laws featuring shocks and complex flow structures.
However, for systems with non-conservative terms, the design of hybrid DG/FV schemes that simultaneously guarantee high-order accuracy for smooth solutions, robustness, and well-balancedness remains challenging.
In particular, for the shallow water equations with variable bottom topography, standard flux-differencing formulations combined with node-wise subcell limiting generally destroy the well-balanced property, even if both the underlying DG and FV discretizations are individually well-balanced.

In this work, we propose a novel flux-differencing formulation for non-conservative systems that enables node-wise subcell limiting while preserving well-balanced steady states exactly.
The key idea is to construct staggered DG fluxes whose non-conservative contributions are expressed in local-times-jump form and vanish individually at equilibrium.
To achieve this structure, we introduce a suitable reformulation of the shallow water equations in which the source term is proportional to the gradient of the total water height.
This reformulation allows the design of staggered fluxes that preserve equilibrium locally at the node level, thereby making arbitrary nodal blending with low-order FV fluxes admissible.

The resulting hybrid DG/FV method is high-order accurate, robust, and exactly well-balanced under node-wise limiting.
Numerical experiments, including challenging two-dimensional dam-break configurations with wet/dry fronts and complex obstacle interactions, demonstrate the improved stability and accuracy of the proposed approach compared to existing subcell limiting strategies.

Although this work focuses on the shallow water equations, the well-balanced hybrid DG/FV methods developed here are applicable to a broader class of nonlinear systems of balance laws, provided that certain requirements are satisfied in the discretization of the non-conservative terms.

## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.11.3.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface.

Then to install the necessary packages from the `Project.toml`
the numerical results described in the article.
Navigate in a terminal to the particular example folder of interest.
Then, you need to start Julia in this directory and follow the instructions
described below.
```bash
cd path/to/repo
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=.
```
Once instantiated that particular example set can be run.

Then, you need to start Julia from the home directory
```bash
julia --project=.
```
and follow the instructions described in the file `examples/README.md`.

## Authors

- Andrés M. Rueda-Ramírez (Universidad Politécnica de Madrid, Spain)
- Patrick Ersing (Linköping University, Sweden)
- Andrew R. Winters (Linköping University, Sweden)
- Gregor J. Gassner (University of Cologne, Germany)

## License

The code in this repository is published under the MIT license, see the
`LICENSE.md` file.

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
