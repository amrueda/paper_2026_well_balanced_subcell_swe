# Numerical experiments

This directory contains all source code required to reproduce the numerical experiments presented in the paper.
The instructions below assume that the Julia REPL has been started in the main directory of this reproducibility repository.

All the simulations were visualized in ParaView where the HDF5 `TrixiShallowWater.jl` output
files were converted using `Trixi2Vtk.jl`.

# Convergence test

Run the full convergence test and output a table of EOCs like in the article
```julia
include Trixi
convergence_test(joinpath("examples", "convergence", "elixir_shallowwater_multilayer_convergence_sc_subcell.jl"), 4);
```

# Well-balancedness test

Generate results for the Ersing-jump formulation
```julia
include(joinpath("examples", "well_balanced", "elixir_shallowwater_sc_wb_random_curved.jl"));
```

Generate results for the Wintermeyer-jump formulation
```julia
include(joinpath("examples", "well_balanced", "elixir_shallowwater_sc_wb_random_curved_wintermeyer_jump.jl"));
```

Generate results for the Wintermeyer-symmetric formulation
```julia
include(joinpath("examples", "well_balanced", "elixir_shallowwater_sc_wb_random_curved_wintermeyer_symmetric.jl"));
```

After a run, use a command like the following to create output `.vtu` files
```julia
trixi2vtk(joinpath("examples", "well_balanced", "results_ersing_jump", "solution_*.h5"), output_directory=joinpath("examples", "well_balanced","results_ersing_jump"), reinterpolate=false)
```

- To create the left and right parts of Figure 2 start Paraview and load the statefiles `examples/well_balanced/setup_wb_mesh.psvm` and `examples/well_balanced/setup_wb_alphas.pvsm`, respectively.

- To create Figure 3(a) run the Ersing-jump version, convert the HDF5 files, start Paraview, and load the statefile `examples/well_balanced/setup_wb_plot.pvsm`
- To create Figure 3(b) run the Wintermeyer-jump version, convert the HDF5 files, start Paraview, and load the statefile `examples/well_balanced/setup_wb_plot.pvsm`
- To create Figure 3(c) run the Wintermeyer-symmetric version, convert the HDF5 files, start Paraview, and load the statefile `examples/well_balanced/setup_wb_plot.pvsm`

# Circular dam break

The circular dam break that generates the result in Figure 4(a)
```julia
include(joinpath("examples", "circular_dam_break", "01_new_formula", "elixir_shallowwater_multilayer_dam_break.jl"));
```

The circular dam break that generates the result in Figure 4(b)
```julia
include(joinpath("examples", "circular_dam_break", "02_nonsymmetric", "elixir_shallowwater_multilayer_dam_break.jl"));
```

The circular dam break that generates the result in Figure 4(c)
```julia
include(joinpath("examples", "circular_dam_break", "03_symmetrized", "elixir_shallowwater_multilayer_dam_break.jl"));
```

> [!WARNING]
> If one runs either the `examples/circular_dam_break/02_nonsymmetric` or `examples/circular_dam_break/03_nonsymmetric` test cases, the Julia session should be ended and restated.
> This is because these elixirs redefine the function `calcflux_fhat!`.

After a run, use a command like the following to create output `.vtu` files
```julia
trixi2vtk(joinpath("examples", "circular_dam_break", "01_new_formula", "out", "solution_*.h5"), output_directory=joinpath("examples", "circular_dam_break", "01_new_formula", "out"), reinterpolate=false)
```
To create Figure 4(a) start Paraview and load the statefile `examples/circular_dam_break/State_3D.pvsm`

# Dam break past an oblique object

To run the dam break flow past an obstacle with elementwise limiting use
```julia
include(joinpath("examples", "channel_obstacle", "elixir_shallowwater_multilayer_channel_obstacle_elementwise_limiting.jl"));
```

To run the dam break flow past an obstacle with nodewise limiting use
```julia
include(joinpath("examples", "channel_obstacle", "elixir_shallowwater_multilayer_channel_obstacle_nodewise_limiting.jl"));
```

By default either elixir will use a polynomial of degree five. To change this, one can do, e.g.,
```julia
using Trixi
trixi_include(joinpath("examples", "channel_obstacle", "elixir_shallowwater_multilayer_channel_obstacle_nodewise_limiting.jl"), polydeg = 3);
```

A run generates an out folder containing the polynomial degree, e.g., `examples/channel_obstacle/nodewise_N5`.

- After running the element-wise version, use the following to create output `.vtu` files for the limiting coefficients
  ```julia
  trixi2vtk(joinpath("examples", "channel_obstacle", "elementwise_N5", "solution_*.h5"), output_directory=joinpath  ("examples", "channel_obstacle", "elementwise_N5"), nvisnodes=10)
  ```
  Then to create Figures 6(a) and 7(a) start Paraview and load the statefile `sol_and_elementwise_coeff.pvsm`

- After running the node-wise version, use the following to create output `.vtu` files for the limiting coefficients
  ```julia
  trixi2vtk(joinpath("examples", "channel_obstacle", "nodewise_N5", "solution_*.h5"), output_directory=joinpath  ("examples", "channel_obstacle", "limit_coeff_N5"), reinterpolate=false)
  ```
  and to reinterpolate the solution quantities onto uniform points
  ```julia
  trixi2vtk(joinpath("examples", "channel_obstacle", "nodewise_N5", "solution_*.h5"), output_directory=joinpath  ("examples", "channel_obstacle", "nodewise_N5"), nvisnodes=10)
  ```
  Then to create Figures 6(b) and 7(b) start Paraview and load the statefile `sol_and_nodewise_coeff.pvsm`

- To create the gauge comparisons in Figure 8 start Paraview and load the statefile `comparison_gauges.pvsm`.
  Only one gauge point is plotted at a time and the user must manually change the sample point to one of those listed in Table 2.