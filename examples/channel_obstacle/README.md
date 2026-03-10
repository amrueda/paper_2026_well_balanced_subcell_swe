
The instructions below will assume that one uses polynomial degree `N = 3` in each spatial direction.
Identical steps can be repeated for polynomial degree `N = 5`.

To run the channel obstacle test case with element-wise limiting execute
```julia
trixi_include("elixir_shallowwater_multilayer_channel_obstacle_elementwise_limiting.jl", polydeg = 3);
```
This will generate output data into a folder named `elementwise_N3/`.
To convert the data to be used together with ParaView execute
```julia
using Trixi2Vtk
trixi2vtk("elementwise_N3/solution_*.h5", output_directory="elementwise_N3", nvisnodes=10)
```
This will reinterpolate the solution onto `10` uniformly spaced points in each direction on each element.
To visualize the solution and the element-wise limiting coefficient open ParaView and load the state file
`sol_and_elementwise_coeff.pvsm`.
Modify the file paths in the pipeline appropriately to point to the generated element-wise data.

To run the channel obstacle test case with node-wise limiting execute
```julia
trixi_include("elixir_shallowwater_multilayer_channel_obstacle_nodewise_limiting.jl", polydeg = 3);
```
This will generate output data into a folder named `nodewise_N3/`.
To convert the data to be used together with ParaView execute
```julia
using Trixi2Vtk
trixi2vtk("nodewise_N3/solution_*.h5", output_directory="nodewise_N3", nvisnodes=10)
```
This will reinterpolate the solution onto `10` uniformly spaced points in each direction on each element.
However, the limiting coefficients should not be reinterpolated, so execute the following to simply copy
over this data into an appropriate format
```julia
trixi2vtk("nodewise_N3/solution_*.h5", output_directory="limit_coeff_N3", reinterpolate=false)
```
To visualize the solution and the element-wise limiting coefficient open ParaView and load the state file
`sol_and_nodewise_coeff.pvsm`.
Modify the file paths in the pipeline appropriately to point to the generated element-wise data.


With both datasets available you can extract the gauge point data from the files using ParaView.
Open ParaView and load the state file `comparison_gauges.pvsm`.
You can then change the "ProbeLocation" filter for each data set to the desired gauge point:
```
 (10.35, 2.95) # G1
 (10.35, 1.2)  # G2
 (11.7, 2.95)  # G3
 (11.7, 1.0)   # G4
 (12.9, 2.1)   # G5
 (5.83, 2.9)   # G6
```
To export the data at a particular gauge point, click on each of the "PlotDataOverTime"
and then click on "File > Save Data..." to save `.csv` files.
The second column in each `.csv` file is the time and the third column is the water height.
These `.csv` files can then be plotted easily against the reference gauge data contained
in the file `ref_building_gauges_h.txt`.