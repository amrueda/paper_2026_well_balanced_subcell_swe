
using OrdinaryDiffEqSSPRK
using Trixi
using TrixiShallowWater

###############################################################################
# Semidiscretization of the multilayer shallow water equations with one layer
# to fallback to be the standard shallow water equations

equations = ShallowWaterMultiLayerEquations2D(gravity = 9.812, H0 = 0.02,
                                              threshold_desingularization = 1e-8,
                                              rhos = 1.0)

"""
    initial_condition_channel_obstacle(x, t, equations::ShallowWaterMultiLayerEquations2D)

Initial condition simulating a dam break in a channel with a small obstacle in the
downstream portion of the domain. The bottom topography (possibly) needs to be smoothed
to avoid weird artifacts as the mesh will not align with the sharp edges of the channel
bottom.

Reference for this test case uses the gate opening, that is (7.7, 1.8), as the origin
for the building coordinates, gauge points, and initial conditions.
So the initial conditions and gauge point values are appropriately shifted.

Full details on this test case are available
- S. Soares-Frazão and Y. Zech (2007)
  Experimental study of dam-break flow against an isolated obstacle
  [DOI: 10.1080/00221686.2007.9521830](https://doi.org/10.1080/00221686.2007.9521830)
"""
function initial_condition_channel_obstacle(x, t, equations::ShallowWaterMultiLayerEquations2D)
    # Initially water is at rest
    v1 = 0.0
    v2 = 0.0

    # Bottom topography is sloped near the channel walls
    x1, x2 = x

    slope2 = 0.155 / 0.34
    slope1 = -slope2

    b = zero(eltype(x))
    # slope near the bottom
    if x[2] <= 0.34
        b = slope1 * x[2] + 0.155
    end
    # slope near the top
    if x[2] >= 3.26
        b = slope2 * (x[2] - 3.26)
    end

    # Basic setup of the total water height for the dam break
    H = 0.4
    if nextfloat(x[1]) >= 7.7
        H = 0.02
    end

    # Clip the initialization to avoid negative water heights and division by zero
    h = max(equations.threshold_limiter, H - b)

    # Return the conservative variables
    return SVector(h, h * v1, h * v2, b)
end

initial_condition = initial_condition_channel_obstacle

function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                    surface_flux_functions,
                                    equations::ShallowWaterMultiLayerEquations2D)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions

    # Impulse and bottom from inside, height from external state
    u_outer = SVector(equations.threshold_limiter, u_inner[2], u_inner[3], u_inner[4])

    # calculate the boundary flux
    flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

    # Compute the nonconservative piece
    noncons_flux = nonconservative_flux_function(u_inner, u_outer, normal_direction,
                                                 equations)

    return flux, noncons_flux
end

# Define missing function needed for the bounds checking within IDP limiting
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_outflow),
                                                normal_direction::AbstractVector,
                                                mesh::P4estMesh{2},
                                                equations::ShallowWaterMultiLayerEquations2D,
                                                dg, cache, indices...)
    return SVector(equations.threshold_limiter, u_inner[2], u_inner[3], u_inner[4])
end

boundary_conditions = (; Bottom = boundary_condition_slip_wall,
                         Top = boundary_condition_slip_wall,
                         Right = boundary_condition_outflow,
                         Left = boundary_condition_slip_wall,
                         Building = boundary_condition_slip_wall)

# Manning friction source term
@inline function source_terms_manning_friction(u, x, t, equations::ShallowWaterMultiLayerEquations2D)
    # Grab the conservative variables
    h, hv_1, hv_2, _ = u

    # Desingularization
    sh = (2.0 * h) / (h^2 + max(h^2, 1e-8))

    # friction coefficient
    n = 0.01

    # Compute the common friction term
    Sf = -equations.gravity * n^2 * sh^(7.0 / 3.0) * sqrt(hv_1^2 + hv_2^2)

    return SVector(zero(eltype(x)),
                   Sf * hv_1,
                   Sf * hv_2,
                   zero(eltype(x)))
end

###############################################################################
# Get the DG approximation space

volume_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal_local_jump)
surface_flux = (FluxHydrostaticReconstruction(FluxPlusDissipation(flux_ersing_etal,
                                                                  DissipationLocalLaxFriedrichs()),
                                                                hydrostatic_reconstruction_ersing_etal),
                FluxHydrostaticReconstruction(flux_nonconservative_ersing_etal_local_jump,
                                              hydrostatic_reconstruction_ersing_etal))

polydeg = 5
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["h1"],
                                local_twosided_variables_cons = ["h1"],
                                )
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the unstructured quad mesh from a file

mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/c3d9b3f5d506101ca0e57d4725aab416/raw/13feda6c2f7b38da664f1315baea1d1a55a0d5b2/channel_obstacle.inp",
                           joinpath(@__DIR__, "channel_obstacle.inp"))

mesh = P4estMesh{2}(mesh_file)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms=source_terms_manning_friction)

###############################################################################
# ODE solver

tspan = (0.0, 30.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous initial condition

# alternative version of the initial condition used to setup a truly discontinuous water height
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the specific mesh loaded above!
function initial_condition_channel_obstacle_discontinuous(x, t, element_id,
                                                          equations::ShallowWaterMultiLayerEquations2D)
    # Initially water is at rest
    v1 = 0.0
    v2 = 0.0

    # Bottom topography is sloped near the channel walls
    x1, x2 = x

    # approximate the absolute value type channel boundaries
    m = 0.5 * 0.155 / 0.34 # half of the slope of the channel walls
    # as this value goes to zero the functions below converge (from above) to absolute value functions
    n = 1e-5
    # instead of subtracting `sqrt(2.1316)` twice, we could just subtract `2.92` once
    b = m * ((sqrt((x[2] - 0.34)^2 + n) - sqrt(2.1316))
           + (sqrt((x[2] - 3.26)^2 + n) - sqrt(2.1316)))

    # Basic setup of the total water height for the dam break
    H = 0.4
    if nextfloat(x[1], 3) >= 7.7
        H = 0.02
    end

    # Reset the element values to the left of the gate opening to be the reservoir height
    IDs = [101, 106, 312, 427] # These IDs are for the `channel_obstacle.inp` file
    if element_id in IDs
        H = 0.4
    end

    # Clip the initialization to avoid negative water heights and division by zero
    h = max(equations.threshold_limiter, H - b)

    # Return the conservative variables
    return SVector(h, h * v1, h * v2, b)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
    for j in eachnode(semi.solver), i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                       semi.solver, i, j, element)
        u_node = initial_condition_channel_obstacle_discontinuous(x_node, first(tspan), element,
                                                                  equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
    end
end

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.2,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     extra_node_variables = (:limiting_coefficient,),
                                     output_directory = joinpath(@__DIR__, "nodewise_N$polydeg"))

stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        save_solution,
                        alive_callback)

###############################################################################
# run the simulation

# Setup the stage callbacks
stage_limiter! = VelocityDesingularization()
stage_callbacks = (SubcellLimiterIDPCorrection(), stage_limiter!)

# run the simulation
sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks, adaptive = false);
