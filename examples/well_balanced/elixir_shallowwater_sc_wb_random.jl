using Trixi
using TrixiShallowWater
using Random

include("random_limiter.jl")
include("es_dissipation_term.jl")

###############################################################################
# Semidiscretization of the shallow water equations with a bottom topography function
# to test well-balancedness

equations = ShallowWaterMultiLayerEquations2D(gravity = 9.81, H0 = 0.45,
                                              rhos = (1.0))

# An initial condition with constant total water height, zero velocities and a smooth bottom 
# topography to test well-balancedness
function initial_condition_well_balanced(x, t, equations::ShallowWaterMultiLayerEquations2D)
    H = SVector(0.45)
    v1 = zero(H)
    v2 = zero(H)
    b = (((x[1])^2 + (x[2])^2) < 0.04 ?
         0.2 * (cos(4 * pi * sqrt((x[1])^2 + (x[2])^2)) + 1) : 0.0)

    return prim2cons(SVector(H..., v1..., v2..., b),
                     equations)
end

initial_condition = initial_condition_well_balanced

###############################################################################
# Get the DG approximation space

polydeg = 3
volume_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal_local_jump)
surface_flux = (FluxPlusDissipation(flux_ersing_etal, DissipationLaxFriedrichsEntropyVariables()), flux_nonconservative_ersing_etal_local_jump)

basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["h1"],)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = false)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver; boundary_conditions = boundary_condition_slip_wall)

###############################################################################
# ODE solver

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error,),
                                     analysis_polydeg = polydeg,)

stepsize_callback = StepsizeCallback(cfl = 0.7)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 1.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     extra_node_variables = (:limiting_coefficient,))

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# Define an IDP limiter that selects a random limiting factor for each
# point of the domain to test well-balancedness
stage_callbacks = (SubcellLimiterRandomIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);