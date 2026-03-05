using Trixi
using TrixiShallowWater
using Random
using StaticArrays

include("es_dissipation_term.jl")
include("random_limiter.jl")

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
        r = 0.4
    b = (((x[1])^2 + (x[2])^2) < r^2 ?
         0.2 * (cos(1/r * pi * sqrt((x[1])^2 + (x[2])^2)) + 1) : 0.0)

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
# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                           joinpath(@__DIR__, "square_unstructured_2.inp"))

# Affine type mapping to take the [-1,1]^2 domain from the mesh file
# and warp it as described in https://arxiv.org/abs/2012.12040
# Warping with the coefficient 0.15 is even more extreme.
function mapping_twist(xi, eta)
    y = eta + 0.15 * cos(1.5 * pi * xi) * cos(0.5 * pi * eta)
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos(2.0 * pi * y)
    return SVector(x, y)
end

mesh = P4estMesh{2}(mesh_file, polydeg = 3,
                    mapping = mapping_twist,
                    initial_refinement_level = 1)

# Create the semi discretization object
boundary_condition = (; all = boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_condition)

###############################################################################
# ODE solver

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (lake_at_rest_error,),
                                     analysis_polydeg = polydeg,)

stepsize_callback = StepsizeCallback(cfl = 0.5)

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