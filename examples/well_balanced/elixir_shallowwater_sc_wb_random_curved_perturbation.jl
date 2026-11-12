using Trixi
using TrixiShallowWater
using Random
using DataFrames
using LaTeXStrings
using CairoMakie
using DelimitedFiles

include("random_limiter.jl")

###############################################################################
# Semidiscretization of the shallow water equations with a bottom topography function
# to test well-balancedness including dry states and a perturbation.
equations = ShallowWaterMultiLayerEquations2D(gravity = 9.81, H0 = 0.4503021589546197,
                                              rhos = (1.0))

# An initial condition with for a lake at rest steady state with a smooth bottom
# topography including dry states to test well-balancedness
function initial_condition_well_balanced_wet_dry(x, t, equations::ShallowWaterMultiLayerEquations2D)
    H = 0.45
    v1 = zero(H)
    v2 = zero(H)
    
    r_b = 0.4
    b = (((x[1])^2 + (x[2])^2) < r_b^2 ?
         0.25 * (cos(1/r_b * pi * sqrt((x[1])^2 + (x[2])^2)) + 1) : 0.0)

    # Add a small perturbation to the water height to test the well-balanced property with a perturbation.
    r_H = 0.2
    if (((x[1] + 0.5)^2 + (x[2])^2) < r_H^2)
        H += 1e-2
    end

    # Clip the initialization to avoid negative water heights and division by zero
    H = max(H, b + equations.threshold_limiter)

    return prim2cons(SVector(H, v1, v2, b),
                     equations)
end

initial_condition = initial_condition_well_balanced_wet_dry

###############################################################################
# Get the DG approximation space
volume_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal_local_jump)
surface_flux = (FluxHydrostaticReconstruction(FluxPlusDissipation(flux_ersing_etal,
                                                                  DissipationLocalLaxFriedrichs()),
                                                                hydrostatic_reconstruction_ersing_etal),
                FluxHydrostaticReconstruction(flux_nonconservative_ersing_etal_local_jump,
                                              hydrostatic_reconstruction_ersing_etal))
polydeg = 3
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
# Unstructured mesh with 96 cells of the square domain [-1, 1]^n
mesh_file = joinpath(@__DIR__, "square_unstructured_2.inp")

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

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error, entropy),
                                     analysis_polydeg = polydeg,
                                     save_analysis = true,
                                     output_directory = joinpath(@__DIR__, "results_perturbation"))

stepsize_callback = StepsizeCallback(cfl = 0.5)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 10.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     extra_node_variables = (:limiting_coefficient,),
                                     output_directory = joinpath(@__DIR__, "results_perturbation"))

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation
stage_limiter! = VelocityDesingularization()

# Define an IDP limiter that selects a random limiting factor for each
# point of the domain to test well-balancedness
stage_callbacks = (SubcellLimiterRandomIDPCorrection(), stage_limiter!)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);

# Time Series data
mat,head = readdlm(joinpath(@__DIR__, "results_perturbation","analysis.dat"),header=true)
df = DataFrame(mat,vec(head))

# Create time series plots for the lake at rest error
####################################################################################################
with_theme(theme_latexfonts()) do
    # Setup figure
    f = Figure(size = (800, 350))
    ax = Axis(f[1,1], xlabel = L"t", ylabel = L"\frac{1}{\Omega}\int_{\Omega} H0(x) - H(x,t) dx", yscale = log10)

    # Plot data
    lines!(ax, df.time[2:end], df.var"|H0-(h+b)|"[2:end])
    save(joinpath(@__DIR__, "results_perturbation","time_series_lake_at_rest.pdf"), f)
end

#########################################################################################
## Helper function to compute the discrete lake-at-rest level from mass conservation.

# Due to approximation errors of the initial condition, the discrete problem satisfies a slightly different lake-at-rest level 
# than the continuous problem. In this case the discrete lake-at-rest level is obtained from  solving
# f(H0) = ∫ (max(H0 - b, ϵ) dx - ∫ h(0,x) dx = 0.
# using the same LGL quadrature of the nodal values from the numerical method.
function lake_at_rest_level(semi)
    # Nodal values of the initial condition
    u0 = compute_coefficients(0.0, semi)
    # Discrete initial mass ∫ h(0,x) dx
    M0 = integrate((u, eq) -> u[1], u0, semi; normalize = false)
    # F(H) = ∫ max(H - b, ϵ) dx - ∫ h(0,x) dx and its derivative
    F(H0) = integrate((u, eq) -> max(H0 - u[end], eq.threshold_limiter), u0, semi;
                     normalize = false) - M0
    DF(H0) = integrate((u, eq) -> u[end] < H0 - eq.threshold_limiter ? 1.0 : 0.0, u0, semi;
                      normalize = false)

    # Initial guess: average total water height H = h + b in the domain
    H0 = integrate((u, eq) -> u[1] + u[end], u0, semi; normalize = true)

    # Newton iteration for the discrete lake-at-rest level
    for _ in 1:5
        H0 -= F(H0) / DF(H0)
    end
    return H0
end
