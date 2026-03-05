####################################################################
# Define an IDP limiter that selects a random limiting factor for each
# point of the domain to test well-balancedness

"""
    SubcellLimiterRandomIDPCorrection()

Perform a random antidiffusive correction stage for the a posteriori IDP limiter [`SubcellLimiterIDP`](@ref)
called with [`VolumeIntegralSubcellLimiting`](@ref).

!!! note
    This callback and the actual limiter [`SubcellLimiterIDP`](@ref) only work together.
    This is not a replacement but a necessary addition.
"""
struct SubcellLimiterRandomIDPCorrection end

function (limiter!::SubcellLimiterRandomIDPCorrection)(u_ode,
                                                       integrator::Trixi.SimpleIntegratorSSP,
                                                       stage)
    semi = integrator.p
    limiter!(u_ode, semi, integrator.t, integrator.dt, semi.solver.volume_integral)
end

function (limiter!::SubcellLimiterRandomIDPCorrection)(u_ode, semi, t, dt,
                                                       volume_integral::VolumeIntegralSubcellLimiting)
    Trixi.@trixi_timeit Trixi.timer() "a posteriori limiter" limiter!(u_ode, semi, t, dt,
                                                                      volume_integral.limiter)
end

function (limiter!::SubcellLimiterRandomIDPCorrection)(u_ode, semi, t, dt,
                                                       limiter::SubcellLimiterIDP)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

    # Calculate blending factor alpha in [0,1]
    # f_ij = alpha_ij * f^(FV)_ij + (1 - alpha_ij) * f^(DG)_ij
    #      = f^(FV)_ij + (1 - alpha_ij) * f^(antidiffusive)_ij
    Trixi.@trixi_timeit Trixi.timer() "blending factors" solver.volume_integral.limiter(u,
                                                                                        semi,
                                                                                        solver,
                                                                                        t,
                                                                                        dt;
                                                                                        random_factors = Trixi.True())

    Trixi.perform_idp_correction!(u, dt, mesh, equations, solver, cache)

    return nothing
end

# Compute random blending factors
function (limiter::SubcellLimiterIDP)(u::AbstractArray{<:Any, 4}, semi, dg::DGSEM, t,
                                      dt;
                                      random_factors::Trixi.True,
                                      kwargs...)
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    # Introduce a fixed seed for reproducibility
    #rng = Xoshiro(Int(floor(1/dt * t)))
    rng = Xoshiro(2026)

    for element in eachelement(dg, semi.cache)
        for j in eachnode(dg), i in 1:nnodes(dg)
            alpha[i, j, element] = rand(rng)   # Set alpha as a random number in [0, 1]
        end
    end

    return nothing
end

Trixi.init_callback(limiter!::SubcellLimiterRandomIDPCorrection, semi) = nothing

Trixi.finalize_callback(limiter!::SubcellLimiterRandomIDPCorrection, semi) = nothing