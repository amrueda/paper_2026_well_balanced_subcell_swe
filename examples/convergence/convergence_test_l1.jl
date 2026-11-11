# Helper functions to run a convergence test that reports L1, L2, and Linf errors.
#
# Trixi.jl only computes L2 and Linf errors (`Trixi.calc_error_norms`). The function
# `calc_l1_error` below mirrors `Trixi.calc_error_norms` for curved meshes and
# evaluates the L1 error with the same analysis quadrature (interpolation to the
# analysis nodes of the `AnalysisCallback`). As for the L2 error in Trixi.jl, the result
# is normalized by the total volume of the domain.
#
# `convergence_test_l1` is a drop-in replacement for `Trixi.convergence_test` that
# additionally collects the L1 error and the maximum CFL number (`cfl_max`) of each run
# and passes all errors to `Trixi.analyze_convergence` to print the EOC tables.

using Trixi

function calc_l1_error(sol, semi, analysis_callback)
    u_ode = sol.u[end]
    t = sol.t[end]
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, mesh, equations, dg, cache)

    analyzer = analysis_callback.affect!.analyzer
    @unpack vandermonde, weights = analyzer
    @unpack u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1 = analysis_callback.affect!.cache
    @unpack node_coordinates, inverse_jacobian = cache.elements

    # Set up data structures
    l1_error = zero(Trixi.get_node_vars(u, equations, dg, 1, 1, 1))
    total_volume = zero(eltype(weights)) * zero(eltype(inverse_jacobian))

    # Iterate over all elements for error calculations
    for element in Trixi.eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        Trixi.multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, element),
                                      u_tmp1)
        Trixi.multiply_dimensionwise!(x_local, vandermonde,
                                      view(node_coordinates, :, :, :, element), x_tmp1)
        Trixi.multiply_scalar_dimensionwise!(jacobian_local, vandermonde,
                                             inv.(view(inverse_jacobian, :, :, element)),
                                             jacobian_tmp1)

        # Calculate errors at each analysis node
        for j in Trixi.eachnode(analyzer), i in Trixi.eachnode(analyzer)
            x = Trixi.get_node_coords(x_local, equations, dg, i, j)
            u_exact = semi.initial_condition(x, t, equations)
            diff = u_exact - Trixi.get_node_vars(u_local, equations, dg, i, j)
            l1_error += abs.(diff) * weights[i] * weights[j] * abs(jacobian_local[i, j])
            total_volume += weights[i] * weights[j] * abs(jacobian_local[i, j])
        end
    end

    return l1_error / total_volume
end

function convergence_test_l1(elixir::AbstractString, iterations; mod = Main, kwargs...)
    @assert iterations > 1 "Number of iterations must be bigger than 1 for a convergence analysis"

    errors = Dict(:l1 => Float64[], :l2 => Float64[], :linf => Float64[])
    cfl_numbers = Float64[]

    initial_resolution = Trixi.extract_initial_resolution(elixir, kwargs)

    for iter in 1:iterations
        println("Running convtest iteration ", iter, "/", iterations)

        Trixi.include_refined(mod, elixir, initial_resolution, iter; kwargs)

        sol = @invokelatest getproperty(mod, :sol)
        semi = @invokelatest getproperty(mod, :semi)
        analysis_callback = @invokelatest getproperty(mod, :analysis_callback)

        # @invokelatest avoids world-age issues since the elixir defines new functions
        l2_error, linf_error = @invokelatest analysis_callback(sol)
        l1_error = @invokelatest calc_l1_error(sol, semi, analysis_callback)

        append!(errors[:l1], l1_error)
        append!(errors[:l2], l2_error)
        append!(errors[:linf], linf_error)

        if @invokelatest isdefined(mod, :cfl_max)
            push!(cfl_numbers, (@invokelatest getproperty(mod, :cfl_max))[])
        end

        println("\n\n")
        println("#"^100)
    end

    eocs, errorsmatrix = @invokelatest Trixi.analyze_convergence(errors, iterations,
                                                                (@invokelatest getproperty(mod,
                                                                                           :semi)))

    if !isempty(cfl_numbers)
        println("Maximum CFL number per refinement level: ", cfl_numbers)
    end

    return eocs, errorsmatrix, cfl_numbers
end
