
@inline function TrixiShallowWater.flux_wintermeyer_etal(u_ll, u_rr,
                                  normal_direction::AbstractVector,
                                  equations::ShallowWaterMultiLayerEquations2D)

    # Unpack left and right state
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    h_v1_ll, h_v2_ll = TrixiShallowWater.momentum(u_ll, equations)
    h_v1_rr, h_v2_rr = TrixiShallowWater.momentum(u_rr, equations)

    # Get the velocities on either side
    v1_ll, v2_ll = velocity(u_ll, equations)
    v1_rr, v2_rr = velocity(u_rr, equations)

    # Initialize flux vector
    f = zero(MVector{3 * nlayers(equations) + 1, real(equations)})

    # Calculate fluxes in each layer
    for i in eachlayer(equations)
        # Compute averages
        v1_avg = 0.5 * (v1_ll[i] + v1_rr[i])
        v2_avg = 0.5 * (v2_ll[i] + v2_rr[i])
        h_v1_avg = 0.5 * (h_v1_ll[i] + h_v1_rr[i])
        h_v2_avg = 0.5 * (h_v2_ll[i] + h_v2_rr[i])
        h_avg = 0.5 * (h_ll[i] + h_rr[i])
        h2_avg = 0.5 * (h_ll[i]^2 + h_rr[i]^2)
        p_avg = equations.gravity * h_avg^2 - 0.5f0 * equations.gravity * h2_avg

        # Compute fluxes
        f_h = h_v1_avg * normal_direction[1] + h_v2_avg * normal_direction[2]
        f_hv1 = f_h * v1_avg + p_avg * normal_direction[1]
        f_hv2 = f_h * v2_avg + p_avg * normal_direction[2]

        TrixiShallowWater.setlayer!(f, f_h, f_hv1, f_hv2, i, equations)
    end

    return SVector(f)
end

# For `VolumeIntegralSubcellLimiting` the nonconservative flux is created as a callable struct to 
# enable dispatch on the type of the nonconservative term (local / jump).
struct FluxNonConservativeWintermeyerLocalJump <:
       Trixi.FluxNonConservative{Trixi.NonConservativeJump()}
end

Trixi.n_nonconservative_terms(::FluxNonConservativeWintermeyerLocal) = 1

const flux_nonconservative_wintermeyer_etal_local_jump = FluxNonConservativeWintermeyerLocalJump()

@inline function (noncons_flux::FluxNonConservativeWintermeyerLocalJump)(u_ll, u_rr,
                                                                    normal_direction::AbstractVector,
                                                                    equations::ShallowWaterMultiLayerEquations2D)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    b_rr = u_rr[end]
    b_ll = u_ll[end]

    # Compute the jumps
    h_jump = h_rr - h_ll
    b_jump = b_rr - b_ll
    g = equations.gravity

    # Initialize flux vector
    f = zero(MVector{3 * nlayers(equations) + 1, real(equations)})

    for i in eachlayer(equations)
        f_h = zero(real(equations))
        f_hv = g * h_ll[i] * b_jump
        for j in eachlayer(equations)
            if j < i
                f_hv += g * h_ll[i] *
                        (equations.rhos[j] / equations.rhos[i] * h_jump[j])
            end
        end
        TrixiShallowWater.setlayer!(f, f_h, f_hv * normal_direction[1],
                  f_hv * normal_direction[2], i, equations)
    end

    return SVector(f)
end

# Local part
@inline function flux_nonconservative_wintermeyer_etal_local_jump(u_ll,
                                                             normal_direction::AbstractVector,
                                                             equations::ShallowWaterMultiLayerEquations2D,
                                                             nonconservative_type::Trixi.NonConservativeLocal,
                                                             nonconservative_term::Integer)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)

    g = equations.gravity

    # Initialize flux vector
    f = zero(Trixi.MVector{3 * nlayers(equations) + 1, real(equations)})

    # Compute the local part of the nonconservative flux in each layer
    # where f_hv[i] = g * h[i] * (b + ∑σ[k] * h[k])_x and σ[k] = ρ[k] / ρ[i] denotes the 
    # density ratio of different layers
    for i in eachlayer(equations)
        f_h = zero(real(equations))
        f_hv = g * h_ll[i]

        TrixiShallowWater.setlayer!(f, f_h, f_hv, f_hv, i, equations)
    end

    return SVector(f)
end

# Jump part
@inline function flux_nonconservative_wintermeyer_etal_local_jump(u_ll, u_rr,
                                                             normal_direction::AbstractVector,
                                                             equations::ShallowWaterMultiLayerEquations2D,
                                                             nonconservative_type::Trixi.NonConservativeJump,
                                                             nonconservative_term::Integer)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    b_rr = u_rr[end]
    b_ll = u_ll[end]

    # Compute the jumps
    h_jump = h_rr - h_ll
    b_jump = b_rr - b_ll
    g = equations.gravity

    # Initialize flux vector
    f = zero(MVector{3 * nlayers(equations) + 1, real(equations)})

    # Compute the jump part of the nonconservative flux in each layer
    # where f_hv[i] = g * h[i] * (b + ∑σ[k] * h[k])_x and σ[k] = ρ[k] / ρ[i] denotes the 
    # density ratio of different layers
    for i in eachlayer(equations)
        f_h = zero(real(equations))
        f_hv = b_jump
        for j in eachlayer(equations)
            if j < i
                f_hv += equations.rhos[j] / equations.rhos[i] * h_jump[j]
            end
        end
        TrixiShallowWater.setlayer!(f, f_h, f_hv * normal_direction[1],
                  f_hv * normal_direction[2], i, equations)
    end

    return SVector(f)
end