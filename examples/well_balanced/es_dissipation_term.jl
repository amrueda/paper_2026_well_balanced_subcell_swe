using StaticArrays

function (dissipation::Trixi.DissipationLaxFriedrichsEntropyVariables)(u_ll, u_rr, orientation_or_normal_direction, equations::ShallowWaterMultiLayerEquations2D{4, 1, T}) where T
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    h_avg = 0.5f0 * (u_ll[1] + u_rr[1])
    v1_avg = 0.5f0 * (u_ll[2]/u_ll[1] + u_rr[2]/u_rr[1])
    v2_avg = 0.5f0 * (u_ll[3]/u_ll[1] + u_rr[3]/u_rr[1])

    g = equations.gravity

    # Compute the jump in entropy variables
    w_ll = cons2entropy(u_ll, equations)
    w_rr = cons2entropy(u_rr, equations)

    # Compute the H := du/dw
    H = @SMatrix [1 v1_avg v2_avg 0;
                  v1_avg g*h_avg+v1_avg^2 v1_avg*v2_avg 0;
                  v2_avg v1_avg*v2_avg g*h_avg+v2_avg^2 0;
                  0 0 0 0]

    return SVector(-0.5f0 * λ / g * H * (w_rr - w_ll))
end

# Test consistency for the dissipation term
# u_ll = SVector(1.0, 0.35, -0.15, 0.1)
# u_rr = SVector(0.5, 0.25, 0.55, 0.1)

# normal_direction = SVector(1.0, 0.0)

# equations = ShallowWaterMultiLayerEquations2D(gravity = 9.81, H0 = 0.45,
#                                               rhos = (1.0))

# dissipation_llf = DissipationLocalLaxFriedrichs()
# dissipation_llfev = DissipationLaxFriedrichsEntropyVariables()

# println("LLF dissipation: ", dissipation_llf(u_ll, u_rr, normal_direction, equations), " - LLF dissipation in entropy variables: ", dissipation_llfev(u_ll, u_rr, normal_direction, equations))