function (dissipation::Trixi.DissipationLaxFriedrichsEntropyVariables)(u_ll, u_rr, orientation_or_normal_direction, equations::ShallowWaterMultiLayerEquations2D{4, 1, T}) where T
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    h_avg = 0.5f0 * (u_ll[1] * u_rr[1])
    v1_avg = 0.5f0 * (u_ll[2] + u_rr[2])
    v2_avg = 0.5f0 * (u_ll[3] + u_rr[3])

    # Compute the jump in entropy variables
    w_ll = cons2entropy(u_ll, equations)
    w_rr = cons2entropy(u_rr, equations)

    # Compute the H := du/dw
    H = @SMatrix [1 v1_avg v2_avg 0;
                    v1_avg h_avg 0 0;
                    v2_avg 0 h_avg 0;
                    0 0 0 0]

    return SVector(-0.5f0 * λ / equations.gravity * H * (w_rr - w_ll))
end