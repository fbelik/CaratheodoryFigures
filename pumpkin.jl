in_pumpkin(X) = begin
    x,y = X
    # Stem
    res = inpoly(x, y, [-0.25, -0.25, 0.25, 0.25], [0.5, 0.75, 0.75, 0.5])
    # Ellipses of body
    res |= ( (x+0.5)^2.0 / (0.5^2) + (y + 0.25)^2.0/(0.75^2) <= 1)
    res |= ( (x-0.5)^2.0 / (0.5^2) + (y + 0.25)^2.0/(0.75^2) <= 1)
    # Rectangle of body
    res |= inpoly(x, y, [-0.5, -0.5, 0.5, 0.5], [-1, 0.5, 0.5, -1])
    # Eyes
    res &= !inpoly(x, y, [-9/16, -0.3125, -1/16], [0.0, 0.4, 0.0])
    res &= !inpoly(-x, y, [-9/16, -0.3125, -1/16], [0.0, 0.4, 0.0])
    # Nose 
    res &= !inpoly(x, y, [-0.175, 0, 0.175], [-0.3825, -0.13, -0.3825])
    # Mouth
    res &= !(((x)^2/(0.7^2) + (y + 0.5)^2/(0.4^2)) <= 1 && y <= -0.5)
    # Middle tooth
    res |= inpoly(x, y, [-0.125, -0.125, 0.125, 0.125], [-0.9, -0.75, -0.75, -0.9])
    # Right and left teeth
    res |= inpoly(x, y, [0.25, 0.25, 0.5, 0.5], [-0.65, -0.5, -0.5, -0.65])
    res |= inpoly(-x, y, [0.25, 0.25, 0.5, 0.5], [-0.65, -0.5, -0.5, -0.65])
end

function pumpkin_figures(M=Int(1e8), dofigs=true; 
                         basis=besselj,
                         in_multi_index_set=p_ball(25, 1/3))
    name = "pumpkin"
    m = MonteCarloQuadrature(in_pumpkin)

    m_pruned = addPrune(m, M, basis, in_multi_index_set)

    save_mc("$(name)_pruned_$(M)_$(length(m_pruned.pts))", m_pruned)

    if dofigs
        plt1 = visualize(m, title="Pumpkin Shape")
        save("$(name)_shape.pdf", plt1)

        plt2 = visualize_multi_indices(in_multi_index_set, m.D)
        save("$(name)_cross.pdf", plt2)

        plt3 = visualize(m_pruned, markersize=10, title="Pumpkin Pruned Quadrature Rule")
        save("$(name)_pruned_$(M)_$(length(m_pruned.pts)).pdf", plt3)
    end
end