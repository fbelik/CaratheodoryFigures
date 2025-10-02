in_torus(X) = begin 
    x,y,z = X
    x0 = 0.5 * x / sqrt(x^2 + y^2)
    y0 = 0.5 * y / sqrt(x^2 + y^2)
    θ = (x == 0 ? 0.0 : atan(y / x) + π/2)
    z0 = 0.5 * sin(θ)^2 - 0.25
    R = (1 + 0.9 * sin(2θ)) * 0.125
    return sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2) <= R
end

function torus_outline!(plt)
    tks = range(-1,1,101);
    θ = 0.0 
    while θ <= 2π
        x0 = 0.5 * cos(θ - π/2)
        y0 = 0.5 * sin(θ - π/2)
        z0 = 0.5 * sin(θ)^2 - 0.25
        R = (1 + 0.9 * sin(2θ)) * 0.125
        xs = (R .* tks .+ 0.5) .* cos(θ - π/2)
        ys = (R .* tks .+ 0.5) .* sin(θ - π/2)
        zs = sqrt.(max.(0.0,R^2 .- (xs .- x0).^2 .- (ys .- y0).^2))
        lines!(plt[1,1], xs, ys,  z0 .+ zs, color=(:grey, 0.5))
        lines!(plt[1,1], xs, ys,  z0 .- zs, color=(:grey, 0.5))
        θ += R/3
    end
    plt
end

function torus_figures(M=Int(1e8), dofigs=true; 
                       basis=(i,x) -> x^i,
                       in_multi_index_set=hyperbolic_cross(11))
    name = "torus"
    m = MonteCarloQuadrature(in_torus, 3)

    m_pruned = addPrune(m, M, basis, in_multi_index_set)

    save_mc("$(name)_pruned_$(M)_$(length(m_pruned.pts))", m_pruned)

    if dofigs
        plt1 = visualize(m, title="Torus Shape")
        torus_outline!(plt1)
        save("$(name)_shape.pdf", plt1)

        plt2 = visualize_multi_indices(in_multi_index_set, m.D)
        save("$(name)_cross.pdf", plt2)

        plt3 = visualize(m_pruned, markersize=10, title="Torus Pruned Quadrature Rule")
        torus_outline!(plt3)
        save("$(name)_pruned_$(M)_$(length(m_pruned.pts)).pdf", plt3)
    end
end