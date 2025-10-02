in_mickey(X) = begin
    x,y = X
    return ((x^2 + y^2) <= 0.5^2 || 
    ((x-0.5)^2 + (y-0.5)^2) <= 0.25^2 || 
    ((x+0.5)^2 + (y-0.5)^2) <= 0.25^2 ||
    inpoly(x,y,[0,0.4,0.4],[-0.6,-0.4,-0.8]) || 
    inpoly(x,y,-1 .* [0,0.4,0.4],[-0.6,-0.4,-0.8]))
end

function mickey_figures(M=Int(1e8), dofigs=true; 
                         basis=hermiteh,
                         in_multi_index_set=hyperbolic_cross(20))
    name = "mickey"
    m = MonteCarloQuadrature(in_mickey)

    m_pruned = addPrune(m, M, basis, in_multi_index_set)

    save_mc("$(name)_pruned_$(M)_$(length(m_pruned.pts))", m_pruned)

    if dofigs
        plt1 = visualize(m, title="Mickey Shape")
        save("$(name)_shape.pdf", plt1)

        plt2 = visualize_multi_indices(in_multi_index_set, m.D)
        save("$(name)_cross.pdf", plt2)

        plt3 = visualize(m_pruned, markersize=10, title="Mickey Pruned Quadrature Rule")
        save("$(name)_pruned_$(M)_$(length(m_pruned.pts)).pdf", plt3)
    end
end