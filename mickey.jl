include("quadrature.jl")
CairoMakie.activate!()

name = "mickey"
in_mickey(X) = begin
    x,y = X
    return ((x^2 + y^2) <= 0.5^2 || 
    ((x-0.5)^2 + (y-0.5)^2) <= 0.25^2 || 
    ((x+0.5)^2 + (y-0.5)^2) <= 0.25^2 ||
    inpoly(x,y,[0,0.4,0.4],[-0.6,-0.4,-0.8]) || 
    inpoly(x,y,-1 .* [0,0.4,0.4],[-0.6,-0.4,-0.8]))
end

m = MonteCarloQuadrature(in_mickey)

plt1 = visualize(m, title="Mickey Shape")
save("$(name)_shape.pdf", plt1)

basis = (i,x) -> hermiteh(i,x)
in_multi_index_set = (is) -> prod(is .+ 1) <= 41

M = 50000
m_pruned = addPrune(m, M, basis, in_multi_index_set)

plt2 = visualize_multi_indices(in_multi_index_set, m.D)
save("$(name)_cross.pdf", plt2)

plt3 = visualize(m_pruned, markersize=10, title="Mickey Pruned Quadrature Rule")
save("$(name)_pruned_$(M)_$(length(m_pruned.pts)).pdf", plt3)
save_mc(name, m_pruned)

f = x -> x[1]
m_pruned(f)

# # To add more points
# dM = 50000
# m_pruned = addPrune(m_pruned, dM, basis, in_multi_index_set)