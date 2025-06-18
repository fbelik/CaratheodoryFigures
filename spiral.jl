include("quadrature.jl")
CairoMakie.activate!()

name = "spiral"
in_spiral(X,a=1/(8π)) = begin
    x,y = X
    θ = mod2pi(atan(y, x))
    r = sqrt(x^2 + y^2)
    r1 = 0.8*a*θ
    r2 = a*θ
    res = false
    for q in 1:5
        if q < 5
            res |= (r >= (r1 + (q-1)*0.8*a*2π)) & (r <= (r2 + (q-1)*2π*a))
        else
            res |= (r >= (r1 + (q-1)*0.8*a*2π)) & (r <= ((q-1)*2π*a))
        end
    end
    return res
end

m = MonteCarloQuadrature(in_spiral)

plt1 = visualize(m, title="Spiral Shape")
save("$(name)_shape.pdf", plt1)

basis = (i,x) -> legendrep(i,x)
in_multi_index_set = (is) -> sum(is) <= 20

M = 50000
m_pruned = addPrune(m, M, basis, in_multi_index_set)

plt2 = visualize_multi_indices(in_multi_index_set, m.D)
save("$(name)_cross.pdf", plt2)

plt3 = visualize(m_pruned, markersize=10, title="Spiral Pruned Quadrature Rule")
save("$(name)_pruned_$(M)_$(length(m_pruned.pts)).pdf", plt3)
save_mc(name, m_pruned)

f = x -> x[1]
m_pruned(f)

# # To add more points
# dM = 50000
# m_pruned = addPrune(m_pruned, dM, basis, in_multi_index_set)