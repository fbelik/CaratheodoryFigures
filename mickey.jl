include("quadrature.jl")
CairoMakie.activate!()

name = "mickey"
in_mickey(x,y) = begin
    return ((x^2 + y^2) <= 0.5^2 || 
    ((x-0.5)^2 + (y-0.5)^2) <= 0.25^2 || 
    ((x+0.5)^2 + (y-0.5)^2) <= 0.25^2 ||
    inpoly(x,y,[0,0.4,0.4],[-0.6,-0.4,-0.8]) || 
    inpoly(x,y,-1 .* [0,0.4,0.4],[-0.6,-0.4,-0.8]))
end

m = MonteCarloQuadrature(in_mickey)

plt1 = visualize(m)
save("$(name)_shape.pdf", plt1)

addPts!(m, 100000)

basis = (i,x) -> hermiteh(i,x)
hyperbolic_cross = (i,j) -> (i+1)*(j+1) <= 21

plt2 = visualize_multi_indices(hyperbolic_cross)
save("$(name)_cross.pdf", plt2)

m_pruned = prune(m, basis, hyperbolic_cross)
plt3 = visualize(m_pruned, markersize=10)
save("$(name)_pruned_$(length(m.pts))_$(length(m_pruned.pts)).pdf", plt3)
save_mc(name, m_pruned)

f(x,y) = basis(1,x) * basis(2,y)
relerr = begin
    mf = m(f)
    mpf = m_pruned(f)
    abs((mf - mpf) / mf)
end