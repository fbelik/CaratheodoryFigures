include("quadrature.jl")
CairoMakie.activate!()

name = "spiral"
in_spiral(x,y,a=1/(8π)) = begin
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

plt1 = visualize(m)
save("$(name)_shape.pdf", plt1)

addPts!(m, 100000)

basis = (i,x) -> legendrep(i,x)
hyperbolic_cross = (i,j) -> (i + j) <= 15

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