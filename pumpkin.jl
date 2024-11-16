include("quadrature.jl")
CairoMakie.activate!()

name = "pumpkin"
in_pumpkin(x,y) = begin
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

m = MonteCarloQuadrature(in_pumpkin)

plt1 = visualize(m)
save("$(name)_shape.pdf", plt1)

addPts!(m, 100000)

basis = (i,x) -> besselj(i,x)
hyperbolic_cross = (i,j) -> (i^(1/3) + j^(1/3))^3 <= 31

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