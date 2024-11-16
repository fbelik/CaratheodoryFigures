include("quadrature.jl")

name = "torus"
in_torus(x,y,z) = begin 
    x0 = 0.5 * x / sqrt(x^2 + y^2)
    y0 = 0.5 * y / sqrt(x^2 + y^2)
    θ = (x == 0 ? 0.0 : atan(y / x) + π/2)
    z0 = 0.5 * sin(θ)^2 - 0.25
    R = (1 + 0.9 * sin(2θ)) * 0.125
    return sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2) <= R
end

m = MonteCarloQuadrature(in_torus, 3)
plt1 = visualize(m, markersize=2)
θ = 0.0; ct = 0; tks = range(-1,1,101); while θ <= (2π)
    global θ, ct
    x0 = 0.5 * cos(θ - π/2)
    y0 = 0.5 * sin(θ - π/2)
    z0 = 0.5 * sin(θ)^2 - 0.25
    R = (1 + 0.9 * sin(2θ)) * 0.125
    xs = (R .* tks .+ 0.5) .* cos(θ - π/2)
    ys = (R .* tks .+ 0.5) .* sin(θ - π/2)
    zs = sqrt.(max.(0.0,R^2 .- (xs .- x0).^2 .- (ys .- y0).^2))
    lines!(plt1[1,1], xs, ys,  z0 .+ zs, color=(:grey, 0.5))
    lines!(plt1[1,1], xs, ys,  z0 .- zs, color=(:grey, 0.5))
    θ += R/3
    ct += 1
end; plt1
save("$(name)_shape.pdf", plt1)

addPts!(m, 100000)

basis = (i,x) -> x^i
hyperbolic_cross = (i,j,k) -> (i+1)*(j+1)*(k+1) <= 16

plt2 = visualize_multi_indices(hyperbolic_cross)
save("$(name)_cross.pdf", plt2)

m_pruned = prune(m, basis, hyperbolic_cross)
plt3 = visualize(m_pruned, markersize=10)
θ = 0.0; ct = 0; tks = range(-1,1,101); while θ <= (2π)
    global θ, ct
    x0 = 0.5 * cos(θ - π/2)
    y0 = 0.5 * sin(θ - π/2)
    z0 = 0.5 * sin(θ)^2 - 0.25
    R = (1 + 0.9 * sin(2θ)) * 0.125
    xs = (R .* tks .+ 0.5) .* cos(θ - π/2)
    ys = (R .* tks .+ 0.5) .* sin(θ - π/2)
    zs = sqrt.(max.(0.0,R^2 .- (xs .- x0).^2 .- (ys .- y0).^2))
    lines!(plt3[1,1], xs, ys,  z0 .+ zs, color=(:grey, 0.1))
    lines!(plt3[1,1], xs, ys,  z0 .- zs, color=(:grey, 0.1))
    θ += R/3
    ct += 1
end; plt3
save("$(name)_pruned_$(length(m.pts))_$(length(m_pruned.pts)).pdf", plt3)
save_mc(name, m_pruned)

f(x,y) = basis(1,x) * basis(2,y)
relerr = begin
    mf = m(f)
    mpf = m_pruned(f)
    abs((mf - mpf) / mf)
end