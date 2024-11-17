include("quadrature.jl")
CairoMakie.activate!()

in_circle(x,y) = (x^2 + y^2) <= 1.0

basis = (i,x) -> legendrep(i,x)
hyperbolic_cross = (i,j) -> (i+1)*(j+1) <= 31

# Construct base quadrature rule accurate w.r.t. basis
m = MonteCarloQuadrature(in_circle, 2)
addPts!(m, 100000)
m = prune(m, basis, hyperbolic_cross)
M0 = length(m.w)
plt1 = visualize(m, markersize=10, title="Unperturbed Quadrature Rule")
save("base_weights.pdf", plt1)
save_mc("perturbation_test_unperturbed", m)
# Perturb quadrature rule with small weights
dM = 10000
w_small = 1e-12
println("Unperturbed rule has sum(w) = $(sum(m.w))")
println("Perturbed rule has sum(w) = $(sum(m.w) + w_small*dM)")
println("Relative change of = $(100*(w_small*dM)/sum(m.w))%")
for i in 1:dM
    push!(m.pts, randPt(m))
    push!(m.w, w_small)
end

m_pruned_cs = prune(m, basis, hyperbolic_cross) 
plt2 = visualize(m_pruned_cs, markersize=10, title="CS Pruned Perturbed Quadrature Rule")
save("cs_pruned_weights.pdf", plt2)

m_pruned_lp = prune(m, basis, hyperbolic_cross, method=:lp)
plt3 = visualize(m_pruned_lp, markersize=10, title="LP Perturbed Quadrature Rule")
save("lp_pruned_weights.pdf", plt3)

m_pruned_nnls = prune(m, basis, hyperbolic_cross, method=:nnls)
plt4 = visualize(m_pruned_nnls, markersize=10, title="NNLS Perturbed Quadrature Rule")
save("nnls_pruned_weights.pdf", plt4)

m_cs_err = MonteCarloQuadrature(in_circle, 2)
m_cs_err.pts = m.pts[1:M0]
m_cs_err.w = zeros(M0)
for (i,p) in enumerate(m_cs_err.pts)
    idx = findfirst(x -> x==p, m_pruned_cs.pts)
    if isnothing(idx)
        m_cs_err.w[i] = 1.0
    else
        m_cs_err.w[i] = abs(m.w[i] - m_pruned_cs.w[idx]) / m.w[i]
    end
end

m_lp_err = MonteCarloQuadrature(in_circle, 2)
m_lp_err.pts = m.pts[1:M0]
m_lp_err.w = zeros(M0)
for (i,p) in enumerate(m_lp_err.pts)
    idx = findfirst(x -> x==p, m_pruned_lp.pts)
    if isnothing(idx)
        m_lp_err.w[i] = 1.0
    else
        m_lp_err.w[i] = abs(m.w[i] - m_pruned_lp.w[idx]) / m.w[i]
    end
end


m_nnls_err = MonteCarloQuadrature(in_circle, 2)
m_nnls_err.pts = m.pts[1:M0]
m_nnls_err.w = zeros(M0)
for (i,p) in enumerate(m_nnls_err.pts)
    idx = findfirst(x -> x==p, m_pruned_nnls.pts)
    if isnothing(idx)
        m_nnls_err.w[i] = 1.0
    else
        m_nnls_err.w[i] = abs(m.w[i] - m_pruned_nnls.w[idx]) / m.w[i]
    end
end

cmin = min(
    minimum(m_cs_err.w),
    minimum(m_lp_err.w),
    minimum(m_nnls_err.w)
)
cmin = 10.0 ^ floor(Int,log10(cmin))

plt5 = visualize(m_cs_err, markersize=10, title="CS Pruned Relative Errors", weight_label="Relative Error", crange=(cmin,1))
save("cs_relerr.pdf", plt5)
plt6 = visualize(m_lp_err, markersize=10, title="LP Relative Errors", weight_label="Relative Error", crange=(cmin,1))
save("lp_relerr.pdf", plt6)
plt7 = visualize(m_nnls_err, markersize=10, title="NNLS Relative Errors", weight_label="Relative Error", crange=(cmin,1))
save("nnls_relerr.pdf", plt7)