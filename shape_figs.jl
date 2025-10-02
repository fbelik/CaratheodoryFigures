include("quadrature.jl")
CairoMakie.activate!()

for shape in ("mickey", "pumpkin", "spiral", "torus")
    include("$(shape).jl")
end

if !isdefined(Main, :M)
    M = 1000 # Can increase to Int(1e9) to replicate figs from paper
end
dofigs = true

mickey_figures(M, dofigs)
pumpkin_figures(M, dofigs)
spiral_figures(M, dofigs)
torus_figures(M, dofigs)