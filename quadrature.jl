using CaratheodoryPruning
using CairoMakie
using LaTeXStrings
using Random
using ClassicalOrthogonalPolynomials
using Bessels
using PolygonOps
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using ProgressBars
using LaTeXStrings
Random.seed!(1234) # For reproducibility

include("lp_nnls.jl")

inpoly(x,y,xv,yv) = begin
    pt = [x,y]
    pts = [[a,b] for (a,b) in zip(xv,yv)]
    if xv[end] != xv[1] || yv[end] != yv[1]
        push!(pts, pts[1])
    end
    return inpolygon(pt, pts) >= 1
end

mutable struct MonteCarloQuadrature
    D::Int
    M::Int
    in_shape::Function
    pts::Vector{Vector{Float64}}
    w::Vector{Float64}
end

function dTV(m::MonteCarloQuadrature,n::MonteCarloQuadrature)
    @assert m.D == n.D
    m_dict = Dict(p => w for (p,w) in zip(m.pts, m.w))
    n_dict = Dict(p => w for (p,w) in zip(n.pts, n.w))
    num = 0.0
    den1 = 0.0
    den2 = 0.0
    for (pt, w) in m_dict
        den1 += abs(w)
        if pt in keys(n_dict)
            num += abs(w - n_dict[pt])
        else
            num += abs(w)
        end
    end
    for (pt, w) in n_dict
        den2 += abs(w)
        if !(pt in keys(m_dict))
            num += abs(w)
        end
    end
    return num / (den1 + den2)
end

function save_mc(name::String, m::MonteCarloQuadrature)
    name *= ".csv"
    df = DataFrame(
        [["x$(i)" => [p[i] for p in m.pts] for i in 1:m.D] ; 
         ["w" => m.w, "M" => m.M]]
    )
    CSV.write(name, df)
end

function Base.show(io::Core.IO, mime::MIME"text/plain", m::MonteCarloQuadrature)
    print(io, "$(length(m.pts)) pt quadrature rule in $(m.D)-dimensions")
end

function MonteCarloQuadrature(in_shape::Function, D=2)
    M = 0
    pts = Vector{Float64}[]
    w = Float64[]
    return MonteCarloQuadrature(D, M, in_shape, pts, w)
end

function MonteCarloQuadrature(in_shape::Function, name::String)
    df = CSV.read("$(name).csv", DataFrame)
    D = size(df, 2) - 2
    M = df[1, end]
    pts = [Vector(df[i,1:D]) for i in 1:size(df,1)]
    w = Vector(df[:,D+1])
    return MonteCarloQuadrature(D, M, in_shape, pts, w)
end

function (m::MonteCarloQuadrature)(f::Function)
    res = 0.0
    for (w,p) in zip(m.w, m.pts)
        res += w * f(p)
    end
    return res
end

function randPt(m::MonteCarloQuadrature, maxiter=10000)::Tuple{Vector{Float64},Int}
    res = rand(m.D) .* 2 .- 1
    ct = 1
    while !(m.in_shape(res))
        if ct == maxiter
            error("Could not find an interior point in $maxiter attempts")
        end
        res .= rand(m.D) .* 2 .- 1
        ct += 1
    end
    return (res, ct)
end

function hyperbolic_cross(r)
    return is -> prod(is .+ 1) <= (r+1)
end

function p_ball(r, p)
    return is -> norm(is, p) <= r
end

function total_degree(r)
    return p_ball(r, 1)
end

function build_multi_index_set(in_multi_index_set::Function, D::Int)
    multi_indices = NTuple{D,Int}[]
    inds = zeros(Int, D)
    push!(multi_indices, tuple(inds...))
    inds[1] += 1
    while true
        # Check if terminated 
        if maximum(inds) == 0
            break
        end
        if in_multi_index_set(inds)
            # Add to multi index set and iterate
            push!(multi_indices, tuple(inds...))
            inds[1] += 1
        else
            # Iterate
            i = findfirst(x -> (x>0), inds)
            inds[i] = 0
            if i < D
                inds[i+1] += 1
            end
        end        
    end
    return multi_indices
end

function index_set_size(in_multi_index_set::Function, D::Int)
    return length(build_multi_index_set(in_multi_index_set, D))
end


function vandermonde_matrix_weights(m::MonteCarloQuadrature, basis, in_multi_index_set::Function, dM::Int=0; maxiter=10000)
    multi_indices = build_multi_index_set(in_multi_index_set, m.D)
    M0 = length(m.w)
    M = M0 + dM
    D = m.D
    P = multi_indices[end][end]
    pts = m.pts
    N = length(multi_indices)
    P_alloc = zeros(D, P+1)
    # cts = Dict{Int,Int}()
    vecfun = ipt -> begin
        pt, ct = begin
            if ipt <= M0
                pts[ipt], 1
            else
                randPt(m, maxiter)
            end
        end
        # cts[ipt] = ct
        for d in 1:D
            for p in 0:P
                P_alloc[d,p+1] = basis(p, pt[d])
            end
        end
        res = ones(N)
        for (i,inds) in enumerate(multi_indices)
            for d in 1:D
                res[i] *= P_alloc[d,inds[d]+1]
            end
        end
        return VandermondeVector(res, pt)
    end
    V = OnDemandMatrix(N, M, vecfun, TV=VandermondeVector{Float64,Vector{Float64},Vector{Float64}})
    elemfun = ipt -> begin
        if ipt <= M0
            return m.w[ipt] * (M0 / M)
        end
        return 1 / M #(2^m.D) / M / cts[ipt]
    end
    w = OnDemandVector(M, elemfun)
    return V, w
end

function addPrune(m::MonteCarloQuadrature, dM::Int, basis::Function, 
                   in_multi_index_set::Function; maxiter=10000, method=:cs, progress=true, 
                   return_error=false, noise=1, kernel_kwargs...)
    D = m.D
    pts = m.pts
    M0 = m.M
    M = M0 + dM
    V, w = vandermonde_matrix_weights(m, basis, in_multi_index_set, dM; maxiter=maxiter)
    res = begin
        if method == :lp
            caratheodory_pruning_lp(Matrix(V), w; kernel_kwargs...)
       elseif method == :nnls
           caratheodory_pruning_nnls(Matrix(V), w; kernel_kwargs...)
       elseif method == :greedy
            caratheodory_pruning_greedy(Matrix(V), w; kernel_kwargs...)
       else
           caratheodory_pruning(V, w, progress=progress, return_error=return_error; kernel_kwargs...)
       end
    end
    w_pruned = res[1]
    inds = res[2]
    pts = [V.vecs[i].pt for i in inds]
    if return_error
        if (method != :lp && method != :nnls && method != :greedy)
            err = res[3]
        else
            err = norm(V*w .- V[:,inds]*w_pruned[inds])
        end
        return MonteCarloQuadrature(D, M, m.in_shape, pts, w_pruned[inds]), err
    end
    return MonteCarloQuadrature(D, M, m.in_shape, pts, w_pruned[inds])
end

function addPts!(m::MonteCarloQuadrature, dM::Int, maxiter=10000)
    M = m.M
    m.w .*= (M / (M + dM))
    for i in 1:dM
        pt, ct = randPt(m, maxiter)
        push!(m.pts, pt)
        push!(m.w, 1 / (M + dM))#(2 ^ m.D) / (M + dM) / ct)
    end
    m.M += dM
    m
end

function prune(m::MonteCarloQuadrature, basis::Function, 
                in_multi_index_set::Function; maxiter=10000, method=:cs, progress=true, 
                return_error=false, kernel_kwargs...)
    D = m.D
    pts = m.pts
    M = length(pts)
    V, w = vandermonde_matrix_weights(m, basis, in_multi_index_set, 0; maxiter=maxiter)
    res = begin
        if method == :lp
            caratheodory_pruning_lp(Matrix(V), w; kernel_kwargs...)
       elseif method == :nnls
           caratheodory_pruning_nnls(Matrix(V), w; kernel_kwargs...)
       elseif method == :greedy
            caratheodory_pruning_greedy(Matrix(V), w; kernel_kwargs...)
       else
           caratheodory_pruning(V, w, progress=progress, return_error=return_error; kernel_kwargs...)
       end
    end
    w_pruned = res[1]
    inds = res[2]
    pts = [V.vecs[i].pt for i in inds]
    if return_error
        if (method != :lp && method != :nnls && method != :greedy)
            err = res[3]
        else
            err = norm(V*w .- V[:,inds]*w_pruned[inds])
        end
        return MonteCarloQuadrature(D, M, m.in_shape, pts, w_pruned[inds]), err
    end
    return MonteCarloQuadrature(m.D, m.M, m.in_shape, pts, w_pruned[inds])
end

function visualize(m::MonteCarloQuadrature; markersize=10, title="Quadrature Rule", weight_label="Weights", crange=nothing)
    if m.D == 2
        x = [p[1] for p in m.pts]
        y = [p[2] for p in m.pts]
        col = log10.(m.w)
        minv = length(col) == 0 ? 0 : floor(Int, minimum(col))
        maxv = length(col) == 0 ? 1.0 : ceil(Int, maximum(col))
        if minv == maxv
            maxv += 1
            minv -= 1
        end
        colorrange = isnothing(crange) ? [minv, maxv] : log10.(crange)

        fig = Figure(fontsize=16);
        Axis(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]),
                       ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
                       limits=((-1,1),(-1,1)), width=400, height=400,
                       title=title)
        tks = range(-1,1,1001)
        cm = cgrad([:grey,:white])
        contourf!(fig[1,1], tks, tks, (x,y) -> m.in_shape([x,y]), colormap = cm)
        cmap = Reverse(:inferno)
        scatter!(fig[1,1], x, y; color=col, colormap = cmap, 
                 markersize=markersize, strokewidth=markersize/10, 
                 colorrange = colorrange)
        if length(col) != 0
            Colorbar(fig[1,2], ticks = (-16:16, [latexstring("10^{$i}") for i in -16:16]), 
                     colorrange = colorrange, colormap = cmap, label = weight_label)
        end
        resize_to_layout!(fig)
        fig
    elseif m.D == 3 # D == 3
        x = [p[1] for p in m.pts]
        y = [p[2] for p in m.pts]
        z = [p[3] for p in m.pts]
        col = log10.(m.w)
        minv = length(col) == 0 ? 0 : floor(Int, minimum(col))
        maxv = length(col) == 0 ? 1.0 : ceil(Int, maximum(col))
        if minv == maxv
            maxv += 1
            minv -= 1
        end
        colorrange = isnothing(crange) ? [minv, maxv] : log10.(crange)
        cmap = Reverse(:inferno)

        fig = Figure(fontsize=16);
        Axis3(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]), 
              ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
              zlabel="z", zticks=(-1:1, [L"-1",L"0",L"1"]),
              title=title,
              limits=((-1,1),(-1,1),(-1,1)),
              xgridvisible = false,
              ygridvisible = false,
              zgridvisible = false)
        scatter!(fig[1, 1], x, y, z; color=col, colormap = cmap, markersize=markersize, strokewidth=0, colorrange = colorrange)
        if length(col) != 0
            Colorbar(fig[1, 2], ticks = (-16:16, [latexstring("10^{$i}") for i in -16:16]), 
                     colorrange = colorrange, colormap = cmap, label = weight_label)
        end
        fig
    else
        error("D must be 2 or 3")
    end
end

function visualize_multi_indices(in_multi_index_set::Function, D::Int; markersize=5)
    multi_index_set = build_multi_index_set(in_multi_index_set, D)
    P = multi_index_set[end][end]
    if D == 2
        x = Int[]; y = Int[]
        for I in multi_index_set
            push!(x, I[1])
            push!(y, I[2])
        end
        fig = Figure(fontsize=16);
        ax = Axis(fig[1,1], xlabel="x", xticks=0:(max(1,floor(Int,P/10))):P,
                ylabel="y", yticks=0:(max(1,floor(Int,P/10))):P,
                title="Basis Multi-Index Set")
        scatter!(ax, x, y; markersize=10, strokewidth=0)
        fig
    else # D == 3
        x = Int[]; y = Int[]; z=Int[];
        for I in multi_index_set
            push!(x, I[1])
            push!(y, I[2])
            push!(z, I[3])
        end
        fig = Figure(fontsize=16);
        ax = Axis3(fig[1,1], xlabel="x", xticks=0:2:P,
                    ylabel="y", yticks=0:2:P,
                    zlabel="z", zticks=0:2:P,
                    title="Basis Multi-Index Set",
                    azimuth=1.275π + π)
        scatter!(ax, x, y, z; markersize=10, strokewidth=0)
        fig
    end
end