using CaratheodoryPruning
using CairoMakie
using GLMakie
using LaTeXStrings
using Random
using ClassicalOrthogonalPolynomials
using Bessels
using PolygonOps
using CSV
using DataFrames
#using Makie.GeometryBasics
Random.seed!(1234) # For reproducibility

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
    if !(D in (2,3))
        error("Must set D=2 or D=3")
    end
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
        res += w * f(p...)
    end
    return res
end

function randPt(m::MonteCarloQuadrature, maxiter=10000)
    res = rand(m.D) .* 2 .- 1
    ct = 1
    while !(m.in_shape(res...))
        if ct == maxiter
            error("Could not find an interior point in $maxiter attempts")
        end
        res .= rand(m.D) .* 2 .- 1
        ct += 1
    end
    return res
end

function addPts!(m::MonteCarloQuadrature, dM::Int, maxiter=1000)
    M = m.M
    m.w .*= (M / (M + dM))
    for i in 1:dM
        push!(m.pts, randPt(m, maxiter))
        push!(m.w, 1 / (M + dM))
    end
    m.M += dM
    m
end

function prune(m::MonteCarloQuadrature, basis::Function, in_poly_cross::Function)
    D = m.D
    pts = m.pts
    M = length(pts)
    P = 0
    multi_indices = Vector{Int}[]
    while in_poly_cross(P+1, zeros(D-1)...)
        P += 1
    end
    if D == 2
        for i in 0:P
            for j in 0:P
                if in_poly_cross(i, j)
                    push!(multi_indices, [i,j])
                else
                    break
                end
            end
        end
    else # D == 3
        for i in 0:P
            for j in 0:P
                for k in 0:P
                    if in_poly_cross(i, j, k)
                        push!(multi_indices, [i,j,k])
                    else
                        break
                    end
                end
            end
        end
    end
    N = length(multi_indices)
    if M < N 
        println("No pruning required, returning original rule")
        return m
    end

    P1 = zeros(P+1)
    P2 = zeros(P+1)
    if D == 2
        vecfun = ipt -> begin
            pt = pts[ipt]
            x = pt[1]
            y = pt[2]
            for i in 0:P
                P1[i+1] = basis(i, x)
            end
            if x == y
                P2 .= P1
            else
                for j in 0:P
                    P2[j+1] = basis(j, y)
                end
            end
            res = zeros(N)
            for idx in 1:N
                i = multi_indices[idx][1]
                j = multi_indices[idx][2]
                res[idx] = P1[i+1] * P2[j+1]
            end
            return res
        end
    else # D == 3
        P3 = zeros(P+1)
        vecfun = ipt -> begin
            pt = pts[ipt]
            x = pt[1]
            y = pt[2]
            z = pt[3]
            for i in 0:P
                P1[i+1] = basis(i, x)
            end
            if x == y
                P2 .= P1
            else
                for j in 0:P
                    P2[j+1] = basis(j, y)
                end
            end
            if x == z
                P3 .= P1
            elseif y == z
                P3 .= P2
            else
                for k in 0:P
                    P3[k+1] = basis(k, z)
                end
            end
            res = zeros(N)
            for idx in 1:N
                i = multi_indices[idx][1]
                j = multi_indices[idx][2]
                k = multi_indices[idx][3]
                res[idx] = P1[i+1] * P2[j+1] * P3[k+1]
            end
            return res
        end
    end
    V = OnDemandMatrix(N, M, vecfun, by=:cols)
    w = m.w
    w_pruned,inds = caratheodory_pruning(V, w, progress=true)
    return MonteCarloQuadrature(m.D, m.M, m.in_shape, pts[inds], w_pruned[inds])
end

function visualize(m::MonteCarloQuadrature; markersize=5)
    if m.D == 2
        x = [p[1] for p in m.pts]
        y = [p[2] for p in m.pts]
        col = log10.(m.w)
        minv = length(col) == 0 ? 0 : floor(Int, minimum(col))
        maxv = length(col) == 0 ? 1.0 : ceil(Int, maximum(col))
        if minv == maxv
            maxv += 1
        end
        colorrange = [minv, maxv]

        fig = Figure();
        Axis(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]),
                       ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
                       limits=((-1,1),(-1,1)), width=400, height=400)
        tks = range(-1,1,1001)
        cm = cgrad([:grey,:white])
        contourf!(fig[1,1], tks, tks, m.in_shape, colormap = cm)
        cmap = Reverse(:inferno)
        scatter!(fig[1,1], x, y; color=col, colormap = cmap, markersize=markersize, strokewidth=0, colorrange = colorrange)
        if length(col) != 0
            Colorbar(fig[1,2], ticks = (minv:maxv, [latexstring("10^{$i}") for i in minv:maxv]), colorrange = colorrange, colormap = cmap, label = "Weights")
        end
        resize_to_layout!(fig)
        fig
    else # D == 3
        x = [p[1] for p in m.pts]
        y = [p[2] for p in m.pts]
        z = [p[3] for p in m.pts]
        col = log10.(m.w)
        minv = length(col) == 0 ? 0 : floor(Int, minimum(col))
        maxv = length(col) == 0 ? 1.0 : ceil(Int, maximum(col))
        if minv == maxv
            maxv += 1
        end
        colorrange = [minv, maxv]
        cmap = Reverse(:inferno)

        fig = Figure();
        Axis3(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]), 
              ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
              zlabel="z", zticks=(-1:1, [L"-1",L"0",L"1"]),
              limits=((-1,1),(-1,1),(-1,1)),
              xgridvisible = false,
              ygridvisible = false,
              zgridvisible = false)
        scatter!(fig[1, 1], x, y, z; color=col, colormap = cmap, markersize=markersize, strokewidth=0, colorrange = colorrange)
        if length(col) != 0
            Colorbar(fig[1, 2], ticks = (minv:maxv, [latexstring("10^{$i}") for i in minv:maxv]), colorrange = colorrange, colormap = cmap, label = "Weights")
        end
        fig
    end
end

function visualize_multi_indices(poly_cross::Function; markersize=5)
    D = begin
        try
            poly_cross(0, 0)
            2
        catch
            3
        end
    end
    
    if D == 2
        P = 0
        while poly_cross(P+1, 0)
            P += 1
        end
        x = Int[]; y = Int[]
        for i in 0:P
            for j in 0:P
                if poly_cross(i,j)
                    push!(x, i)
                    push!(y, j)
                else
                    break
                end
            end
        end
        fig = Figure();
        ax = Axis(fig[1,1], xlabel="x", xticks=0:P,
                ylabel="y", yticks=0:P,
                title="Basis Multi-Index Set")
        scatter!(ax, x, y; markersize=10, strokewidth=0)
        fig
    else # D == 3
        P = 0
        while poly_cross(P+1, 0, 0)
            P += 1
        end
        x = Int[]; y = Int[]; z=Int[];
        for i in 0:P
            for j in 0:P
                for k in 0:P
                    if poly_cross(i,j,k)
                        push!(x, i)
                        push!(y, j)
                        push!(z, k)
                    else
                        break
                    end
                end
            end
        end
        fig = Figure();
        ax = Axis3(fig[1,1], xlabel="x", xticks=0:P,
                    ylabel="y", yticks=0:P,
                    zlabel="z", zticks=0:P,
                    title="Basis Multi-Index Set")
        scatter!(ax, x, y, z; markersize=10, strokewidth=0)
        fig
    end
end