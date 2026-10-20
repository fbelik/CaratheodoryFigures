# Convergence study for the cut-cell DG advection-diffusion solver from advection_diffusion.jl.
#
# Manufactured solution (2D analogue of Trixi's elixir_advection_diffusion.jl convergence test):
#   u(x, y, t) = 1 + 0.5 sin(π(x + y − (aₓ + a_y) t)) exp(−2 ε π² t)
# with constant advection velocity a = (1.5, 1.0) and diffusivity ε = 0.01 on [-1, 1]² minus two
# circles. The advective part uses an upwind flux with u_exact imposed at inflow boundary nodes
# (a·n < 0); the diffusive (BR1) part imposes u_exact on the entire boundary. Forcing is zero.
#
# Run with `julia --project=. advection_diffusion_convergence.jl` from this directory, or `include`
# it in a REPL. Afterwards, additional cases can be added interactively, e.g.
#   push!(results[4], run_case(4, 64)); plot_convergence(results)
# (K = 64 has cut cells with volume fraction ~1e-6 and would need state redistribution.)

using StartUpDG
using OrdinaryDiffEqTsit5
using LinearAlgebra
using Printf
using Plots
using LaTeXStrings

# ---------------------------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------------------------
const a_vel   = (1.5, 1.0)   # constant advection velocity
const epsilon = 0.01         # diffusivity
const T_final = 1.0

u_exact(x, y, t) = 1 + 0.5 * sin(pi * (x + y - (a_vel[1] + a_vel[2]) * t)) * exp(-2 * epsilon * pi^2 * t)

const R0 = 0.425
circles = (PresetGeometries.Circle(R=R0, x0=0.5, y0=-0.5),
           PresetGeometries.Circle(R=R0, x0=-0.5, y0=0.5))

# polynomial degrees and cells per dimension for the h-refinement study
P_list = 1:2#1:4
K_list = [4,8]#[4, 8, 16, 32]

# ---------------------------------------------------------------------------------------------
# interpolation helpers 
# ---------------------------------------------------------------------------------------------
function interp_to_face_nodes(u, rd, md)
    uM = similar(md.xf)
    uM.cartesian .= rd.Vf * u.cartesian
    for e in 1:size(md.x.cut, 2)
        ids = md.mesh_type.cut_face_nodes[e]
        Vf = md.mesh_type.cut_cell_operators.face_interpolation_matrices[e]
        uM.cut[ids] .= Vf * u.cut[:, e]
    end
    return uM
end

function interp_to_volume_nodes(u, rd, md)
    uq = similar(md.xq)
    uq.cartesian .= rd.Vq * u.cartesian
    for e in 1:size(md.x.cut, 2)
        Vq = md.mesh_type.cut_cell_operators.volume_interpolation_matrices[e]
        uq.cut[:, e] .= Vq * u.cut[:, e]
    end
    return uq
end

# ---------------------------------------------------------------------------------------------
# discretization setup
# ---------------------------------------------------------------------------------------------
function setup(N, K)
    rd = RefElemData(Quad(), N=N)
    md = MeshData(rd, circles, K, K, Subtriangulation(); precompute_operators=true)

    # mesh summary: report the smallest cut cell relative to a Cartesian cell
    cut_volumes = vec(sum(md.wJq.cut, dims=1))
    cartesian_volume = (2 / K)^2
    @printf("  N=%d K=%2d: %4d Cartesian cells, %3d cut cells, cut volume fraction min/max = %.3g / %.3g\n",
            N, K, size(md.x.cartesian, 2), size(md.x.cut, 2),
            minimum(cut_volumes) / cartesian_volume, maximum(cut_volumes) / cartesian_volume)

    # weak differentiation matrices for cut cells
    (; cut_cell_operators) = md.mesh_type
    operatorType = eltype(cut_cell_operators.mass_matrices)
    weak_differentiation_matrices = Tuple{operatorType, operatorType}[]
    for e in axes(md.x.cut, 2)
        M = cut_cell_operators.mass_matrices[e]
        Pq = cut_cell_operators.projection_matrices[e]
        Dx, Dy = cut_cell_operators.differentiation_matrices[e]
        push!(weak_differentiation_matrices, (M \ (-Dx' * M * Pq), M \ (-Dy' * M * Pq)))
    end

    # weak differentiation matrices for Cartesian cells
    Drw = rd.M \ (-rd.Dr' * rd.M * rd.Pq)
    Dsw = rd.M \ (-rd.Ds' * rd.M * rd.Pq)

    # constant advection field
    ax = fill!(similar(md.x), a_vel[1])
    ay = fill!(similar(md.y), a_vel[2])
    axq, ayq = map(a -> interp_to_volume_nodes(a, rd, md), (ax, ay))
    a_dot_n = interp_to_face_nodes(ax, rd, md) .* md.nx + interp_to_face_nodes(ay, rd, md) .* md.ny
    inflow = findall(a_dot_n[md.mapB] .< -100 * eps())

    u0 = u_exact.(md.x, md.y, 0.0)

    params = (; axq, ayq, a_dot_n, rd, md, inflow,
                weak_differentiation_matrices, Drw, Dsw,
                dudx = similar(u0), dudy = similar(u0), du_visc = similar(u0),
                epsilon)
    return rd, md, u0, params
end

# ---------------------------------------------------------------------------------------------
# semi-discretization (copied from advection_diffusion.jl, with exact-solution boundary data
# and zero forcing)
# ---------------------------------------------------------------------------------------------
function rhs!(du, u, params, t)
    (; axq, ayq, a_dot_n, rd, md, inflow, weak_differentiation_matrices, Drw, Dsw) = params
    (; cut_cell_operators, cut_face_nodes) = md.mesh_type

    uq = interp_to_volume_nodes(u, rd, md)
    uM = interp_to_face_nodes(u, rd, md)
    uP = similar(uM)
    @. uP = uM[md.mapP]

    # advective part: exact solution at inflow boundary nodes (a·n < 0), uP = uM at outflow
    inflow_nodes = md.mapB[inflow]
    @. uP[inflow_nodes] = u_exact(md.xf[inflow_nodes], md.yf[inflow_nodes], t)
    interface_flux = @. (a_dot_n * 0.5 * (uP + uM) - 0.5 * abs(a_dot_n) * (uP - uM)) * md.Jf

    du.cartesian .= md.rxJ.cartesian .* (Drw * (axq.cartesian .* uq.cartesian)) +
                    md.syJ.cartesian .* (Dsw * (ayq.cartesian .* uq.cartesian)) +
                    rd.LIFT * interface_flux.cartesian
    for e in axes(md.x.cut, 2)
        Dxw, Dyw = weak_differentiation_matrices[e]
        LIFT = cut_cell_operators.lift_matrices[e]
        du.cut[:, e] .= (Dxw * (axq.cut[:, e] .* uq.cut[:, e])) +
                        (Dyw * (ayq.cut[:, e] .* uq.cut[:, e])) +
                         LIFT * interface_flux.cut[cut_face_nodes[e]]
    end

    # diffusive part (BR1): exact solution imposed on the entire boundary, so that
    # u* = 0.5 * (uP + uM) = u_exact there
    @. uP[md.mapB] = 2 * u_exact(md.xf[md.mapB], md.yf[md.mapB], t) - uM[md.mapB]
    u_jump = @. 0.5 * (uP - uM)

    (; dudx, dudy) = params
    dudx.cartesian .= (md.rxJ.cartesian .* (rd.Dr * u.cartesian) +
                       rd.LIFT * (u_jump.cartesian .* md.nxJ.cartesian))
    dudy.cartesian .= (md.syJ.cartesian .* (rd.Ds * u.cartesian) +
                       rd.LIFT * (u_jump.cartesian .* md.nyJ.cartesian))
    for e in axes(md.x.cut, 2)
        Dx, Dy = cut_cell_operators.differentiation_matrices[e]
        LIFT = cut_cell_operators.lift_matrices[e]
        fids = cut_face_nodes[e]
        dudx.cut[:, e] .= (Dx * u.cut[:, e]) + LIFT * (u_jump.cut[fids] .* md.nxJ.cut[fids])
        dudy.cut[:, e] .= (Dy * u.cut[:, e]) + LIFT * (u_jump.cut[fids] .* md.nyJ.cut[fids])
    end
    @. dudx /= md.J
    @. dudy /= md.J

    # compute normal derivative
    dudxM = interp_to_face_nodes(dudx, rd, md)
    dudyM = interp_to_face_nodes(dudy, rd, md)
    dudnJM = @. dudxM * md.nxJ + dudyM * md.nyJ

    # compute jump, account for sign of normal in dudnP; zero jump on the boundary
    dudnJP = similar(dudnJM)
    @. dudnJP = -dudnJM[md.mapP]
    @. dudnJP[md.mapB] = dudnJM[md.mapB]
    dudnJ_jump = @. 0.5 * (dudnJP - dudnJM)

    (; epsilon, du_visc) = params
    du_visc.cartesian .= md.rxJ.cartesian .* (rd.Dr * dudx.cartesian) +
                         md.syJ.cartesian .* (rd.Ds * dudy.cartesian) +
                         rd.LIFT * (dudnJ_jump.cartesian)
    for e in axes(md.x.cut, 2)
        Dx, Dy = cut_cell_operators.differentiation_matrices[e]
        LIFT = cut_cell_operators.lift_matrices[e]
        fids = cut_face_nodes[e]
        du_visc.cut[:, e] .= (Dx * dudx.cut[:, e]) + (Dy * dudy.cut[:, e]) +
                              LIFT * (dudnJ_jump.cut[fids])
    end

    @. du = -(du - epsilon * du_visc) / md.J
    return du
end

# ---------------------------------------------------------------------------------------------
# error calculation and driver codes
# ---------------------------------------------------------------------------------------------

function l2_error(u, t, rd, md)
    uq = interp_to_volume_nodes(u, rd, md)
    uq_exact = u_exact.(md.xq, md.yq, t)
    return sqrt(sum(md.wJq .* (uq .- uq_exact).^2))
end

function run_case(P, K; abstol=1e-8, reltol=1e-8)
    rd, md, u0, params = setup(P, K)
    ode = ODEProblem(rhs!, u0, (0.0, T_final), params)
    walltime = @elapsed sol = solve(ode, Tsit5(); abstol, reltol, save_everystep=false)
    error = l2_error(sol.u[end], T_final, rd, md)
    @printf("  P=%d K=%2d: L2 error = %.4e  (%d accepted / %d rejected steps, %.1f s)\n",
            P, K, error, sol.stats.naccept, sol.stats.nreject, walltime)
    return (; P, K, h = 2 / K, error, nsteps = sol.stats.naccept, walltime)
end

const series_colors  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
const series_markers = [:circle, :rect, :diamond, :utriangle]

function plot_convergence(results; filename = joinpath(@__DIR__, "convergence_advection_diffusion.png"))
    gr()
    Ns = sort(collect(keys(results)))
    h_all = sort(unique(x.h for N in Ns for x in results[N]))
    p = plot(xscale=:log10, yscale=:log10,
             xlabel=L"\mathrm{Mesh\ size}\ h", ylabel=L"L^2\ \mathrm{error}",
             xticks=(h_all, latexstring.(h_all)), xlims=(minimum(h_all) / 1.3, maximum(h_all) * 2.0),
             legend=:bottomright, gridalpha=0.15, size=(700, 500), dpi=400,
             fontfamily="Computer Modern")
    minerr = Inf
    maxerr = -Inf
    for (i, N) in enumerate(Ns)
        r = sort(results[N], by = x -> x.K)
        h = [x.h for x in r]
        err = [x.error for x in r]
        minerr = min(minimum(err), minerr)
        maxerr = max(maximum(err), maxerr)
        color = series_colors[i]
        plot!(p, h, err, color=color, linewidth=2, marker=series_markers[i], markersize=6,
              markerstrokecolor=:white, label=L"P = %$N")
        # reference slope h^(N+1), anchored at the finest level, in the series color
        plot!(p, h, err[end] .* (h ./ h[end]).^(N+1), color=color, linestyle=:dash,
              linewidth=1, label="")
        # direct label at the coarsest point
        annotate!(p, h[1] * 1.12, err[1], text(L"P = %$N", 9, :left, "#0b0b0b"))
    end
    plot!(p, [NaN], [NaN], color=:gray, linestyle=:dash, linewidth=1, label=L"O(h^{P+1})")
    log10min = floor(Int, log10(minerr))
    log10max = ceil(Int, log10(maxerr))
    plot!(p, ylims=10.0 .^ (log10min, log10max), yticks=10.0 .^ (log10min:log10max))
    savefig(p, filename)
    println("Saved convergence plot to $filename")
    return p
end

function print_convergence_table(P_list, K_list, results)
    errs = Dict(P => [r.error for r in results[P]] for P in P_list)
    rates = Dict(P => [NaN ; -1 * diff(log.(errs[P])) ./ diff(log.(K_list))] for P in P_list)
    colwidth = 9 + 2 + 5   # error, gap, rate
    @printf("%8s", " ")
    for P in P_list
        @printf(" | %-*s", colwidth, "P = $P")
    end
    println()
    @printf("%-8s", "h")
    for P in P_list
        @printf(" | %-9s  %5s", "L2 error", "rate")
    end
    println()
    println("-"^(8 + length(P_list) * (3 + colwidth)))
    for (i, K) in enumerate(K_list)
        @printf("%-8.4f", 2 / K)
        for P in P_list
            rate = isnan(rates[P][i]) ? "-" : @sprintf("%.2f", rates[P][i])
            @printf(" | %9.3e  %5s", errs[P][i], rate)
        end
        println()
    end
end

results = Dict{Int, Vector{NamedTuple}}()
for P in P_list
    println("Running P = $P")
    results[P] = [run_case(P, K) for K in K_list]
end

print_convergence_table(P_list, K_list, errs, rates)

plot_convergence(results)
