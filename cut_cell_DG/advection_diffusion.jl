using StartUpDG
using OrdinaryDiffEqTsit5
using StaticArrays
using Trixi: AliveCallback
using Plots

cells_per_dimension = 8
cells_per_dimension_x, cells_per_dimension_y = cells_per_dimension, cells_per_dimension
R0 = 0.425
circle = PresetGeometries.Circle(R=R0, x0=0.5, y0=-0.5)
circle2 = PresetGeometries.Circle(R=R0, x0=-0.5, y0=0.5)

rd = RefElemData(Quad(), N=7; Nplot=25)
md = MeshData(rd, (circle, circle2), 
            cells_per_dimension_x, cells_per_dimension_y,
            Subtriangulation(); 
            precompute_operators=true)

# visualize un-pruned quadrature
N = rd.N
N_phys_frame_geo = 2 * N^2 - N + 2 * (N-1) 
rd_tri = RefElemData(Tri(), Polynomial(MultidimensionalQuadrature()), N, 
                    quad_rule_vol=NodesAndModes.quad_nodes_tri(N_phys_frame_geo))

e = 1
(; cutcells) = md.mesh_type.cut_cell_data
Np_target = StartUpDG.Np_cut(2 * N - 1)
xq, yq, wJq = StartUpDG.subtriangulated_cutcell_quadrature(cutcells[e], rd_tri)

plotly()
scatter(xq, yq, leg=false, ms=2, markercolor=:royalblue3, label="Before pruning")
scatter!(md.xq.cut[:, e], md.yq.cut[:, e], ms=4, msw=3, markerstrokecolor=:darkorange1, 
         marker=:circle, markercolor = :transparent, label="After pruning")
plot!(dpi=600, leg=false, ratio=1, axis=([], false))

# plot cut domain and quadrature points
s = LinRange(0, 1, 25)
plot()
for cutcell in cutcells
    x = getindex.(cutcell.(s), 1)
    y = getindex.(cutcell.(s), 2)
    plot!(x, y, color=:black, linewidth=1.5)
end
for e in axes(md.xf.cartesian, 2)
    xf = reshape(md.x.cartesian[rd.Fmask, e], :, 4)
    yf = reshape(md.y.cartesian[rd.Fmask, e], :, 4)
    for f in axes(xf, 2)
        plot!(xf[:, f], yf[:, f], color=:black, linewidth=1.5)
    end
end
scatter!(md.xq.cut, md.yq.cut, ms=2, msw=1, legend=false)
plot!(dpi=600, leg=false, ratio=1, axis=([], false))
savefig("p")

# compute additional operators            
(; cut_cell_operators) = md.mesh_type
operatorType = eltype(cut_cell_operators.mass_matrices)
weak_differentiation_matrices = Tuple{operatorType, operatorType}[]
for e in axes(md.x.cut, 2)
    M = cut_cell_operators.mass_matrices[e]
    Vq = cut_cell_operators.volume_interpolation_matrices[e]
    Pq = cut_cell_operators.projection_matrices[e]
    Dx, Dy = cut_cell_operators.differentiation_matrices[e]
    push!(weak_differentiation_matrices, (M \ (-Dx' * M * Pq), M \ (-Dy' * M * Pq)))
end

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

function interp_to_plot_nodes(u, rd, md; Nplot=25)
    (; physical_frame_elements) = md.mesh_type

    up_cartesian = rd.Vp * u.cartesian
    up_cut = typeof(u.cut[:,1])[]
    for (e, elem) in enumerate(physical_frame_elements)
        VDM = vandermonde(elem, rd.N, md.x.cut[:, e], md.y.cut[:, e])
        Vp = vandermonde(elem, rd.N, equi_nodes(elem, md.mesh_type.objects, Nplot)...) / VDM
        push!(up_cut, Vp * u.cut[:,e])
    end
    return NamedArrayPartition(; cartesian=up_cartesian, cut=vcat(up_cut...))
end
xp = interp_to_plot_nodes(md.x, rd, md)
yp = interp_to_plot_nodes(md.y, rd, md)

initial_condition(x, y, t) = 0.0
u = initial_condition.(md.xyz..., 0)

# advection field
ax = @. -md.y
ay = @. md.x
axq, ayq = map(a -> interp_to_volume_nodes(a, rd, md), (ax, ay))
a_dot_n = interp_to_face_nodes(ax, rd, md) .* md.nx + interp_to_face_nodes(ay, rd, md) .* md.ny

inflow = findall(a_dot_n[md.mapB] .< -100 * eps())

function rhs!(du, u, params, t)
    (; axq, ayq, a_dot_n, rd, md, weak_differentiation_matrices, ) = params
    (; cut_cell_operators, cut_face_nodes) = md.mesh_type

    uq = interp_to_volume_nodes(u, rd, md)
    uM = interp_to_face_nodes(u, rd, md)
    uP = similar(uM)
    @. uP = uM[md.mapP]

    # advective part 
    @. uP[md.mapB[inflow]] = initial_condition(md.xf[md.mapB[inflow]], md.yf[md.mapB[inflow]], t) 
    interface_flux = @. (a_dot_n * 0.5 * (uP + uM) - 0.5 * abs(a_dot_n) * (uP - uM)) * md.Jf   

    Drw = rd.M \ (-rd.Dr' * rd.M * rd.Pq)
    Dsw = rd.M \ (-rd.Ds' * rd.M * rd.Pq)
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

    # diffusive part
    uP[md.mapB] .= -uM[md.mapB] # uBC = 0 
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

    # compute jump, account for sign of normal in dudnP
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

    forcing = 1
    @. du = forcing - (du - epsilon * du_visc) ./ md.J
    return du
end

params = (; axq, ayq, a_dot_n, rd, md, inflow, 
            weak_differentiation_matrices, 
            dudx = similar(u), dudy = similar(u), 
            du_visc = similar(u), epsilon=1e-2) 

tspan = (0.0, 3.0)
ode = ODEProblem(rhs!, u, tspan, params)
sol = solve(ode, Tsit5(), callback=AliveCallback(alive_interval=100),
            saveat=LinRange(tspan..., 50))

gr()
for i in [10, 30, 50]
    up = interp_to_plot_nodes(sol.u[i], rd, md)
    scatter(xp, yp, zcolor=up, ratio=1, msw=0, ms=3) #, clims=(0, 2.25))
    plot!(dpi=600, leg=false, colorbar=false, axis=([], false))
    png("solution_$i.png")
end

@gif for (u, t) in zip(sol.u, sol.t)
    up = interp_to_plot_nodes(u, rd, md)
    scatter(xp, yp, zcolor=up, ratio=1, msw=0, ms=2, clims=(0, 2.5))
    title!("Time = $t")
end
