include("quadrature.jl")
CairoMakie.activate!()

function perturbation_change_weights(m::MonteCarloQuadrature, alpha=1e-8)
    mp = MonteCarloQuadrature(m.D, m.M, m.in_shape, deepcopy(m.pts), copy(m.w))
    M = length(m.w)
    wp = randn(M)
    wp .-= sum(wp) / M
    wp .*= 2 * norm(m.w, 1) * alpha / norm(wp, 1)
    mp.w .+= wp
    return mp
end

function perturbation_add_weights(m::MonteCarloQuadrature, alpha=1e-8, dM=10000)
    mp = deepcopy(m)
    addPts!(mp, dM)
    mp.w[eachindex(m.w)] .= m.w
    mp.w[length(m.w)+1:end] .= 2*alpha*norm(m.w,1) / (dM * (1-alpha))
    return mp
end

in_circle(x) = (x[1]^2 + x[2]^2) <= 1.0

basis = (i,x) -> legendrep(i,x)
in_multi_index_set = hyperbolic_cross(30)

function perturbation_plot(;alphas=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], reps=6, M=1000, quan=0.8, perturb=perturbation_change_weights, lpopt=HiGHS.Optimizer)
    Random.seed!(1234)
    m0 = MonteCarloQuadrature(in_circle, 2)
    addPts!(m0, M)
    f1 = visualize(m0, markersize=3, title=L"\textbf{Discrete Measure} $\mu_M$")
    m0_pruned = prune(m0, basis, in_multi_index_set, progress=false)
    c = rand(M)
    m0_pruned_lp = prune(m0, basis, in_multi_index_set, progress=false, method=:lp, c=c, optimizer=lpopt)
    m0_pruned_nnls = prune(m0, basis, in_multi_index_set, progress=false, method=:nnls)
    ress_cp = [zeros(reps) for _ in alphas]
    ress_lp = [zeros(reps) for _ in alphas]
    ress_nnls = [zeros(reps) for _ in alphas]
    for i in ProgressBar(eachindex(alphas))
        for j in 1:reps
            mp = perturb(m0, alphas[i])
            mp_pruned = prune(mp, basis, in_multi_index_set, progress=false)
            ress_cp[i][j] = dTV(m0_pruned, mp_pruned)
            try
                mp_pruned = prune(mp, basis, in_multi_index_set, progress=false, method=:lp, noise=0,c=[c ; (ones(length(mp.w)-M))], optimizer=lpopt)
                ress_lp[i][j] = dTV(m0_pruned_lp, mp_pruned)
            catch
                ress_lp[i][j] = 1.0
            end
            mp_pruned = prune(mp, basis, in_multi_index_set, progress=false, method=:nnls, noise=0)
            ress_nnls[i][j] = dTV(m0_pruned_nnls, mp_pruned)
        end
    end
    resss = [ress_cp, ress_lp, ress_nnls]
    f2 = Figure()
    ax = Axis(f2[1, 1], xscale=log10, yscale=log10, xticks=LogTicks(-10:0), yticks=LogTicks(-10:1),
                    xlabel=L"$d_{TV}(\mu_M,\tilde{\mu}_M)$", ylabel=L"$d_{TV}(\nu,\tilde{\nu})$",
                    title="Pruned Perturbation Errors")
    cols = [:blue, :orange, :green]
    pltvals = []
    for (i,ress) in enumerate(resss)
        vals = quantile.(ress, 0.5)
        lowerrs = vals .- quantile.(ress, 1-quan)
        higherrs = quantile.(ress, quan) .- vals
        pltval = errorbars!(alphas, vals, lowerrs, higherrs, whiskerwidth = 14, color=cols[i])
        push!(pltvals, pltval)
        pltval = scatter!(alphas, vals, color=cols[i], markersize=12)
        push!(pltvals, pltval)
    end
    ylims!(ax, minimum(alphas)*1e-1, 1e1)
    xlims!(ax, minimum(alphas)*1e-1, maximum(alphas)*1e1)
    lines!([minimum(alphas)*1e-1,maximum(alphas)*1e1],[minimum(alphas)*1e-1,maximum(alphas)*1e1], color=:black, linestyle=:dash)
    lines!([minimum(alphas)*1e-1,maximum(alphas)*1e1],[1.0, 1.0], color=:red, linestyle=:dash)
    Legend(f2[1, 2],
        [pltvals[i:i+1] for i in 1:2:5],
        ["GSCSP", "LP", "NNLS"])
    return (f1, f2, resss)
end

function perturbation_lp_plot(;alphas=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], reps=6, M=1000, quan=0.8, perturb=(m,a) -> perturbation_add_weights(m,a,10), lpopt=HiGHS.Optimizer)
    Random.seed!(1234)
    m0 = MonteCarloQuadrature(in_circle, 2)
    addPts!(m0, M)
    f1 = visualize(m0, markersize=3, title=L"\textbf{Discrete Measure} $\mu_M$")
    c = rand(M)
    m0_pruned_lp = prune(m0, basis, in_multi_index_set, progress=false, method=:lp, c=c, optimizer=lpopt)
    ress_lp_ones = [zeros(reps) for _ in alphas]
    ress_lp_rands = [zeros(reps) for _ in alphas]
    for i in ProgressBar(eachindex(alphas))
        for j in 1:reps
            mp = perturb(m0, alphas[i])
            try
                mp_pruned = prune(mp, basis, in_multi_index_set, progress=false, method=:lp, noise=0,c=[c ; (ones(length(mp.w)-M))], optimizer=lpopt)
                ress_lp_ones[i][j] = dTV(m0_pruned_lp, mp_pruned)
            catch e
                println(e)
                ress_lp_ones[i][j] = 1.0
            end
            try
                mp_pruned = prune(mp, basis, in_multi_index_set, progress=false, method=:lp, noise=0,c=[c ; (rand(length(mp.w)-M))], optimizer=lpopt)
                ress_lp_rands[i][j] = dTV(m0_pruned_lp, mp_pruned)
            catch
                ress_lp_rands[i][j] = 1.0
            end
        end
    end
    resss = [ress_lp_ones,ress_lp_rands]
    f2 = Figure()
    ax = Axis(f2[1, 1], xscale=log10, yscale=log10, xticks=LogTicks(-10:0), yticks=LogTicks(-10:1),
                    xlabel=L"$d_{TV}(\mu_M,\tilde{\mu}_M)$", ylabel=L"$d_{TV}(\nu,\tilde{\nu})$",
                    title="Pruned Perturbation Errors")
    cols = [:orange, :purple]
    pltvals = []
    for (i,ress) in enumerate(resss)
        vals = quantile.(ress, 0.5)
        lowerrs = vals .- quantile.(ress, 1-quan)
        higherrs = quantile.(ress, quan) .- vals
        pltval = errorbars!(alphas, vals, lowerrs, higherrs, whiskerwidth = 14, color=cols[i])
        push!(pltvals, pltval)
        pltval = scatter!(alphas, vals, color=cols[i], markersize=12)
        push!(pltvals, pltval)
    end
    ylims!(ax, minimum(alphas)*1e-1, 1e1)
    xlims!(ax, minimum(alphas)*1e-1, maximum(alphas)*1e1)
    lines!([minimum(alphas)*1e-1,maximum(alphas)*1e1],[minimum(alphas)*1e-1,maximum(alphas)*1e1], color=:black, linestyle=:dash)
    lines!([minimum(alphas)*1e-1,maximum(alphas)*1e1],[1.0, 1.0], color=:red, linestyle=:dash)
    Legend(f2[1, 2],
        [pltvals[i:i+1] for i in 1:2:(length(pltvals)-1)],
        ["LP 1","LPrand"])
    return (f1, f2, resss)
end

lponly = false

M=10000
dM1=10
dM2=10000
reps=20
if lponly
    f1, f2, resss = perturbation_lp_plot(reps=reps, M=M, quan=0.8, perturb=(m,a) -> perturbation_add_weights(m,a,dM1));
    save("perturbation_plot_lp_add_$(dM1)_weights_$M.pdf", f2)

    f1, f2, resss = perturbation_lp_plot(reps=reps, M=M, quan=0.8, perturb=(m,a) -> perturbation_add_weights(m,a,dM2));
    save("perturbation_plot_lp_add_$(dM2)_weights_$M.pdf", f2)
else 
    f1, f2, resss = perturbation_plot(reps=reps, M=M, perturb=perturbation_change_weights);
    save("perturbation_plot_origmeasure_$M.pdf", f1)
    save("perturbation_plot_change_weights_$M.pdf", f2)

    f1, f2, resss = perturbation_plot(reps=reps, M=M, quan=0.8, perturb=(m,a) -> perturbation_add_weights(m,a,dM1));
    save("perturbation_plot_add_$(dM1)_weights_$M.pdf", f2)

    f1, f2, resss = perturbation_plot(reps=reps, M=M, quan=0.8, perturb=(m,a) -> perturbation_add_weights(m,a,dM2));
    save("perturbation_plot_add_$(dM2)_weights_$M.pdf", f2)

    # Construct base quadrature rule accurate w.r.t. basis
    m0 = MonteCarloQuadrature(in_circle, 2)
    addPts!(m0, 100000)
    m = prune(m0, basis, in_multi_index_set, method=:nnls)
    M0 = length(m.w)
    plt1 = visualize(m, markersize=10, title="Unperturbed Quadrature Rule")
    save("base_weights.pdf", plt1)
    save_mc("perturbation_test_unperturbed", m)
    # Perturb quadrature rule with small weights
    mp = perturbation_add_weights(m, 1e-5, 100)

    m_pruned_cs = prune(mp, basis, in_multi_index_set) 
    plt2 = visualize(m_pruned_cs, markersize=10, title="GSCSP Pruned Perturbed Quadrature Rule")
    save("cs_pruned_weights.pdf", plt2)

    c = ones(length(mp.w))
    m_pruned_lp = prune(mp, basis, in_multi_index_set, method=:lp, c=c)
    plt3 = visualize(m_pruned_lp, markersize=10, title="LP Perturbed Quadrature Rule")
    save("lp_pruned_weights.pdf", plt3)

    m_pruned_nnls = prune(mp, basis, in_multi_index_set, method=:nnls)
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

    plt5 = visualize(m_cs_err, markersize=10, title="GSCSP Pruned Relative Errors", weight_label="Relative Error", crange=(cmin,1))
    save("cs_relerr.pdf", plt5)
    plt6 = visualize(m_lp_err, markersize=10, title="LP Relative Errors", weight_label="Relative Error", crange=(cmin,1))
    save("lp_relerr.pdf", plt6)
    plt7 = visualize(m_nnls_err, markersize=10, title="NNLS Relative Errors", weight_label="Relative Error", crange=(cmin,1))
    save("nnls_relerr.pdf", plt7)

    # Visualize against each other
    plt8 = begin # GSCSP
        title = "GSCSP Pruned"
        opts = deepcopy(m_pruned_cs.pts)
        fig = Figure()
        ax = Axis(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]),
                            ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
                            limits=((-1,1),(-1,1)), width=400, height=400,
                            title=title)
        tks = range(-1,1,1001)
        cm = cgrad([:grey,:white])
        contourf!(fig[1,1], tks, tks, (x,y) -> m.in_shape([x,y]), colormap = cm)
        for pt in m.pts[1:M0]
            idx = findfirst(p -> p==pt, opts)
            if !isnothing(idx)
                scatter!(ax, [pt[1]],[pt[2]], marker=:star5, markersize=15, color=:blue, label="Original")
                deleteat!(opts, idx)
            else
                scatter!(ax, [pt[1]],[pt[2]], marker=:xcross, markersize=10, color=:red)
            end
        end
        for pt in opts
            scatter!(ax, [pt[1]],[pt[2]], marker=:cross, markersize=10, color=:darkred)
        end
        elem1 = [MarkerElement(color = :blue, marker = :star5, markersize = 20)]
        elem2 = [MarkerElement(color = :red, marker = :xcross, markersize = 20)]
        elem3 = [MarkerElement(color = :darkred, marker = :cross, markersize = 20)]
        Legend(fig[1, 2], [elem1,elem2,elem3], ["In both","In unperturbed","In perturbed"])
        resize_to_layout!(fig)
        fig
    end

    plt9 = begin # LP
        title = "LP Pruned"
        opts = deepcopy(m_pruned_lp.pts)
        fig = Figure()
        ax = Axis(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]),
                            ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
                            limits=((-1,1),(-1,1)), width=400, height=400,
                            title=title)
        tks = range(-1,1,1001)
        cm = cgrad([:grey,:white])
        contourf!(fig[1,1], tks, tks, (x,y) -> m.in_shape([x,y]), colormap = cm)
        for pt in m.pts[1:M0]
            idx = findfirst(p -> p==pt, opts)
            if !isnothing(idx)
                scatter!(ax, [pt[1]],[pt[2]], marker=:star5, markersize=15, color=:blue)
                deleteat!(opts, idx)
            else
                scatter!(ax, [pt[1]],[pt[2]], marker=:xcross, markersize=10, color=:red)
            end
        end
        for pt in opts
            scatter!(ax, [pt[1]],[pt[2]], marker=:cross, markersize=10, color=:darkred)
        end
        elem1 = [MarkerElement(color = :blue, marker = :star5, markersize = 20)]
        elem2 = [MarkerElement(color = :red, marker = :xcross, markersize = 20)]
        elem3 = [MarkerElement(color = :darkred, marker = :cross, markersize = 20)]
        Legend(fig[1, 2], [elem1,elem2,elem3], ["In both","In unperturbed","In perturbed"])
        resize_to_layout!(fig)
        fig
    end

    plt10 = begin # NNLS
        title = "NNLS Pruned"
        opts = deepcopy(m_pruned_nnls.pts)
        fig = Figure()
        ax = Axis(fig[1,1], xlabel="x", xticks=(-1:1, [L"-1",L"0",L"1"]),
                            ylabel="y", yticks=(-1:1, [L"-1",L"0",L"1"]),
                            limits=((-1,1),(-1,1)), width=400, height=400,
                            title=title)
        tks = range(-1,1,1001)
        cm = cgrad([:grey,:white])
        contourf!(fig[1,1], tks, tks, (x,y) -> m.in_shape([x,y]), colormap = cm)
        for pt in m.pts[1:M0]
            idx = findfirst(p -> p==pt, opts)
            if !isnothing(idx)
                scatter!(ax, [pt[1]],[pt[2]], marker=:star5, markersize=15, color=:blue)
                deleteat!(opts, idx)
            else
                scatter!(ax, [pt[1]],[pt[2]], marker=:xcross, markersize=10, color=:red)
            end
        end
        for pt in opts
            scatter!(ax, [pt[1]],[pt[2]], marker=:cross, markersize=10, color=:darkred)
        end
        elem1 = [MarkerElement(color = :blue, marker = :star5, markersize = 20)]
        elem2 = [MarkerElement(color = :red, marker = :xcross, markersize = 20)]
        elem3 = [MarkerElement(color = :darkred, marker = :cross, markersize = 20)]
        Legend(fig[1, 2], [elem1,elem2,elem3], ["In both","In unperturbed","In perturbed"])
        resize_to_layout!(fig)
        fig
    end

    save("cs_pointcomp.pdf", plt8)
    save("lp_pointcomp.pdf", plt9)
    save("nnls_pointcomp.pdf", plt10)
end
