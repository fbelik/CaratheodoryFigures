using CaratheodoryPruning
using CairoMakie
using LaTeXStrings
using JuMP
using Tulip
using HiGHS
using Ipopt
using NonNegLeastSquares
using Random

function caratheodory_pruning_lp(V, w; optimizer=HiGHS.Optimizer, zero_tol=1e-5, noise=1)
    M, N = size(V)
    if M < N
        V = transpose(V)
        M, N = N, M
    end
    η = V' * w
    c = rand(M)
    # Form LP model
    model = Model(optimizer)
    set_silent(model)
    @variable(model, w_pruned[1:M] >= 0)
    @objective(model, Min, c' * w_pruned)
    @constraint(model, V' * w_pruned == η)
    optimize!(model)
    solved_and_feasible = is_solved_and_feasible(model)
    (!solved_and_feasible && noise >= 1) ? println("Warning: LP solution not solved/feasible") : nothing
    # Return solution
    w_pruned = [value(w_pruned[i]) for i in eachindex(w)]
    inds = [i for i in eachindex(w_pruned) if w_pruned[i] >= zero_tol]
    if length(inds) != N && noise >= 1
        println("Warning: LP solution found $(length(inds)) indices instead of $N")
    end
    return w_pruned, inds
end

function caratheodory_pruning_nnls(V, w; alg=:nnls, zero_tol=1e-16, noise=1, max_iter=1000)
    M, N = size(V)
    if M < N
        V = transpose(V)
        M, N = N, M
    end
    η = V' * w
    w_pruned = nonneg_lsq(V', η, alg=alg, max_iter=max_iter)[:,1]
    inds = [i for i in eachindex(w_pruned) if w_pruned[i] >= zero_tol]
    if length(inds) != N && noise >= 1
        println("Warning: NNLS solution found $(length(inds)) indices instead of $N")
    end
    return w_pruned, inds
end

function time_test(Ms, Ns; time_tol=5.0, reps=5)
    Random.seed!(123)
    times = [zeros(length(Ms)) .* NaN for i in 1:3]

    # Run each once for compilation
    M = 100; N=10;
    V = rand(M, N)
    w = rand(M)
    caratheodory_pruning(V, w)
    caratheodory_pruning_lp(V, w)
    caratheodory_pruning_nnls(V, w)

    curtimes = [zeros(reps) for i in 1:3]
    for (i,(M,N)) in enumerate(zip(Ms, Ns))
        doCS = i == 1 || (times[1][i-1] <= time_tol)
        doLP = i == 1 || (times[2][i-1] <= time_tol)
        doNNLS = i == 1 || (times[3][i-1] <= time_tol)
        doAny = doCS || doLP || doNNLS
        if doAny
            for j in 1:reps
                V = rand(M, N)
                w = rand(M)
                if doCS
                    t = @timed caratheodory_pruning(V, w)
                    curtimes[1][j] = t.time
                end
                if doLP
                    t = @timed caratheodory_pruning_lp(V, w)
                    curtimes[2][j] = t.time
                end
                if doNNLS
                    t = @timed caratheodory_pruning_nnls(V, w)
                    curtimes[3][j] = t.time
                end
                println("- finished j=$j/$(reps)")
            end
        else
            times[1] = times[1][1:(i-1)]
            times[2] = times[2][1:(i-1)]
            times[3] = times[3][1:(i-1)]
            break
        end
        times[1][i] = doCS ? sum(curtimes[1]) / 5 : NaN
        times[2][i] = doLP ? sum(curtimes[2]) / 5 : NaN
        times[3][i] = doNNLS ? sum(curtimes[3]) / 5 : NaN
        println("finished i=$i/$(length(Ms))")
    end
    return times
end

function time_test_M(itrs = 4:14, time_tol=5.0, reps=5)
    Ms = 2 .^ itrs
    Ns = [2^(itrs[1]-1) for _ in itrs]
    times = time_test(Ms, Ns, time_tol=time_tol, reps=reps)
    idx = length(times[1])
    fig = Figure(fontsize=16);
    Axis(fig[1,1], xlabel="M", xscale=log2, xticks=(2 .^ (itrs), [latexstring("2^{$i}") for i in itrs]),
                ylabel="Time (s)", yscale=log2, yticks=(2.0 .^ (-16:2:16), [latexstring("2^{$i}") for i in (-16:2:16)]),
                title="Random Matrix Runtime Comparison with N=$(Ns[1])")
    scatter!(fig[1,1], Ms[1:idx], times[1], label="CS Pruning", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[2], label="LP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[3], label="NNLS", markersize=15)
    lines!(fig[1,1], Ms[1:idx], 2.0 .^ (itrs[1:idx]) .* (times[3][1]) ./ (2.0^(itrs[1])), label=nothing, linestyle=:dash, color=:black)
    axislegend(position=:lt)
    fig
end

function time_test_NM(itrs = 1:8, time_tol=1.0, reps=5)
    Ns = 2 .^ itrs
    Ms = 2 .^ (2 .* itrs)
    times = time_test(Ms, Ns, time_tol=time_tol, reps=reps)
    idx = length(times[1])
    fig = Figure(fontsize=16);
    Axis(fig[1,1], xlabel="M", xscale=log2, xticks=(2 .^ (2 .* itrs), [latexstring("2^{$(2*i)}") for i in itrs]),
                ylabel="Time (s)", yscale=log2, yticks=(2.0 .^ (-16:2:16), [latexstring("2^{$i}") for i in (-16:2:16)]),
                title="Random Matrix Runtime Comparison with M=N²")
    scatter!(fig[1,1], Ms[1:idx], times[1], label="CS Pruning", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[2], label="LP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[3], label="NNLS", markersize=15)
    lines!(fig[1,1], Ms[1:idx], Ms[1:idx] .* (times[3][1]) ./ (2.0^(itrs[1])), label=nothing, linestyle=:dash, color=:black)
    axislegend(position=:lt)
    fig
end

plt01 = time_test_M(4:20, 5)
plt02 = time_test_M(9:20, 5)
# plt2 = time_test_NM(1:10, 0.5)

M = 1000; N = 10;
V0 = rand(M,N);
V = OnDemandMatrix(M, N, i -> view(V0, i, :), by=:rows)
V = copy(V0);
w = rand(M); w[1:N] .= 1e4; w[N+1:end] .= 1e-4;

w1,inds1 = caratheodory_pruning(V, w)
sqrt(sum(abs.(V'w .- V[inds1,:]'w1[inds1]) .^ 2))
w2,inds2 = caratheodory_pruning_lp(V, w, optimizer=HiGHS.Optimizer)
sqrt(sum(abs.(V'w .- V[inds2,:]'w2[inds2]) .^ 2))
w3,inds3 = caratheodory_pruning_nnls(V, w)
sqrt(sum(abs.(V'w .- V[inds3,:]'w3[inds3]) .^ 2))


# @profview w1,inds1 = caratheodory_pruning(V, w);
# sqrt(sum(abs.(V'w .- V[inds1,:]'w1[inds1]) .^ 2))

# @time w2,inds2 = caratheodory_pruning_lp(V, w, optimizer=HiGHS.Optimizer);
# sqrt(sum(abs.(V'w .- V[inds2,:]'w2[inds2]) .^ 2))

# @profview w3,inds3 = caratheodory_pruning_nnls(V, w);
# sqrt(sum(abs.(V'w .- V[inds3,:]'w3[inds3]) .^ 2))