using CaratheodoryPruning
using CairoMakie
using LaTeXStrings
using JuMP
using Tulip
using HiGHS
using Ipopt
using NonNegLeastSquares
using Random
Random.seed!(1234)

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

function caratheodory_pruning_nnls(V, w; alg=:nnls, zero_tol=1e-16, noise=1, max_iter=10000)
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

# M = 1000; N = 10;
# V0 = rand(M,N);
# V = OnDemandMatrix(M, N, i -> view(V0, i, :), by=:rows)
# V = copy(V0);
# w = rand(M); w[1:N] .= 1e4; w[N+1:end] .= 1e-4;

# w1,inds1 = caratheodory_pruning(V, w)
# sqrt(sum(abs.(V'w .- V[inds1,:]'w1[inds1]) .^ 2))
# w2,inds2 = caratheodory_pruning_lp(V, w, optimizer=HiGHS.Optimizer)
# sqrt(sum(abs.(V'w .- V[inds2,:]'w2[inds2]) .^ 2))
# w3,inds3 = caratheodory_pruning_nnls(V, w)
# sqrt(sum(abs.(V'w .- V[inds3,:]'w3[inds3]) .^ 2))


# @profview w1,inds1 = caratheodory_pruning(V, w);
# sqrt(sum(abs.(V'w .- V[inds1,:]'w1[inds1]) .^ 2))

# @time w2,inds2 = caratheodory_pruning_lp(V, w, optimizer=HiGHS.Optimizer);
# sqrt(sum(abs.(V'w .- V[inds2,:]'w2[inds2]) .^ 2))

# @profview w3,inds3 = caratheodory_pruning_nnls(V, w);
# sqrt(sum(abs.(V'w .- V[inds3,:]'w3[inds3]) .^ 2))