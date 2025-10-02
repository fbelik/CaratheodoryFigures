include("lp_nnls.jl")
using CSV
using DataFrames

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
    pltiters = floor(Int, itrs[1]-1):ceil(Int, itrs[end]+1)
    fig = Figure(fontsize=16);
    ax = Axis(fig[1,1], xlabel="M", xscale=log2, xticks=(2 .^ (pltiters), [latexstring("2^{$i}") for i in pltiters]),
                ylabel="Time (s)", yscale=log2, yticks=(2.0 .^ (-16:2:16), [latexstring("2^{$i}") for i in (-16:2:16)]),
                limits=(2^3,2^24,2^(-16.0),2^4),
                title="Random Matrix Runtime Comparison with N=$(floor(Int,Ns[1]))")
    scatter!(fig[1,1], Ms[1:idx], times[1], label="GSCSP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[2], label="LP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[3], label="NNLS", markersize=15)
    lines!(fig[1,1], 2.0 .^ (-16:1.0:25), 2.0 .^ (-16:1.0:25) .* (times[3][1]) ./ (2.0^(itrs[1])), label="Slope 1", linestyle=:dash, color=:black)
    axislegend(position=:rb)
    Ms = Ms[eachindex(times[1])]
    Ns = Ns[eachindex(times[1])]
    df = DataFrame([[Ms, Ns] ; times], [:M, :N, :CSP,:LP,:NNLS])
    return fig, df
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
    scatter!(fig[1,1], Ms[1:idx], times[1], label="GSCSP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[2], label="LP", markersize=15)
    scatter!(fig[1,1], Ms[1:idx], times[3], label="NNLS", markersize=15)
    lines!(fig[1,1], Ms[1:idx], Ms[1:idx] .* (times[3][1]) ./ (2.0^(itrs[1])), label=nothing, linestyle=:dash, color=:black)
    axislegend(position=:lt)
    df = DataFrame([[Ms[eachindex(times[1])]] ; times], [:M, :CSP,:LP,:NNLS])
    fig, df
end

function plot_time_test(df, title, limits)
    Ms = 2 .^ df.M
    idx = length(df.CSP[1])
    fig = Figure(fontsize=16)
    ax = Axis(fig[1,1], xlabel="M", xscale=log2, xticks=(2 .^ (-16:1.0:16), [latexstring("2^{$i}") for i in (-16:1:16)]),
                limits=limits,
                ylabel="Time (s)", yscale=log2, yticks=(2.0 .^ (-16:2.0:16), [latexstring("2^{$i}") for i in (-16:2:16)]),
                title=title)
    scatter!(fig[1,1], Ms, df.CSP, label="GSCSP", markersize=15)
    scatter!(fig[1,1], Ms, df.LP, label="LP", markersize=15)
    scatter!(fig[1,1], Ms, df.NNLS, label="NNLS", markersize=15)
    lines!(fig[1,1], Ms, Ms .* (df.NNLS[1]) ./ (Ms[1]), label="Slope 1", linestyle=:dash, color=:black)
    axislegend(position=:lt)
    fig, ax
end

time_tol = 10.0
reps = 20
plt1, df1 = time_test_M(4:30, time_tol, reps)
save("run_comp_1_$(time_tol)_$(reps).pdf", plt1)
CSV.write("run_comp_1_$(time_tol)_$(reps).csv", df1)
plt2, df2 = time_test_M(9:30, time_tol, reps)
save("run_comp_2_$(time_tol)_$(reps).pdf", plt2)
CSV.write("run_comp_2_$(time_tol)_$(reps).csv", df2)

title = "Random Matrix Runtime Comparison with N=$(Ns[1])"