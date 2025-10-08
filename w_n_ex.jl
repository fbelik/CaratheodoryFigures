using CairoMakie
using LaTeXStrings
using Random

Random.seed!(1234)

w = ones(5)
n = rand(5) .* 2 .- 1

cp, mp = findmin(i -> n[i]>0 ? w[i]/n[i] : Inf, eachindex(n))
cm, mm = findmax(i -> n[i]<0 ? w[i]/n[i] : -Inf, eachindex(n))

ws = [w .- c*n for c in [cm, cp]]

f = Figure(fontsize=24, size=(1200,300))
ax1 = Axis(f[1,1], limits=(0.5, 5.5, 0, 2.5), xticks=(1:5), yticks=(0:0.5:2.5), title=L"$\mathbf{w}$")
barplot!(ax1, w, color=:blue, width=0.3)
ax2 = Axis(f[1,2], limits=(0.5, 5.5, -1, 1), xticks=(1:5), yticks=(-1:0.5:1), title=L"$\mathbf{n}$")
barplot!(ax2, n, color=:green, width=0.3)
#text!(ax2, L"$m_+$", position=(mp-0.075, -0.2))
#text!(ax2, L"$m_-$", position=(mm-0.075, 0.05))
ax3 = Axis(f[1,3], limits=(0.5, 5.5, 0, 2.5), xticks=(1:5), yticks=(0:0.5:2.5), title=L"$\mathbf{w}-c_+\mathbf{n}$")
barplot!(ax3, w .- cp*n, color=:purple, width=0.3)
ax4 = Axis(f[1,4], limits=(0.5, 5.5, 0, 2.5), xticks=(1:5), yticks=(0:0.5:2.5), title=L"$\mathbf{w}-c_-\mathbf{n}$")
barplot!(ax4, w .- cm*n, color=:purple, width=0.3)

save("wn_1x4.pdf", f)


cs = range(cm, cp, 101)
cval = Observable(cs[1])
wplt = @lift w .- $cval .* n
fig = Figure()
ax = Axis(fig[1,1], limits=(0.5, 5.5, 0, 2.5), xticks=(1:5), yticks=(0:0.5:2.5), title=@lift ("w - cn, c=$(round($cval,digits=2))"))
barplot!(ax, wplt, color=:purple, width=0.3)
record(fig, "c_animation.mp4", cs;
        framerate = 30) do c
    cval[] = c
end