
using CairoMakie
using DelimitedFiles

set_theme!(theme_latexfonts())

# Load reference data
ref_data = joinpath(@__DIR__, "ref_building_gauges_h.txt")
building = readdlm(ref_data, skipstart=2)

N = 5

# Create figure with 6 rows
# fig = Figure(size = (700, 1200))
fig = Figure(size = (600, 800))

axes = Axis[]

set = [2, 4, 5, 6]

for (row, i) in enumerate(set)

    elementwise_data = joinpath(@__DIR__, "elementwise_G$(i)_N$(N).csv")
    nodewise_data    = joinpath(@__DIR__, "nodewise_G$(i)_N$(N).csv")

    elementwise = readdlm(elementwise_data, ',', skipstart=1)
    nodewise    = readdlm(nodewise_data, ',', skipstart=1)

    x1 = elementwise[:,2]
    y1 = elementwise[:,3]

    x2 = nodewise[:,2]
    y2 = nodewise[:,3]

    xb = building[:,1]
    yb = building[:,i+1]

    idx = i < 6 ? (1:2:length(xb)) : (1:15:length(xb))
    ylim = i < 6 ? (0.0, 0.16) : (0.15, 0.42)

    ax = Axis(fig[row,1],   # <-- use row instead of i
        ylabel = "water height (m)",
        limits = (0, 30, ylim[1], ylim[2])
    )

    push!(axes, ax)

    scatter!(ax, xb[idx], yb[idx],
        color = :gray40,
        markersize = 6,
        label = "Experiment"
    )

    lines!(ax, x2, y2,
        linestyle = :solid,
        linewidth = 2,
        color = :blue,
        label = "Node-wise limiting"
    )

    lines!(ax, x1, y1,
        linestyle = :dash,
        linewidth = 2,
        color = :orange,
        label = "Element-wise limiting"
    )

    ypos = i == 6 ? (ylim[2] * 0.6) : (ylim[2] * 0.9)

    text!(ax, 28, ypos,
        text = "G$(i)",
        align = (:right, :top),
        fontsize = 16
    )

    ax.xticks = 0:5:30
    ax.xminorticks = 0:1:30
    ax.xminorticksvisible = true
end

# Put in a legend only in the bottom figure
axislegend(axes[end], position = :rt)

# Share x-axis
linkxaxes!(axes...)

# Only bottom axis shows x labels
for ax in axes[1:end-1]
    hidexdecorations!(ax, grid=false)
end

axes[end].xlabel = "time (s)"

save("gauges_four_N$(N).pdf", fig)
