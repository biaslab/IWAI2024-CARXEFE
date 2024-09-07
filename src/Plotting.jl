module Plotting

using Plots # make sure to ]add FFMPEG
default(grid=false, label="", linewidth=3,margin=20Plots.pt)
using LinearAlgebra # diag
using Distributions # mean
#using Images

using Printf # @sprintf

using ..NARXAgents
using ..SpringMassSystems
#using Base64 # base64encode

using LaTeXStrings

export compare_agent_inference
export plot_predictions_paper
export plot_controls_paper
export plot_step
export animate_trial

export compute_performance_stats, plot_performance_comparison

"""    tickfontsize: Font size for the tick labels (x and y axis).
    guidefontsize: Font size for the axis labels (x and y labels).
    legendfontsize: Font size for the legend text.
    titlefontsize: Font size for the title.
    labelrotation: Rotation of the x and y axis labels (can be useful for adjusting space).
    series_annotations_fontsize: Font size for series annotations
        Title: 18-20 points
    Axis Labels: Title font size minus 2-4 points
    Tick Labels: Axis label font size minus 2-4 points
    Legend: Axis label font size or tick label font size
    """

titlefontsize=16
guidefontsize=titlefontsize-2
tickfontsize=titlefontsize-2
legendfontsize=guidefontsize

DEFAULTS = (
    titlefontsize=titlefontsize,
    guidefontsize=guidefontsize,
    tickfontsize=tickfontsize,
    legendfontsize=tickfontsize
)

MAX_FPS = 50

EnvironmentUnion = Union{DoubleDampedSpringMassSystem}

function generate_filename(field_name::Symbol, mode::Symbol)
    # Map the field_name and mode to appropriate strings
    field_name_string = field_name == :Pys ? "surprise" : "goal-alignment"
    mode_string = mode == :absolute ? "comparison" : "difference"

    # Construct the filename using the pattern "CARX-EFE-<field-name-string>-<mode-string>.png"
    return "figures/CARX-EFE-$(field_name_string)-$(mode_string).png"
end

function sanitize_values(values::Vector{Float64})
    has_zero = any(==(0.0), values)
    if has_zero
        println("WARNING: Some density values are 0. Replacing with floatmin(Float64)")
        values .= ifelse.(values .== 0.0, floatmin(Float64), values)
    end
    return values
end

function plot_agent_metric(agent_coupled::NARXAgent, agent_uncoupled::NARXAgent, realtime_xticks::Tuple{Vector{Float64}, Vector{String}}, field_name::Symbol; log_scale::Bool, xlabel::String, ylabel::LaTeXString, mode::Symbol=:absolute)
    # Access and sanitize the specified field (Pys or Pgs)
    values_coupled = sanitize_values(copy(getfield(agent_coupled, field_name)))
    values_uncoupled = sanitize_values(copy(getfield(agent_uncoupled, field_name)))

    # Apply log scale if specified
    if log_scale
        values_coupled = -log.(values_coupled)
        values_uncoupled = -log.(values_uncoupled)
    end

    # orange = RGB(1.0, 0.647, 0.0)  # Orange
    # blue = RGB(0.0, 0.0, 1.0)      # Blue
    # https://palett.es/e0dedf-28202f-2c3338-216985-16485b
    # https://davidmathlogic.com/colorblind/
    color1 = RGB(204/255, 102/255, 119/255) # CC6677
    color2 = RGB(40/255, 32/255, 47/255) # 28202F
    color2 = RGB(44/255, 51/255, 56/255) # 2C3338

    if mode == :absolute
        # Plot coupled and uncoupled values
        p = plot(values_coupled, label="coupled", ylabel=ylabel, xlabel=xlabel, xticks=realtime_xticks, color=color1)
        plot!(p, values_uncoupled, label="uncoupled", color=color2)
    elseif mode == :difference
        # Plot the difference between coupled and uncoupled values
        values_difference = values_uncoupled - values_coupled
        p = plot(values_difference, label="uncoupled - coupled", ylabel=ylabel, xlabel=xlabel, xticks=realtime_xticks)
        hline!([0.0], color=:black, alpha=0.5)
    else
        error("Invalid mode: $mode. Use :absolute or :difference.")
    end

    return p
end

function compare_agent_inference(agents_coupled::Vector{NARXAgent}, agents_uncoupled::Vector{NARXAgent}, realtime_xticks::Tuple{Vector{Float64}, Vector{String}}; field_name::Symbol=:Pys, log_scale::Bool=true, xlabel::String = "time [s]", mode::Symbol=:absolute)
    @assert length(agents_coupled) == length(agents_uncoupled)

    ylabel = field_name == :Pys ? L"-\log p( y_k \:\vert\: u_k)" : L"-\log p( y_k \:\vert\: y_*)"

    n_agents = length(agents_coupled)

    # Generate the filename based on field_name and mode
    f_name = generate_filename(field_name, mode)

    # Create plots for each agent comparison
    plots = [plot_agent_metric(agents_coupled[i], agents_uncoupled[i], realtime_xticks, field_name; log_scale=log_scale, xlabel=xlabel, ylabel=ylabel, mode=mode) for i in 1:n_agents]

    # Combine plots into a grid layout
    p = plot(plots..., layout=grid(n_agents, 1), size=(1500, 300 * n_agents),
             tickfontsize=DEFAULTS.tickfontsize, guidefontsize=DEFAULTS.guidefontsize,
             legendfontsize=DEFAULTS.legendfontsize, titlefontsize=DEFAULTS.titlefontsize)

    savefig(f_name)
    return p
end

function plot_predictions_paper(f_name::String, agents_coupled::Vector{NARXAgent}, sys_coupled::EnvironmentUnion, agents_uncoupled::Vector{NARXAgent}, sys_uncoupled::EnvironmentUnion, K::Int=1, n_std::Float64=1.0, color_observations::Symbol=:black, color_predictions::Symbol=:purple, color_goals::Symbol=:green, xlabel::String = "time [s]")
    plots = []
    n_plots = 1
    plot_ID = 1
    ys_coupled = hcat(sys_coupled.ys...)
    ys_uncoupled = hcat(sys_uncoupled.ys...)
    @assert size(ys_coupled) == size(ys_uncoupled)
    @assert size(ys_coupled)[1] == size(ys_uncoupled)[1]
    n_dim = size(ys_coupled)[1]
    ylims = zeros(n_dim, 2)
    for d in 1:n_dim
        ylims[d, :] .= (
            minimum([minimum(ys_coupled[d,:]), minimum(ys_uncoupled[d,:])]),
            maximum([maximum(ys_coupled[d,:]), maximum(ys_uncoupled[d,:])])
        )
    end

    println("ylims before: $ylims")
    ylims[1, :] = [-1.0, 2.5]
    ylims[2, :] = [-2.5, 5.0]
    println("ylims after: $ylims")

    for (agents_idx, agents) in enumerate([agents_coupled, agents_uncoupled])
        sys = agents_idx == 1 ? sys_coupled : sys_uncoupled
        xticks_pos, xticks_labels = get_realtime_xticks(sys)

        prelabel = agents_idx == 1 ? "coupled ARX-EFE" : "uncoupled ARX-EFE"
        ys = agents_idx == 1 ? ys_coupled : ys_uncoupled
        for (agent_idx, agent) in enumerate(agents)
            pred_m = hcat(agent.mys...)'
            pred_v = hcat(agent.vys...)'
            y_EFE_m = [pred_m[k] for k in 1:(agent.t-agent.time_horizon)]
            y_EFE_v = [pred_v[k] for k in 1:(agent.t-agent.time_horizon)]

            y_EFE_std = sqrt.(y_EFE_v)
            title = plot_ID == 1 ? "Observations vs $K-step ahead predictions" : ""
            title = "$prelabel"
            ylabel = agent_idx == 1 ? "displacement (z₁)" : "displacement (z₂)"
            p = plot(title=title, ylims=ylims[agent_idx, :]*1.2, xtickfontcolor=:black, xticks=(xticks_pos, xticks_labels))
            ylabel!(ylabel)
            label = plot_ID == 1 ? "observation" : nothing
            scatter!(ys[agent_idx, K:agent.t-agent.time_horizon], color=color_observations, label=label)
            label = plot_ID == 1 ? "prediction" : nothing
            plot!(y_EFE_m, ribbon=n_std*y_EFE_std, color=color_predictions, label=label)

            if agents_idx == 2 xlabel!(xlabel) end

            gs = hcat(agent.gs...)
            μgs = mean.(gs[1,:])
            σgs = std.(gs[1,:])
            label = plot_ID == 1 ? "goal prior" : nothing
            plot!(μgs, ribbon=n_std*σgs, color=color_goals, label=label, alpha=1.0, fillalpha=0.33)
            push!(plots, p)
            plot_ID += 1
        end
    end
    p = plot(plots...,layout=grid(2,2), size=(1500, 4*300), link=:both)
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    savefig(f_name)
    return p
end

function plot_controls_paper(f_name::String, sys_coupled::DoubleDampedSpringMassSystem, sys_uncoupled::DoubleDampedSpringMassSystem)
    plots = []
    xlabel = "time [s]"
    ylabel_base = "torque"
    for (sys_idx, sys) in enumerate([sys_coupled, sys_uncoupled])
        xticks_pos, xticks_labels = get_realtime_xticks(sys)
        prelabel = sys_idx == 1 ? "coupled" : "uncoupled"
        states = hcat(sys.zs...)
        torques = hcat(sys.us...)
        controls = hcat(sys.as...)
        for i in 1:sys.n_input
            title = "$prelabel ARX-EFE"
            push!(plots, plot())
            title!(title)
            ylabel = i == 1 ? "$(ylabel_base)₁" : "$(ylabel_base)₂"
            ylabel!(ylabel)
            if sys_idx == 2 xlabel!(xlabel) end
            plot!(controls[i, 1:sys.n_repeat:end], color="red", xticks=(xticks_pos, xticks_labels))
            plot!(torques[i, 1:sys.n_repeat:end], color="purple")
        end
    end
    dpi = 300
    w_in = 5
    h_in = 2 # 4
    p = plot(plots..., layout=grid(2,2), heights=[0.50, 0.50, 0.25, 0.25], size=(w_in*dpi,h_in*dpi))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)
    savefig(f_name)
    return p
end

rectangle(w, h, x, y) = Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

struct Point
    x :: Float64
    y :: Float64
end

function Base.:+(p1::Point, p2::Point)
    return Point(p1.x + p2.x, p1.y + p2.y)
end

function Base.:*(v::Float64, p::Point)
    return Point(v*p.x, v*p.y)
end

# calculates the realtime fps
function get_fps(sys::EnvironmentUnion)
    return Int64(ceil(1/(sys.Δt*sys.n_repeat)))
end

function calculate_opacity(base_opacity::Float64, j::Int, mid_idx::Int)
    dist_from_mid = abs(j - mid_idx)
    return base_opacity * (1 - dist_from_mid / (mid_idx - 1))
end

function plot_cart!(p_carts::Vector{Vector{Float64}}, box_w::Float64, box_h::Float64, n_std::Float64, stds::Vector{Float64}, opacity::Float64, colors::Vector{Symbol}; scale::Float64=1.0, scatter_opacity_factor::Float64=2.0, scatter_color=nothing, n_steps::Int=100)
    i_values = LinRange(-n_std, n_std, n_steps)
    mid_idx = ceil(Int, length(i_values) / 2)

    for (cart_idx, p_cart) in enumerate(p_carts)
        for (j, i) in enumerate(i_values)
            adjusted_opacity = calculate_opacity(opacity, j, mid_idx)
            plot!(
                rectangle(scale * box_w, scale * box_h, p_cart[1] - scale * 0.5 * box_w + i * stds[cart_idx], p_cart[2] - scale * 0.5 * box_h),
                opacity=adjusted_opacity,
                color=colors[cart_idx],
                label=nothing
            )
        end
        scatter!(
            [p_cart[1]], [p_cart[2]],
            color=scatter_color === nothing ? colors[cart_idx] : scatter_color,
            label=nothing,
            opacity=scatter_opacity_factor * opacity
        )
    end
end

function calculate_cart_positions(positions::Vector{Float64}, p_offset::Vector{Float64}, y_offset::Float64)
    p_carts = Vector{Vector{Float64}}()
    for pos in positions
        push!(p_carts, p_offset + [pos, y_offset])
    end
    return p_carts
end

function plot_distributions(p_carts::Vector{Vector{Float64}}, σ_values::Vector{Float64}, y_offset_base::Float64, y_direction::Int64, cart_colors::Vector{Symbol})
    for (cart_idx, p_cart) in enumerate(p_carts)
        x = p_cart[1] - 2:0.01:p_cart[1] + 2
        dist = Normal(p_cart[1], σ_values[cart_idx])
        y_offset = y_offset_base .+ y_direction * pdf(dist, x)
        plot!(x, y_offset, fillalpha=0.3, fillrange=y_offset_base, color=cart_colors[cart_idx])
        plot!([p_cart[1], p_cart[1]], [y_offset_base, y_offset_base .+ y_direction * pdf(dist, p_cart[1])], linestyle=:dash, color=cart_colors[cart_idx], linewidth=2)
    end
end

function plot_cart_box_and_com(p_carts::Vector{Vector{Float64}}, box_w::Float64, box_h::Float64, scale::Float64, y_offset::Float64, opacity::Float64, color::Vector{Symbol}, com_color::Symbol)
    for (cart_idx, p_cart) in enumerate(p_carts)
        plot!(
            rectangle(scale * box_w, scale * box_h, p_cart[1] - scale * 0.5 * box_w, p_cart[2] - scale * 0.5 * box_h),
            opacity = opacity, color = color[cart_idx], label = nothing
        )
        scatter!([p_cart[1]], [p_cart[2] + y_offset], color = com_color, label = nothing, opacity = opacity)
    end
end

function plot_system(sys::DoubleDampedSpringMassSystem, t::Int64;
                    box_l::Float64 = 1.0,
                    p_offset::Vector{Float64} = [0.0, 0.0],
                    y_offset_prediction::Float64 = 0.5,
                    y_offset_observation::Float64 = 0.0,
                    y_offset_goal::Float64 = -0.5,
                    scale_observations::Float64 = 0.5,
                    scale_prediction::Float64 = 0.5,
                    scale_goals::Float64 = 0.5,
                    opacity_observed::Float64 = 1.0,
                    opacity_pred::Float64 = 0.05,
                    opacity_goal::Float64 = 0.05,
                    gs::Union{Vector{Distributions.Normal{Float64}}, Nothing} = nothing,
                    pys::Union{Vector{Float64}, Nothing} = nothing,
                    vys::Union{Vector{Float64}, Nothing} = nothing,
                    n_std::Float64 = 1.0,
                    cart_colors::Vector{Symbol} = [:orange, :blue],
                    com_color::Symbol = :black,
                    cart_colors_goal::Vector{Symbol} = [:green, :lime],
                    x_limits::Tuple{Float64, Float64} = (-2.0, 4.0),
                    y_limits::Tuple{Float64, Float64} = (-1.0, 1.0),
                    xlabel::String="displacement",
                    ylabel::String="height",
                    prefix_title::String="")
    plt = plot(grid=true, title=@sprintf("%st = %.1f", prefix_title, t), xlabel=xlabel, ylabel=ylabel)
    box_w = 0.5 * box_l
    box_h = 0.5 * box_l
    t_real = t * sys.n_repeat
    x1, x2, dx1, dx2 = sys.zs[t_real]
    y1, y2 = sys.ys[t]

    # CART: observed position
    p_carts_observed = calculate_cart_positions([y1, y2], p_offset, y_offset_observation)
    plot_cart_box_and_com(p_carts_observed, box_w, box_h, scale_observations, y_offset_observation, opacity_observed, cart_colors, com_color)

    # CART: desired/goal
    if !isnothing(gs)
        p_carts_goal = calculate_cart_positions(mean.(gs), p_offset, y_offset_goal)
        plot_distributions(p_carts_goal, std.(gs), -0.25, -1, cart_colors_goal)
    end

    # CART: predictions
    if !isnothing(pys)
        p_carts_pred = calculate_cart_positions(pys, p_offset, y_offset_prediction)
        plot_distributions(p_carts_pred, sqrt.(vys), 0.25, 1, cart_colors)
    end

    xlims!(x_limits)
    ylims!(y_limits)
    w_pixel = 1600
    h_pixel = 900
    plot!(legend=:bottomright, size=(w_pixel, h_pixel))
    plot!(tickfontsize = DEFAULTS.tickfontsize, guidefontsize = DEFAULTS.guidefontsize, legendfontsize = 2*DEFAULTS.legendfontsize, titlefontsize = DEFAULTS.titlefontsize)

    return plt
end

function animate_trial(f_name::String, agents::Vector{NARXAgent}, sys::EnvironmentUnion; fps::Union{Int,Nothing}=nothing, prefix_title::String="")
    N = agents[1].t
    anim = @animate for k in 1:90
        plot_step(agents, sys, k, prefix_title=prefix_title)
    end
    if isnothing(fps) fps=get_fps(sys) end
    return mp4(anim, f_name, fps=fps, loop=0)
end

function plot_step(agents::Vector{NARXAgent}, sys::EnvironmentUnion, k::Int; plot_gs::Bool=true, prefix_title::String="")
    gs = plot_gs ? [ agent.gs[k][1] for agent in agents ] : nothing
    pys = [ agent.mys[k][1] for agent in agents ]
    vys = [ agent.vys[k][1] for agent in agents ]
    return plot_system(sys, k, gs=gs, pys=pys, vys=vys, prefix_title=prefix_title)
end
end # module
