### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d8924203-9760-40d5-8c46-7b6777834b95
begin
import Pkg
Pkg.activate(".")
using Revise
using PlutoUI
using CARX
using Plots; default(grid=false, label="", linewidth=3,margin=20Plots.pt)
using Random
if !isdir("figures") mkdir("figures") end
end

# ╔═╡ 0940f021-850a-44c7-ae01-94ef2058bc34
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(80px, 5%);
    	padding-right: max(80px, 5%);
	}
</style>
"""

# ╔═╡ 2223ab39-ea50-4874-bdee-bc0dbcedab23
@bind hyperparams CARX.Utils.ui_choices([
	Dict(:type => :Slider, :label => "σ_goal", :range => 0e1:1e-3:2e0, :default => 1e0),
	Dict(:type => :Slider, :label => "ubound", :range => 0.1:0.1:5.0, :default => 1.0),
	Dict(:type => :Slider, :label => "N", :range => 60:300, :default => 90),
	Dict(:type => :MultiCheckBox, :label => "EFE_terms", :options => ["EFE_CE_1", "EFE_CE_2", "EFE_MI", "EFE_control_prior"], :default => ["EFE_CE_1", "EFE_CE_2", "EFE_MI", "EFE_control_prior"]),
    Dict(:type => :Slider, :label => "n_repeat", :range => 2:300, :default => 120),
	Dict(:type => :Slider, :label => "Nu", :range => 1:1:2000, :default => 999),
]) # another good combo: ubound = 2.0, N = 100, n_repeat = 100

# ╔═╡ 80981c18-5666-11ef-3560-692ba1a92eaa
begin
	results = []
	for Mys in [[2,0], [2,2]]
	    Mus = [2, 0]
	    M = sum(Mys) + sum(Mus)
	    Random.seed!(1337)
	    kwargs = Dict(
	        :z0 => zeros(4),
	        :μ0 => 0.0,
	        :Λ0_y => 1.0,
	        :Λ0_u => 1.0,
	        :α0 => 2.0,
	        :β0 => 3.0,
	        :Mys => Mys,
	        :Mus => Mus,
	        :mnoise_S => 1e-5,
	        :σ_goal => hyperparams.σ_goal,
	        :ubound => hyperparams.ubound,
	        :spring => 1.0*ones(2),
	        :damping => 0.1*ones(2),
	        :N => hyperparams.N,
	        :EFE_terms => [label in hyperparams.EFE_terms for label in ["EFE_CE_1", "EFE_CE_2", "EFE_MI", "EFE_control_prior"]],
	        :use_RK4 => false, # false
	        :n_repeat => hyperparams.n_repeat, # 120
	        :Δt => 0.01,
	        :η0 => 1e-3,
	        :Nu => hyperparams.Nu # 300
	    )
	    agents, sys, performance = @time CARX.Experiments.run_experiment(;kwargs...)
	    push!(results, (agents, sys, performance))
	end
	agents_uncoupled, sys_uncoupled, performance_uncoupled = results[1]
	agents_coupled, sys_coupled, performance_coupled = results[2]
	@assert length(agents_coupled) == length(agents_uncoupled)
	n_agents = length(agents_coupled)
end

# ╔═╡ 8fe4e411-0e2b-4a49-8328-29fa34381a2a
realtime_xticks = CARX.SpringMassSystems.get_realtime_xticks(sys_coupled)

# ╔═╡ cea31e95-617f-49d8-817a-9ee4e5b9b732
CARX.Plotting.compare_agent_inference(agents_coupled, agents_uncoupled, realtime_xticks, field_name = :Pys, mode=:absolute)

# ╔═╡ b5db8e05-d549-4d3d-9354-06a303831edc
CARX.Plotting.compare_agent_inference(agents_coupled, agents_uncoupled, realtime_xticks, field_name = :Pgs, mode=:absolute)

# ╔═╡ 9533c56d-a4a5-4853-9855-ef50617c30a8
CARX.Plotting.compare_agent_inference(agents_coupled, agents_uncoupled, realtime_xticks, field_name = :Pys, mode=:difference)

# ╔═╡ 5575a53e-c395-4999-ba3e-6c6357c6e9f8
CARX.Plotting.compare_agent_inference(agents_coupled, agents_uncoupled, realtime_xticks, field_name = :Pgs, mode=:difference)

# ╔═╡ dc4edfe4-d2d6-454f-8dd6-84edb6657b4a
CARX.Plotting.plot_predictions_paper("figures/CARX-EFE-ddsms-predictions.png", agents_coupled, sys_coupled, agents_uncoupled, sys_uncoupled)

# ╔═╡ 47f68512-9d14-4c33-bc27-0200312c732d
CARX.Plotting.plot_controls_paper("figures/CARX-EFE-ddsms-controls.png", sys_coupled, sys_uncoupled)

# ╔═╡ 96d5a75d-91c9-4210-bfcc-462bfdfd336e
@bind params_plot CARX.Utils.ui_choices([
	Dict(:type => :Slider, :label => "t", :range => 1:hyperparams.N, :default => 1),
	#Dict(:type => :Slider, :label => "agent_idx", :range => 1:2, :default => 1),
])

# ╔═╡ 3c45960d-7021-4fde-994d-47a690be5cc8
CARX.Plotting.plot_step(agents_uncoupled, sys_uncoupled, params_plot.t, prefix_title="original ARX-EFE ")

# ╔═╡ 28ba5d11-5d5d-464e-8753-7b181ec9d568
CARX.Plotting.plot_step(agents_coupled, sys_coupled, params_plot.t, prefix_title="coupled ARX-EFE ")

# ╔═╡ 997f5805-a0e7-4720-b258-e53901399d6f
CARX.Plotting.animate_trial("results/uncoupled.mp4", agents_uncoupled, sys_uncoupled, fps=15, prefix_title="original ARX-EFE ")

# ╔═╡ fa142e34-961c-42ff-b819-1f21f0c7cd3b
CARX.Plotting.animate_trial("results/coupled.mp4", agents_coupled, sys_coupled, fps=15, prefix_title="coupled ARX-EFE ")

# ╔═╡ Cell order:
# ╟─0940f021-850a-44c7-ae01-94ef2058bc34
# ╟─d8924203-9760-40d5-8c46-7b6777834b95
# ╟─2223ab39-ea50-4874-bdee-bc0dbcedab23
# ╠═80981c18-5666-11ef-3560-692ba1a92eaa
# ╟─8fe4e411-0e2b-4a49-8328-29fa34381a2a
# ╠═cea31e95-617f-49d8-817a-9ee4e5b9b732
# ╟─b5db8e05-d549-4d3d-9354-06a303831edc
# ╟─9533c56d-a4a5-4853-9855-ef50617c30a8
# ╟─5575a53e-c395-4999-ba3e-6c6357c6e9f8
# ╟─dc4edfe4-d2d6-454f-8dd6-84edb6657b4a
# ╟─47f68512-9d14-4c33-bc27-0200312c732d
# ╟─96d5a75d-91c9-4210-bfcc-462bfdfd336e
# ╟─3c45960d-7021-4fde-994d-47a690be5cc8
# ╟─28ba5d11-5d5d-464e-8753-7b181ec9d568
# ╟─997f5805-a0e7-4720-b258-e53901399d6f
# ╟─fa142e34-961c-42ff-b819-1f21f0c7cd3b
