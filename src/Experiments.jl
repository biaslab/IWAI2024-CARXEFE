module Experiments

using JLD # @save macro
#using SpecialFunctions
using LinearAlgebra
using LaTeXStrings
using Distributions
using Random; Random.seed!(123)
#using Base64

using ..NARXAgents
using ..SpringMassSystems
using ..Utils

export run_experiment

EnvironmentUnion = Union{SpringMassSystems.DoubleDampedSpringMassSystem}

const DEFAULTS = (
    interaction = (
        #N = 200::Int, # interaction: number of episode steps
        n_agents = 2::Int, # agent: number of agents
        # run_experiment and evaluate_parameter parameters
        do_debug_print = false::Bool, # print some debug information in all functions
        # run_experiment parameters
        save_results = false::Bool, # save the results
        dynamic_goals = false::Bool, # change the goal prior mean over time
        decaying_goals = false::Bool, # decay the goal prior variance over time
        override_controls = nothing::Union{Vector{Vector{Float64}}, Nothing},
        # evaluate_parameter parameters
        Ne = 5::Int, # number of evaluations/experiments per parameter,
    ),
    agent = NARXAgents.DEFAULTS
)

function merge_tuples(tupA, tupB)
    return NamedTuple{keys(tupA)}((haskey(tupB, k) ? tupB[k] : tupA[k] for k in keys(tupA)))
end

function run_experiment(;kwargs...)
    DEFAULTS_environment = SpringMassSystems.DEFAULTS_SYS.DoubleDampedSpringMassSystem
    kwargs_interaction = merge_tuples(DEFAULTS.interaction, kwargs)
    kwargs_environment = merge_tuples(DEFAULTS_environment, kwargs)
    kwargs_agent = merge_tuples(DEFAULTS.agent, kwargs)
    @assert 0 < kwargs_interaction.n_agents <= 2 "Cannot handle more than 2 or less than 1 agents. How do you assign observations and policies of the other agent to the update! function?"

    u0_zeros = zeros(kwargs_agent.time_horizon)

    # Start system
    environment = DoubleDampedSpringMassSystem(;kwargs_environment...)

    # Start agent
    agents = Vector{NARXAgent}()
    for agent_idx in 1:kwargs_interaction.n_agents
        agent = NARXAgent(agent_idx, environment.torque_lims; kwargs_agent...)
        push!(agents, agent)
    end

    policy = zeros(kwargs_interaction.n_agents, kwargs_agent.time_horizon) # for each agent, a policy for the current time step includes T controls
    for k in 1:kwargs_environment.N
        y_cur = environment.sensor

        for (agent_idx, agent) in enumerate(agents)
            y_agent = y_cur[agent_idx]

            # Set goal priors for time horizon
            μ_goal = kwargs_environment.μ_goal[agent_idx]
            if kwargs_interaction.dynamic_goals
                μ_goal += floor(y_agent/(2π)) * 2π
            end
            σ_goal = kwargs_environment.σ_goal
            if kwargs_interaction.decaying_goals
                σ_goal_start = 1.5
                σ_goal_end = 0.1
                σ_goal = exponential_decay_step(σ_goal_start, σ_goal_end, k, kwargs_environment.N)
            end
            goals = [Normal(μ_goal, σ_goal) for t in 1:kwargs_agent.time_horizon]
            agent_idx_other = agent_idx % kwargs_interaction.n_agents + 1
            NARXAgents.update!(agent, y_cur[agent_idx], y_cur[agent_idx_other], policy[agent_idx], policy[agent_idx_other], goals)

            u0_noisy = 0.01*rand(kwargs_agent.time_horizon)
            policy[agent_idx,:] = minimizeEFE!(agent, u_0=u0_zeros, time_limit=30., verbose=false, f_tol=1e-8, x_tol=1e-8, g_tol=1e-32)
            if kwargs_interaction.do_debug_print; println("[$k, $agent_idx] mcu = $(policy[agent_idx,:])") end
            if ! isnothing(kwargs_interaction.override_controls)
                policy[agent_idx, :] = [ kwargs_interaction.override_controls[agent_idx][k] for _ in 1:agent.time_horizon]
            end
        end

        for (agent_idx, agent) in enumerate(agents)
            predict!(agent, policy[agent_idx,:], policy[agent_idx%2+1,:])
        end

        # Act upon environment
        SpringMassSystems.step!(environment, policy[:,1]) # execute the first action of all agents
    end

    metrics = Dict(
        :goalerrors => [ sqrt(mean(agent.goalerrors)) for agent in agents], # sqrt(mean((y_pred - μ_goal)^2))
        :prederrors => [ sqrt(mean(agent.prederrors)) for agent in agents], # sqrt(mean((y_pred - y_true)^2))
        :return => real_return(environment, agents), # sum(z_real - μ_goal)
        :avg_τs => [mean(agent.αs ./ agent.βs) for agent in agents],
        :avg_vys => [mean(hcat(agent.vys...)[1,:]) for agent in agents]
    )

    #check_numerical_instabilities(n_agents, agents)
    d_name = "results/"
    f_name = generate_filename_with_time(prefix="$d_name/", suffix=".jld")
    if !isdir(d_name) mkdir(d_name) end
    if kwargs_interaction.save_results @save f_name environment agents end
    return agents, environment, metrics
end
end # module
