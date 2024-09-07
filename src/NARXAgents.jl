module NARXAgents

using Distributions
using SpecialFunctions # loggamma etc
using LinearAlgebra
using Printf # sprintf

#using ..SpringMassSystems # for has_nan checks
using ..location_scale_tdist # for LocationScaleT

function check_nan_or_inf(vector, errormsg::String)
    if ! has_nan_or_inf(vector) return end
    error("The vector $vector = $errormsg contains NaN or Inf.")
end

function has_nan_or_inf(vector)
    return any(x -> isnan(x) || isinf(x), vector)
end

export NARXAgent, update!, predict!, posterior_predictive, pol, crossentropy, mutualinfo, minimizeEFE!, backshift, EFE, set_goals!, create_memory_buffer

DEFAULTS = (
    μ0 = 1e-8::Float64, # coefficient prior mean
    Λ0_y = 1e-3::Float64, # coefficient prior precision for ybuffer
    Λ0_u = 1e-8::Float64, # coefficient prior precision for ubuffer
    α0 = 1e2::Float64, # parameter prior alpha. must be larger than 1.0
    β0 = 1e-1::Float64, # parameter prior beta
    η0 = 1e-3::Float64, # control prior precision
    Mys = [2, 2]::Vector{Int}, # memory size for each agent (first idx is for this agent), previous observations. 0 means exclude
    Mus = [2, 0]::Vector{Int}, # memory size for each agent (first idx is for this agent), previous controls. 0 means exclude
    pol_degree = 1::Int, # degree of polynomial
    zero_order = false::Bool, # polynomial includes zero order (memory buffer extends by one)
    time_horizon = 1::Int, # prediction horizon
    Nu = 300::Int, # agent: discrete control space to naively optimize over
    display_warning = false::Bool, # print some warning errors in some functions
    EFE_terms = [true, true, true, true]::Vector{Bool} # whether to disable any of CE_t1 + CE_t2 - MI + Up
)

mutable struct NARXAgent
    """
    Active inference agent based on a Nonlinear Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    ID              ::Integer
    ybuffer         ::Vector{Float64}
    ybuffer_other   ::Vector{Float64}
    ubuffer         ::Vector{Float64}
    ubuffer_other   ::Vector{Float64}

    M               ::Integer
    Mys             ::Vector{Int}
    Mus             ::Vector{Int}

    pol_degree      ::Integer
    zero_order      ::Bool
    order           ::Integer

    μ               ::Vector{Float64}   # Coefficients mean
    Λ               ::Matrix{Float64}   # Coefficients precision
    α               ::Float64           # Likelihood precision shape
    β               ::Float64           # Likelihood precision rate
    η               ::Float64           # Control prior precision

    μs              ::Vector{Vector{Float64}}
    Λs              ::Vector{Matrix{Float64}}
    αs              ::Vector{Float64}
    βs              ::Vector{Float64}
    ηs              ::Vector{Float64}

    gs              ::Vector{Vector{Normal{Float64}}}
    goals           ::Union{Vector{Normal{Float64}}, Nothing}

    time_horizon    ::Integer
    t               ::Integer

    Nu              ::Integer
    control_lims    ::Tuple{Float64, Float64}
    control_space   ::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}

    uis             ::Vector{Vector{CartesianIndex{2}}} # control indices

    free_energy     ::Float64
    EFEs            ::Vector{Matrix{Float64}}
    EFE_CEs         ::Vector{Matrix{Float64}}
    EFE_MIs         ::Vector{Matrix{Float64}}
    EFE_ups         ::Vector{Matrix{Float64}}

    EFE_CEs_t1      ::Vector{Matrix{Float64}}
    EFE_CEs_t2      ::Vector{Matrix{Float64}}

    EFE_terms       ::Vector{Bool}

    mysU            ::Vector{Matrix{Float64}}
    vysU            ::Vector{Matrix{Float64}}
    mys             ::Vector{Vector{Float64}}
    vys             ::Vector{Vector{Float64}}

    prederrorsU     ::Vector{Matrix{Float64}}
    prederrors      ::Vector{Float64}
    goalerrors      ::Vector{Float64}

    ϕus             ::Vector{Matrix{Float64}}
    ϕys             ::Vector{Vector{Float64}}

    Pys             ::Vector{Float64} # p(ŷₖ | ûₖ)
    Pgs             ::Vector{Float64} # p(ŷₖ | gₖ)

    display_warning ::Bool

    function NARXAgent(
        ID::Int,
        control_lims::Tuple{Float64, Float64};
        μ0::Float64=DEFAULTS.μ0, # coefficients mean, diagonal value
        Λ0_y::Float64=DEFAULTS.Λ0_y, # coefficients_precision, diagonal value, observations
        Λ0_u::Float64=DEFAULTS.Λ0_u, # coefficients_precision, diagonal value, controls
        α0::Float64=DEFAULTS.α0, # noise_shape
        β0::Float64=DEFAULTS.β0, # noise_rate
        η0::Float64=DEFAULTS.η0, # control prior precision
        Mys::Vector{Int}=DEFAULTS.Mys,
        Mus::Vector{Int}=DEFAULTS.Mus,
        pol_degree::Integer=DEFAULTS.pol_degree,
        zero_order::Bool=DEFAULTS.zero_order,
        time_horizon::Integer=DEFAULTS.time_horizon,
        Nu::Integer=DEFAULTS.Nu, # number of control states in control space
        display_warning::Bool=DEFAULTS.display_warning,
        EFE_terms::Vector{Bool}=DEFAULTS.EFE_terms # whether to disable any of CE_t1 + CE_t2 - MI + Up
    )
        #println("$μ0 $Λ0_y $Λ0_u $α0 $β0 $η0 $Mys $Mus $pol_degree $zero_order $zero_order $Nu $display_warning")
        ubuffer = zeros(Mus[1])
        ybuffer = zeros(Mys[1])
        ubuffer_other = zeros(Mus[2])
        ybuffer_other = zeros(Mys[2])
        # time horizon is already included in the ubuffer

        M = sum(Mys) + sum(Mus)
        if zero_order M += 1 end

        μ0 = μ0*ones(M)
        Λ0 = diagm([Λ0_y*ones(sum(Mys)); Λ0_u*ones(sum(Mus))])
        #println("$μ0\n$Λ0")

        order = size(pol(zeros(M), degree=pol_degree, zero_order=zero_order),1)
        lcoeff = length(μ0)
        if order != lcoeff
            error("Dimensionality of coefficients $lcoeff and model order $order do not match.")
        end

        free_energy = Inf
        EFEs = Vector{Matrix{Float64}}()
        EFE_CEs = Vector{Matrix{Float64}}()
        EFE_MIs = Vector{Matrix{Float64}}()
        EFE_ups = Vector{Matrix{Float64}}()

        EFE_CEs_t1 = Vector{Matrix{Float64}}()
        EFE_CEs_t2 = Vector{Matrix{Float64}}()

        mysU = Vector{Matrix{Float64}}()
        vysU = Vector{Matrix{Float64}}()
        prederrorsU = Vector{Matrix{Float64}}()

        mys = Vector{Vector{Float64}}()
        vys = Vector{Vector{Float64}}()
        prederrors = Vector{Float64}()
        goalerrors = Vector{Float64}()

        ϕus = Vector{Matrix{Float64}}()
        ϕys = Vector{Vector{Float64}}()

        t = 0
        μs = Vector{Vector{Float64}}()
        Λs = Vector{Matrix{Float64}}()
        αs = Vector{Float64}()
        βs = Vector{Float64}()
        ηs = Vector{Float64}()
        push!(μs, μ0)
        push!(Λs, Λ0)
        push!(αs, α0)
        push!(βs, β0)
        push!(ηs, η0)

        gs = Vector{Vector{Normal{Float64}}}()

        control_space = range(control_lims[1], stop=control_lims[2], length=Nu)

        uis = Vector{Vector{CartesianIndex{2}}}()

        Pys = Vector{Float64}()
        Pgs = Vector{Float64}()

        return new(ID, ybuffer, ybuffer_other,
                   ubuffer, ubuffer_other,
                   M, Mys, Mus,
                   pol_degree,
                   zero_order,
                   order,
                   μ0, Λ0, α0, β0, η0,
                   μs, Λs, αs, βs, ηs,
                   gs, nothing,
                   time_horizon,
                   t,
                   Nu, control_lims, control_space,
                   uis,
                   free_energy, EFEs, EFE_CEs, EFE_MIs, EFE_ups, EFE_CEs_t1, EFE_CEs_t2, EFE_terms,
                   mysU, vysU, mys, vys, prederrorsU, prederrors, goalerrors,
                   ϕus, ϕys, Pys, Pgs,
                   display_warning
                   )
    end
end

function pol(x; degree::Integer = 1, zero_order=true)
    if zero_order
        return cat([1.0; [x.^d for d in 1:degree]]...,dims=1)
    else
        return cat([x.^d for d in 1:degree]...,dims=1)
    end
end

function create_memory_buffer(agent::NARXAgent, ybuffer::Vector{Float64}, ybuffer_other::Vector{Float64}, ubuffer::Vector{Float64}, ubuffer_other::Vector{Float64})
    memory_buffer = Vector{Float64}()
    push!(memory_buffer, ybuffer...)
    if agent.Mys[2] > 0
        push!(memory_buffer, ybuffer_other...)
    end
    push!(memory_buffer, ubuffer...)
    if agent.Mus[2] > 0
        push!(memory_buffer, ubuffer_other...)
    end
    return memory_buffer
end

# for a coupled NARXAgent
function update!(agent::NARXAgent, y::Float64, y_other::Float64, u::Float64, u_other::Float64, goals::Vector{Normal{Float64}})
    do_debug_print = false
    if do_debug_print; println("update! [ ] agent.ubuffer: ", agent.ubuffer) end
    if agent.Mus[1] > 0
        agent.ubuffer = backshift(agent.ubuffer, u)
    end
    if do_debug_print
        println("update! [X] agent.ubuffer: ", agent.ubuffer)
        println("update! [ ] agent.ubuffer_other: ", agent.ubuffer_other)
    end
    if agent.Mus[2] > 0
        agent.ubuffer_other = backshift(agent.ubuffer_other, u_other)
    end
    if do_debug_print; println("update! [X] agent.ubuffer_other: ", agent.ubuffer_other) end

    if !isnothing(agent.goals)
        my = agent.mys[agent.t][1]

        # goalerror: distance prediction vs goal prior mean
        μg = mean(agent.goals[1])
        goalerror = (my - μg)^2
        push!(agent.goalerrors, goalerror)

        # prederror: distance prediction vs observation
        prederror = (my - y)^2
        push!(agent.prederrors, prederror)
    end

    agent.goals = goals
    push!(agent.gs, goals)

    memory_buffer = create_memory_buffer(agent, agent.ybuffer, agent.ybuffer_other, agent.ubuffer, agent.ubuffer_other)
    ϕ = pol(memory_buffer, degree=agent.pol_degree, zero_order=agent.zero_order)


    μ0 = agent.μ
    Λ0 = agent.Λ
    α0 = agent.α
    β0 = agent.β

    # Before the parameter update, we will record the likelihood of the prediction given the executed control
    ν_t, m_t, s2_t = posterior_predictive(agent, ϕ)
    s_t = sqrt(s2_t)
    dist = LocationScaleT(ν_t, m_t, s_t)
    py = location_scale_tdist.pdf(dist, y)
    #println("$dist $prob_y")
    push!(agent.Pys, py)

    pg = Distributions.pdf(agent.goals[1], y)
    #println(agent.goals[1], " $y $pg") # TODO: deal with y values that yield densities that exceed floatmin(Float64)
    push!(agent.Pgs, pg)

    agent.μ = inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0)
    agent.Λ = ϕ*ϕ' + Λ0
    agent.α = α0 + 1/2
    agent.β = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - (ϕ*y + Λ0*μ0)'*inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0))
    β_test = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - agent.μ'*agent.Λ*agent.μ)
    E = agent.Λ*agent.μ
    β_test_B = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - E'*inv(agent.Λ)*E)

    #println("β - β_test ", agent.β - β_test)
    #println("β - β_test_B ", agent.β - β_test_B)
    agent.β = β_test

    push!(agent.μs, agent.μ)
    push!(agent.Λs, agent.Λ)
    push!(agent.αs, agent.α)
    push!(agent.βs, agent.β)
    push!(agent.ηs, agent.η)

    if do_debug_print; println("update! [ ] agent.ybuffer: ", agent.ybuffer) end
    if agent.Mys[1] > 0
        agent.ybuffer = backshift(agent.ybuffer, y)
    end
    if do_debug_print
        println("update! [X] agent.ybuffer: ", agent.ybuffer)
        println("update! [ ] agent.ybuffer_other: ", agent.ybuffer_other)
    end
    if agent.Mys[2] > 0
        agent.ybuffer_other = backshift(agent.ybuffer_other, y_other)
    end
    if do_debug_print; println("update! [X] agent.ybuffer_other: ", agent.ybuffer_other) end

    #agent.free_energy = -log_marginal_likelihood(agent, (μ0, Λ0, α0, β0))
    agent.t += 1
end

function params(agent::NARXAgent)
    return agent.μ, agent.Λ, agent.α, agent.β
end

function marginal_likelihood(agent::NARXAgent, prior_params)
    μn, Λn, αn, βn = params(agent)
    μ0, Λ0, α0, β0 = prior_params

    return (det(Λn)^(-1/2)*gamma(αn)*βn^αn)/(det(Λ0)^(-1/2)*gamma(α0)*β0^α0) * (2π)^(-1/2)
end

function log_marginal_likelihood(agent::NARXAgent, prior_params)
    μn, Λn, αn, βn = params(agent)
    μ0, Λ0, α0, β0 = prior_params

    return -1/2*logdet(Λn) + log(gamma(αn)) + αn*log(βn) -(-1/2*logdet(Λ0) +log(gamma(α0)) + α0*log(β0)) -1/2*log(2π)
end

function posterior_predictive(agent::NARXAgent, ϕ_t::Vector{Float64})
    "Posterior predictive distribution is location-scale t-distributed"

    ν_t = 2*agent.α
    m_t = dot(agent.μ, ϕ_t)
    s2_t = (agent.β/agent.α)*(1 + ϕ_t'*inv(agent.Λ)*ϕ_t)

    return ν_t, m_t, s2_t
end

function predict!(agent::NARXAgent, controls, controls_other)
    m_y = zeros(agent.time_horizon)
    v_y = zeros(agent.time_horizon)
    ϕys = zeros(agent.time_horizon, agent.M)

    ybuffer = agent.ybuffer
    ybuffer_other = agent.ybuffer_other
    ubuffer = agent.ubuffer
    ubuffer_other = agent.ubuffer_other
    for t in 1:agent.time_horizon
        # Update control buffer
        if agent.Mus[1] > 0
            ubuffer = backshift(ubuffer, controls[t])
        end
        if agent.Mus[2] > 0
            ubuffer_other = backshift(ubuffer_other, controls_other[t])
        end
        memory_buffer = create_memory_buffer(agent, ybuffer, ybuffer_other, ubuffer, ubuffer_other)
        ϕ_t = pol(memory_buffer, degree=agent.pol_degree, zero_order=agent.zero_order)
        ϕys[t,:] = ϕ_t

        ν_t, m_t, s2_t = posterior_predictive(agent, ϕ_t)

        # Prediction
        m_y[t] = m_t
        v_y[t] = s2_t * ν_t/(ν_t - 2)

        # Update previous
        if agent.Mys[1] > 0
            ybuffer = backshift(ybuffer, m_y[t])
        end
        if agent.Mys[2] > 0
            ybuffer_other = backshift(ybuffer_other, 0.0) # XXX this has to change with multiple agents and agent.time_horizon > 1
        end
    end
    push!(agent.mys, m_y)
    push!(agent.vys, v_y)
    #push!(agent.ϕys, ϕys)
    return m_y, v_y
end

function mutualinfo(agent::NARXAgent, ϕ_t)
    "Mutual information between parameters and posterior predictive (constant terms dropped)"
    return 1/2*log(1 + ϕ_t'*inv(agent.Λ)*ϕ_t)
end

function crossentropy(agent::NARXAgent, goal::Distribution{Univariate, Continuous}, m_pred, v_pred)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"
    return ( v_pred + (m_pred - mean(goal))^2 ) / ( 2var(goal) )
    # return (m_pred - mean(goal))^2/(2var(goal))
end

function EFE(agent::NARXAgent, controls::Float64)
    "Expected Free Energy"

    ybuffer = agent.ybuffer
    ybuffer_other = agent.ybuffer_other
    ubuffer = agent.ubuffer
    ubuffer_other = agent.ubuffer_other

    J_CEs = Vector{Float64}()
    J_CEs_t1 = Vector{Float64}()
    J_CEs_t2 = Vector{Float64}()
    J_EFEs = Vector{Float64}()
    J_MIs = Vector{Float64}()
    J_ups = Vector{Float64}()
    mys = Vector{Float64}()
    vys = Vector{Float64}()
    ϕus = Vector{Vector{Float64}}()
    prederrors = Vector{Float64}()
    J = 0
    for t in 1:agent.time_horizon
        # Update control buffer
        if agent.Mus[1] > 0
            ubuffer = backshift(ubuffer, controls[t])
        end
        if agent.Mus[2] > 0
            ubuffer_other = backshift(ubuffer_other, 0.0)
        end
        memory_buffer = create_memory_buffer(agent, ybuffer, ybuffer_other, ubuffer, ubuffer_other)
        fmt_string = @sprintf("[%03d] [%02d] [%01d]", agent.t, t, agent.ID)
        str_array = [@sprintf("%.3f", x) for x in memory_buffer]
        fmt_string = "$fmt_string [" * join(str_array, ", ") * "] $controls"
        #println(fmt_string)
        ϕ_t = pol(memory_buffer, degree=agent.pol_degree, zero_order=agent.zero_order)
        check_nan_or_inf(ϕ_t, "ϕ_t = $ϕ_t")
        # TODO: change ubuffer and ubuffer_other for n_agents > 1 and/or agent.time_horizon > 1
        #ϕ_t = pol([ybuffer; ybuffer_other; ubuffer; ubuffer_other], degree=agent.pol_degree, zero_order=agent.zero_order)
        push!(ϕus, ϕ_t)

        # Prediction
        ν_t, m_t, s2_t = posterior_predictive(agent, ϕ_t)

        m_y = m_t
        v_y = s2_t * ν_t/(ν_t - 2)

        push!(mys, m_y)
        push!(vys, v_y)

        m_g = mean(agent.goals[t])
        v_g = var(agent.goals[t])

        r2_t = ϕ_t'*inv(agent.Λ)*ϕ_t + 1

        J_MI = log(r2_t)

        J_CE_t1 = (m_y - m_g)^2
        J_CE_t1 /= v_g
        J_CE_t2 = (agent.β/(agent.α - 1)) * r2_t
        J_CE_t2 /= v_g

        J_CE = J_CE_t1 + J_CE_t2 # (J_CE_t1 + J_CE_t2)/v_g


        # Accumulate EFE
        prederror = (m_y - mean(agent.goals[t]))^2
        push!(prederrors, prederror)
        J_up = agent.η/2*controls[t]^2


        if agent.EFE_terms[1]
            J += J_CE_t1
        end
        if agent.EFE_terms[2]
            J += J_CE_t2
        end

        if agent.EFE_terms[3]
            J -= J_MI
        end

        if agent.EFE_terms[4]
            J += J_up
        end

        push!(J_EFEs, J)
        push!(J_CEs, J_CE)
        push!(J_CEs_t1, J_CE_t1)
        push!(J_CEs_t2, J_CE_t2)
        push!(J_MIs, J_MI)
        push!(J_ups, J_up)

        # Update previous observations
        if agent.Mys[1] > 0
            ybuffer = backshift(ybuffer, m_y)
        end
        if agent.Mys[2] > 0
            ybuffer_other = backshift(ybuffer_other, 0.0) # TODO: change for n_agents > 1 and/or agent.time_horizon > 1
        end
    end
    return J, J_EFEs, J_CEs, J_MIs, J_ups, mys, vys, prederrors, J_CEs_t1, J_CEs_t2
end

function minimizeEFE!(agent::NARXAgent; u_0=nothing, time_limit=10.0, verbose=false, show_every=10, f_tol=1e-8, x_tol=1e-8, g_tol=1e-8)
    "Minimize EFE objective and return policy."
    if isnothing(u_0); u_0 = zeros(agent.time_horizon); end

    # Objective function
    J(u) = EFE(agent, u)

    J_uu = zeros(agent.Nu, agent.time_horizon)
    J_EFE = zeros(agent.Nu, agent.time_horizon)
    J_CE = zeros(agent.Nu, agent.time_horizon)
    J_CE_t1 = zeros(agent.Nu, agent.time_horizon)
    J_CE_t2 = zeros(agent.Nu, agent.time_horizon)
    J_MI = zeros(agent.Nu, agent.time_horizon)
    J_up = zeros(agent.Nu, agent.time_horizon)
    mysU = zeros(agent.Nu, agent.time_horizon)
    vysU = zeros(agent.Nu, agent.time_horizon)
    prederrorsU = zeros(agent.Nu, agent.time_horizon)
    for (i, u_) in enumerate(agent.control_space)
        J_uu[i], J_EFE[i,:], J_CE[i,:], J_MI[i,:], J_up[i,:], mysU[i,:], vysU[i,:], prederrorsU[i,:], J_CE_t1[i,:], J_CE_t2[i,:] = J(u_)
    end

    push!(agent.EFEs, J_EFE)
    push!(agent.EFE_CEs, J_CE)
    push!(agent.EFE_CEs_t1, J_CE_t1)
    push!(agent.EFE_CEs_t2, J_CE_t2)
    push!(agent.EFE_MIs, J_MI)
    push!(agent.EFE_ups, J_up)
    push!(agent.mysU, mysU)
    push!(agent.vysU, vysU)
    push!(agent.prederrorsU, prederrorsU)

    min_indices = findall(J_uu .== minimum(J_uu))
    if length(min_indices) == 0
        error("ERROR: Cannot find a min_index via naive search $min_indices for $J_uu")
    end
    u_naive = agent.control_space[min_indices]
    ui = min_indices
    if length(min_indices) > 1
        min_indices = [min_indices[1]]
        if agent.display_warning println("WARNING: Reducing u_naive $u_naive to one element") end
        u_naive = agent.control_space[min_indices]
        ui = min_indices
    end
    push!(agent.uis, ui)

    return u_naive
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

end
