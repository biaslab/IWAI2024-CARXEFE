module SpringMassSystems

using LinearAlgebra
using Printf # sprintf

export DEFAULTS_SYS, SpringMassSystem, DoubleDampedSpringMassSystem, dzdt, step!, get_realtime_xticks

abstract type SpringMassSystem end

# TODO: allow σ_goal to be a vector (ability to choose separate σ_goal for each dimension) which currently breaks sweeping
# TODO: allow mnoise_S to be a vector (ability to choose separate σ_goal for each dimension) which currently breaks sweeping
const DEFAULTS_SYS = (
    DoubleDampedSpringMassSystem = (
        μ_goal = [1.0, 2.0]::Vector{Float64}, # default goals for the agent
        σ_goal = 0.1::Float64, # agent: default goal prior variance
        z0 = [0.0, 0.0, 0.0, 0.0]::Vector{Float64}, # initial (hidden) state: x1, x2, dx1, dx2. wall should be at (0,0)
        ubound = 0.1::Float64, # bounds for the control limits
        mass = ones(2)::Vector{Float64}, # ∀ cart: mass
        damping = 0.1*ones(2)::Vector{Float64}, # ∀ cart: damping coefficient
        spring = 1.0*ones(2)::Vector{Float64}, # ∀ cart: spring coefficient
        mnoise_S = 1e-5::Float64, # measurement noise std matrix
        N = 500::Int, # number of episode steps
        Δt = 0.01::Float64, # time step duration in s
        n_repeat = 100::Int, # how many actions to repeat (in case of low Δt),
        use_RK4 = true::Bool
    ),
)

mutable struct DoubleDampedSpringMassSystem <: SpringMassSystem
    μ_goal      ::Vector{Float64} # default goals for the agent
    σ_goal      ::Float64 # agent: default goal prior variance
    state       ::Vector{Float64}
    sensor      ::Vector{Float64}
    torque      ::Vector{Float64}
    torque_lims ::Tuple{Float64,Float64}
    N           ::Integer
    Δt          ::Float64
    mass        ::Vector{Float64}
    damping     ::Vector{Float64} # [c1, c2]
    spring      ::Vector{Float64} # [k1, k2]
    mnoise_S    ::Matrix{Float64}
    zs          ::Vector{Vector{Float64}} # hidden states of the environment
    ys          ::Vector{Vector{Float64}} # observations emitted to the agent(s)
    us          ::Vector{Vector{Float64}} # actual controls (torques)
    as          ::Vector{Vector{Float64}} # intent controls (agent actions)
    step        ::Integer
    n_output    ::Integer
    n_input     ::Integer
    RKslopes    ::Vector{Vector{Vector{Float64}}}
    n_repeat    ::Integer # how many actions to repeat
    use_RK4     ::Bool

    function DoubleDampedSpringMassSystem(;
        μ_goal::Vector{Float64}=DEFAULTS_SYS.DoubleDampedSpringMassSystem.μ_goal,
        σ_goal::Float64=DEFAULTS_SYS.DoubleDampedSpringMassSystem.σ_goal,
        z0::Vector{Float64}=DEFAULTS_SYS.DoubleDampedSpringMassSystem.z0,
        ubound::Float64=DEFAULTS_SYS.DoubleDampedSpringMassSystem.ubound,
        mass::Vector{Float64}=DEFAULTS_SYS.DoubleDampedSpringMassSystem.mass,
        damping::Vector{Float64}=DEFAULTS_SYS.DoubleDampedSpringMassSystem.damping,
        spring::Vector{Float64}=DEFAULTS_SYS.DoubleDampedSpringMassSystem.spring,
        mnoise_S::Float64=DEFAULTS_SYS.DoubleDampedSpringMassSystem.mnoise_S,
        N::Int=DEFAULTS_SYS.DoubleDampedSpringMassSystem.N,
        Δt::Float64=DEFAULTS_SYS.DoubleDampedSpringMassSystem.Δt,
        n_repeat::Int=DEFAULTS_SYS.DoubleDampedSpringMassSystem.n_repeat,
        use_RK4::Bool=DEFAULTS_SYS.DoubleDampedSpringMassSystem.use_RK4
    )

        #println("$damping $spring")
        zs = Vector{Vector{Float64}}() # ∀ cart: (x, dx)
        ys = Vector{Vector{Float64}}() # ∀ cart: x + noise
        us = Vector{Vector{Float64}}()
        as = Vector{Vector{Float64}}()
        torque_lims = (-ubound, ubound)
        n_output = 2 # dimensionality of observation space
        n_input = 2 # dimensionality of control space
        mnoise_S = mnoise_S*diagm(ones(2))
        init_sensor = z0[1:2] + cholesky(mnoise_S).L*randn(2)
        RKslopes = Vector{Vector{Vector{Float64}}}()
        t = 1
        torque = zeros(2)
        return new(μ_goal, σ_goal, z0, init_sensor, torque, torque_lims, N, Δt, mass, damping, spring, mnoise_S, zs, ys, us, as, t, n_output, n_input, RKslopes, n_repeat, use_RK4)
    end
end

function check_nan_or_inf(vector, errormsg::String)
    if ! has_nan_or_inf(vector) return end
    error("The vector $vector = $errormsg contains NaN or Inf.")
end

function has_nan_or_inf(vector)
    return any(x -> isnan(x) || isinf(x), vector)
end

# https://www.youtube.com/watch?v=NV7cd-7Rz-I
# http://www.dem.ist.utl.pt/engopt2010/Book_and_CD/Papers_CD_Final_Version/pdf/02/01124-01.pdf
function dzdt_RK4(sys::DoubleDampedSpringMassSystem, u::Vector{Float64}; Δstate::Vector=zeros(4))
    check_nan_or_inf(sys.state, "sys.state: $(sys.state)")
    check_nan_or_inf(Δstate, "Δstate: $(Δstate)")

    x1, x2, dx1, dx2 = sys.state + Δstate
    c1, c2 = sys.damping
    k1, k2 = sys.spring
    m1, m2 = sys.mass

    # Equations of motion
    ddx1 = -(c1+c2)*dx1 + c2*dx2 - (k1+k2)*x1 + k2*x2 + u[1]
    ddx2 =       c2*dx1 - c2*dx2 +     k2 *x1 - k2*x2 + u[2]

    # Mass matrix (inertia matrix)
    M = [m1 0.0; 0.0 m2]
    ddx1, ddx2 = inv(M)*[ddx1, ddx2]

    return [dx1; dx2; ddx1; ddx2]
end

function dzdt_secondorder(sys::DoubleDampedSpringMassSystem, u::Vector{Float64})
    check_nan_or_inf(sys.state, "sys.state: $(sys.state)")

    x1, x2, dx1, dx2 = sys.state # + Δstate

    debug_print = false
    c1, c2 = sys.damping
    k1, k2 = sys.spring
    m1, m2 = sys.mass
    if debug_print
        println("z $(sys.state)")
        println("c $c1 $c2")
        println("k $k1 $k2")
        println("m $m1 $m2")
    end

    # Equations of motion
    ddx1 = -(c1+c2)*dx1 + c2*dx2 - (k1+k2)*x1 + k2*x2 + u[1]
    ddx2 =       c2*dx1 - c2*dx2 +     k2 *x1 - k2*x2 + u[2]
    if debug_print println("$ddx1 $ddx2") end

    # Mass matrix (inertia matrix)
    M = [m1 0.0; 0.0 m2]
    ddx1, ddx2 = inv(M)*[ddx1, ddx2]

    # Control vector
    B = sys.Δt*[0.5*sys.Δt/m1            0.0;
                0.0            0.5*sys.Δt/m2;
                1/m1                     0.0;
                0.0                     1/m2]

    return sys.Δt*[dx1; dx2; ddx1; ddx2] + B*u
end

function step!(sys::SpringMassSystem, u::Vector{Float64})
    for n in 1:sys.n_repeat
        update!(sys, u)
    end
    sys.step += 1
    emit!(sys)
end

function update!(sys::SpringMassSystem, u::Vector{Float64})
    push!(sys.as, u)
    sys.torque = clamp.(u, sys.torque_lims...)
    push!(sys.us, sys.torque)

    if sys.use_RK4
        Δstate, RKslopes = RK4(sys, sys.torque)
        push!(sys.RKslopes, RKslopes)
    else
        Δstate = dzdt_secondorder(sys, sys.torque)
    end
    sys.state += Δstate
    push!(sys.zs, sys.state)
end

function emit!(sys::SpringMassSystem)
    sys.sensor = sys.state[1:2] + cholesky(sys.mnoise_S).L * randn(2)
    push!(sys.ys, sys.sensor)
end

function RK4(sys::DoubleDampedSpringMassSystem, u::Vector{Float64})
    K1 = dzdt_RK4(sys, u)
    K2 = dzdt_RK4(sys, u, Δstate=K1*sys.Δt/2)
    K3 = dzdt_RK4(sys, u, Δstate=K2*sys.Δt/2)
    K4 = dzdt_RK4(sys, u, Δstate=K3*sys.Δt  )
    # increment of z is the weighted sum of all derivatives
    Δz = sys.Δt/6 * (K1 + 2K2 + 2K3 + K4)
    return Δz, [K1, K2, K3, K4]
end

function get_realtime_xticks(sys::SpringMassSystem)
    xticks_pos = collect(0:(sys.N/5):sys.N) .* (sys.Δt * sys.n_repeat)
    xticks_labels = string.(Int.(round.(xticks_pos)))
    return xticks_pos, xticks_labels
end

end # module
