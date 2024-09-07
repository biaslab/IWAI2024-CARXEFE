module Utils

using ..NARXAgents
using ..SpringMassSystems
using Dates # now(), Dates.format
using Distributions # Normal

using Markdown # md macro/string
using PlutoUI

EnvironmentUnion = Union{DoubleDampedSpringMassSystem}

export check_numerical_instabilities, create_initial_structures, exponential_decay, exponential_decay_step, generate_filename_with_time, rewards, real_return
export prederror, pretty_print_named_tuple
export ui_choices, set_margins

function check_numerical_instabilities(n_agents::Int, agents::Vector{NARXAgent})
    for agent_idx in 1:n_agents
        println((any(isnan.(agents[agent_idx].μ)), any(isnan.(agents[agent_idx].Λ)), any(isnan.(agents[agent_idx].α)), any(isnan.(agents[agent_idx].β))))
    end
end

function exponential_decay(start_value, end_value, num_steps)
    λ = log(end_value / start_value) / (num_steps - 1)
    decayed_values = [start_value * exp(λ * i) for i in 0:num_steps-1]
    return decayed_values
end

function exponential_decay_step(start_value, end_value, current_step, total_steps)
    λ = log(end_value / start_value) / (total_steps - 1)
    decayed_value = start_value * exp(λ * current_step)
    return decayed_value
end

function generate_filename_with_time(;prefix::String="", suffix::String="")
    current_time = now()
    formatted_time = Dates.format(current_time, "dd-HH-MM-SS")
    filename = "$prefix$formatted_time$suffix"
    return filename
end

function rewards(sys::EnvironmentUnion, agents::Vector{NARXAgent})
    rewards = []
    for (agent_idx, agent) in enumerate(agents)
        zsreal = hcat(sys.zs...)[agent_idx, :]
        zsreal = [zsreal[i] for i in 1:sys.n_repeat:length(zsreal)]
        cost = sqrt.((zsreal - mean.(hcat(agent.gs...)[1,:])).^2)
        reward = -mean(cost)
        push!(rewards, reward)
    end
    return rewards
end

function real_return(sys::EnvironmentUnion, agents::Vector{NARXAgent})
    return sum(rewards(sys, agents))
end

function prederror(sys::EnvironmentUnion, agents::Vector{NARXAgent})
    prederrors = Vector{Float64}()
    for (agent_idx, agent) in enumerate(agents)
        agent = agents[agent_idx]
        py = hcat(agent.mys...)[1,:]
        ry = hcat(sys.ys...)[agent_idx, :]
        residuals = (py .- ry).^2
        push!(prederrors, sqrt(mean(residuals)))
    end
    return prederrors
end

function pretty_print_named_tuple(tup::NamedTuple, indent::Int = 0)
    indent_str = "  "^indent
    println(indent_str * "{")
    for (key, value) in pairs(tup)
        if isa(value, NamedTuple)
            println(indent_str * "  $key: ")
            pretty_print_named_tuple(value, indent + 1)
        else
            println(indent_str * "  $key: $value")
        end
    end
    println(indent_str * "}")
end

function ui_choices(values::Vector{Dict{Symbol,Any}})
    return PlutoUI.combine() do Child
        # Generate the UI elements based on the values parameter
        elements = [
            if value[:type] == :Slider
                # Include the default value for the slider
                md""" $(value[:label]): $(
                    Child(value[:label], Slider(value[:range], default = value[:default], show_value=true))
                )"""
            elseif value[:type] == :MultiCheckBox
                md""" $(value[:label]): $(
                    Child(value[:label], MultiCheckBox(value[:options], default=value[:default]))
                )"""
            else
                throw(ArgumentError("Unsupported UI type: $(value[:type])"))
            end

            for value in values
        ]

        md"""
        $(elements)
        """
    end
end

# TODO does not work
function set_margins(; max_width::Int = 2000, padding::String = "max(80px, 5%)")
    html"""
    <style>
        main {
            margin: 0 auto;
            max-width: $(max_width)px;
            padding-left: $(padding);
            padding-right: $(padding);
        }
    </style>
    """
end

end # end module
