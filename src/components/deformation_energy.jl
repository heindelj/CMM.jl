
function get_coupled_morse_and_bend_energy_and_grads!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::Vector{Vector{Int}},
    params::Dict{Symbol,Float64},
    grads::Union{AbstractVector{MVector{3,Float64}},Nothing}=nothing
)
    energy = 0.0
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3 && labels[fragment_indices[i_frag]] == ["O", "H", "H"]
            ### PARAMETERS ###
            # bond parameters #
            k_b = params[:kb_water]
            D = params[:D_water]
            β = sqrt(k_b / (2 * D))
            r_e = params[:re_water]
            
            # angle parameters #
            k_a = params[:ka_water]
            cosθ_e = params[:cos_angle_eq_water]

            # bond-bond coupling #
            k_bb = params[:kbb_water]

            # bond-angle coupling #
            k_ba = params[:kba_water]

            ### EVALUATE ENERGIES ###
            i_O = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]

            r_OH1_mag = norm(coords[i_H1] - coords[i_O])
            r_OH2_mag = norm(coords[i_H2] - coords[i_O])

            cosθ = r_OH1 ⋅ r_OH2 / (r_OH1_mag * r_OH2_mag)

            # morse energy #
            energy += D * (1 - exp(-β * (r_OH1_mag - r_e)))^2
            energy += D * (1 - exp(-β * (r_OH2_mag - r_e)))^2

            # angle energy #
            energy += 0.5 * k_a * (cosθ - cosθ_e)^2
            
            # bond-bond coupling energy #
            energy += k_bb * (r_OH1_mag - r_e) * (r_OH2_mag - r_e)

            # bond-angle coupling energy #
            energy += k_ba * (r_OH1_mag - r_e) * (cosθ - cosθ_e)
            energy += k_ba * (r_OH2_mag - r_e) * (cosθ - cosθ_e)
            ### EVALUATE GRADIENTS ###
            if grads !== nothing
                # bond gradients #
                r_OH1_grad = 2 * D * (1 - exp(-β * (r_OH1_mag - r_e))) * (
                    β * exp(-β * (r_OH1_mag - r_e))
                ) * r_OH1 / r_OH1_mag
                r_OH2_grad = 2 * D * (1 - exp(-β * (r_OH2_mag - r_e))) * (
                    β * exp(-β * (r_OH2_mag - r_e))
                ) * r_OH2 / r_OH2_mag

                # angle gradients #
                r_OH1_cos_grad = (
                    r_OH2 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH1 / r_OH1_mag^2
                )
                r_OH2_cos_grad = (
                    r_OH1 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH2 / r_OH2_mag^2
                )
                r_OH1_grad += k_a * (cosθ - cosθ_e) * r_OH1_cos_grad
                r_OH2_grad += k_a * (cosθ - cosθ_e) * r_OH2_cos_grad

                # bond-bond coupling gradients #
                r_OH1_grad += k_bb * (r_OH2_mag - r_e) * r_OH1 / r_OH1_mag
                r_OH2_grad += k_bb * (r_OH1_mag - r_e) * r_OH2 / r_OH2_mag

                # bond-angle coupling gradients #
                r_OH1_grad += k_ba * (
                    (r_OH1_mag - r_e) * r_OH1_cos_grad +
                    (r_OH2_mag - r_e) * r_OH1_cos_grad +
                    (cosθ - cosθ_e) * r_OH1 / r_OH1_mag
                )
                r_OH2_grad += k_ba * (
                    (r_OH2_mag - r_e) * r_OH2_cos_grad +
                    (r_OH1_mag - r_e) * r_OH2_cos_grad +
                    (cosθ - cosθ_e) * r_OH2 / r_OH2_mag
                )
                grads[i_H1] += r_OH1_grad
                grads[i_H2] += r_OH2_grad
                grads[i_O]  -= r_OH1_grad + r_OH2_grad
            end
        else
            # currently just assuming this is an ion! #
            energy += 0.0
        end
    end
    return energy
end


function get_field_dependent_morse_and_bend_energy_and_grads!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    E_field::Vector{MVector{3, Float64}},
    μ_1::Float64, μ_2::Float64,
    E_field_slope_1::Float64, E_field_slope_2::Float64,
    Δq_ct::Vector{Float64}, ct_slope_1::Float64, ct_slope_2::Float64,
    include_charge_transfer::Bool,
    grads::Union{AbstractVector{MVector{3,Float64}},Nothing}=nothing,
    E_field_gradients::Union{Nothing, AbstractVector{MMatrix{3,3,Float64,9}}}=nothing
)
    energy = 0.0
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3 && labels[fragment_indices[i_frag]] == ["O", "H", "H"]
            ### PARAMETERS ###
            # bond parameters #
            k_b = params[:kb_water]
            D = params[:D_water]
            r_e = params[:re_water]
            β = sqrt(k_b / (2 * D))
            
            # angle parameters #
            k_a = params[:ka_water]
            cosθ_e = params[:cos_angle_eq_water]

            # bond-bond coupling #
            k_bb = params[:kbb_water]

            # bond-angle coupling #
            k_ba = params[:kba_water]

            ### EVALUATE ENERGIES ###
            i_O = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]

            r_OH1_mag = norm(coords[i_H1] - coords[i_O])
            r_OH2_mag = norm(coords[i_H2] - coords[i_O])

            cosθ = r_OH1 ⋅ r_OH2 / (r_OH1_mag * r_OH2_mag)

            # morse energy #
            E_OH1 = E_field[i_H1] ⋅ normalize(r_OH1)
            E_OH2 = E_field[i_H2] ⋅ normalize(r_OH2)

            μ_1_OH1 = copy(μ_1)
            μ_1_OH2 = copy(μ_1)
            μ_2_OH1 = copy(μ_2)
            μ_2_OH2 = copy(μ_2)

            Δr_e_OH1 = E_OH1 * μ_1_OH1 / (k_b - E_OH1 * μ_2_OH1) + ct_slope_1 * Δq_ct[i_H1]^2
            Δr_e_OH2 = E_OH2 * μ_1_OH2 / (k_b - E_OH2 * μ_2_OH2) + ct_slope_1 * Δq_ct[i_H2]^2
            
            shift_OH1 = (3 * k_b * β * Δr_e_OH1 + E_OH1 * μ_2_OH1)
            shift_OH2 = (3 * k_b * β * Δr_e_OH2 + E_OH2 * μ_2_OH2)

            r_e_OH1 = r_e + Δr_e_OH1
            r_e_OH2 = r_e + Δr_e_OH2

            k_b_OH1 = k_b - shift_OH1 + ct_slope_2 * Δq_ct[i_H1]^2
            k_b_OH2 = k_b - shift_OH2 + ct_slope_2 * Δq_ct[i_H2]^2

            # Ideally these branches never actually get hit
            # and as far as I can tell, they don't even for Fluoride
            # but still best to avoid this problem...
            if k_b_OH1 < 0.4 * k_b
                k_b_OH1 = 0.4 * k_b
            end
            if k_b_OH2 < 0.4 * k_b
                k_b_OH2 = 0.4 * k_b
            end

            β1 = sqrt(k_b_OH1 / (2 * D))
            β2 = sqrt(k_b_OH2 / (2 * D))
            energy += D * ((1 - exp(-β1 * (r_OH1_mag - r_e_OH1)))^2)
            energy += D * ((1 - exp(-β2 * (r_OH2_mag - r_e_OH2)))^2)

            # angle energy #
            energy += 0.5 * k_a * (cosθ - cosθ_e)^2
            
            # bond-bond coupling energy #
            energy += k_bb * (r_OH1_mag - r_e) * (r_OH2_mag - r_e)

            # bond-angle coupling energy #
            energy += k_ba * (r_OH1_mag - r_e) * (cosθ - cosθ_e)
            energy += k_ba * (r_OH2_mag - r_e) * (cosθ - cosθ_e)

            ### EVALUATE GRADIENTS ###
            if grads !== nothing
                # bond gradients #
                r_OH1_grad = 2 * D * (1 - exp(-β * (r_OH1_mag - r_e))) * (
                    β * exp(-β * (r_OH1_mag - r_e))
                ) * r_OH1 / r_OH1_mag
                r_OH2_grad = 2 * D * (1 - exp(-β * (r_OH2_mag - r_e))) * (
                    β * exp(-β * (r_OH2_mag - r_e))
                ) * r_OH2 / r_OH2_mag

                # angle gradients #
                r_OH1_cos_grad = (
                    r_OH2 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH1 / r_OH1_mag^2
                )
                r_OH2_cos_grad = (
                    r_OH1 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH2 / r_OH2_mag^2
                )
                r_OH1_grad += k_a * (cosθ - cosθ_e) * r_OH1_cos_grad
                r_OH2_grad += k_a * (cosθ - cosθ_e) * r_OH2_cos_grad

                # bond-bond coupling gradients #
                r_OH1_grad += k_bb * (r_OH2_mag - r_e) * r_OH1 / r_OH1_mag
                r_OH2_grad += k_bb * (r_OH1_mag - r_e) * r_OH2 / r_OH2_mag

                # bond-angle coupling gradients #
                r_OH1_grad += k_ba * (
                    (r_OH1_mag - r_e) * r_OH1_cos_grad +
                    (r_OH2_mag - r_e) * r_OH1_cos_grad +
                    (cosθ - cosθ_e) * r_OH1 / r_OH1_mag
                )
                r_OH2_grad += k_ba * (
                    (r_OH2_mag - r_e) * r_OH2_cos_grad +
                    (r_OH1_mag - r_e) * r_OH2_cos_grad +
                    (cosθ - cosθ_e) * r_OH2 / r_OH2_mag
                )
                grads[i_H1] += r_OH1_grad
                grads[i_H2] += r_OH2_grad
                grads[i_O]  -= r_OH1_grad + r_OH2_grad
            end
        else
            # currently just assuming this is an ion! #
            energy += 0.0
        end
    end
    return energy
end

function get_repulsion_field_dependent_morse!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    E_field_repulsion::Vector{MVector{3, Float64}},
    μ_1::Float64, μ_2::Float64,
    grads::Union{AbstractVector{MVector{3,Float64}},Nothing}=nothing,
    E_field_gradients::Union{Nothing, AbstractVector{MMatrix{3,3,Float64,9}}}=nothing
)
    energy = 0.0
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3 && labels[fragment_indices[i_frag]] == ["O", "H", "H"]
            ### PARAMETERS ###
            # bond parameters #
            k_b = params[:kb_water]
            D = params[:D_water]
            r_e = params[:re_water]
            β = sqrt(k_b / (2 * D))
            
            # angle parameters #
            k_a = params[:ka_water]
            cosθ_e = params[:cos_angle_eq_water]

            # bond-bond coupling #
            k_bb = params[:kbb_water]

            # bond-angle coupling #
            k_ba = params[:kba_water]

            ### EVALUATE ENERGIES ###
            i_O = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]

            r_OH1_mag = norm(coords[i_H1] - coords[i_O])
            r_OH2_mag = norm(coords[i_H2] - coords[i_O])

            cosθ = r_OH1 ⋅ r_OH2 / (r_OH1_mag * r_OH2_mag)

            # morse energy #
            E_OH1 = E_field_repulsion[i_H1] ⋅ normalize(r_OH1)
            E_OH2 = E_field_repulsion[i_H2] ⋅ normalize(r_OH2)

            Δr_e_OH1 = E_OH1 * μ_1 / (k_b - E_OH1 * μ_2)
            Δr_e_OH2 = E_OH2 * μ_1 / (k_b - E_OH2 * μ_2)
            
            # Ideally these branches never actually get hit
            # and as far as I can tell, they don't even for Fluoride
            # but still best to avoid this problem...
            shift_OH1 = (3 * k_b * β * Δr_e_OH1 + E_OH1 * μ_2)
            shift_OH2 = (3 * k_b * β * Δr_e_OH2 + E_OH2 * μ_2)
            if shift_OH1 > 0.5 * k_b
                shift_OH1 = 0.5 * k_b
            end
            if shift_OH2 > 0.5 * k_b
                shift_OH2 = 0.5 * k_b
            end

            r_e_OH1 = r_e + Δr_e_OH1
            r_e_OH2 = r_e + Δr_e_OH2

            k_b_OH1 = k_b - shift_OH1
            k_b_OH2 = k_b - shift_OH2

            β1 = sqrt(k_b_OH1 / (2 * D))
            β2 = sqrt(k_b_OH2 / (2 * D))
            energy += D * ((1 - exp(-β1 * (r_OH1_mag - r_e_OH1)))^2)
            energy += D * ((1 - exp(-β2 * (r_OH2_mag - r_e_OH2)))^2)

            # angle energy #
            energy += 0.5 * k_a * (cosθ - cosθ_e)^2
            
            # bond-bond coupling energy #
            energy += k_bb * (r_OH1_mag - r_e) * (r_OH2_mag - r_e)

            # bond-angle coupling energy #
            energy += k_ba * (r_OH1_mag - r_e) * (cosθ - cosθ_e)
            energy += k_ba * (r_OH2_mag - r_e) * (cosθ - cosθ_e)

            ### EVALUATE GRADIENTS ###
            if grads !== nothing
                # bond gradients #
                r_OH1_grad = 2 * D * (1 - exp(-β * (r_OH1_mag - r_e))) * (
                    β * exp(-β * (r_OH1_mag - r_e))
                ) * r_OH1 / r_OH1_mag
                r_OH2_grad = 2 * D * (1 - exp(-β * (r_OH2_mag - r_e))) * (
                    β * exp(-β * (r_OH2_mag - r_e))
                ) * r_OH2 / r_OH2_mag

                # angle gradients #
                r_OH1_cos_grad = (
                    r_OH2 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH1 / r_OH1_mag^2
                )
                r_OH2_cos_grad = (
                    r_OH1 / (r_OH1_mag * r_OH2_mag) -
                    cosθ * r_OH2 / r_OH2_mag^2
                )
                r_OH1_grad += k_a * (cosθ - cosθ_e) * r_OH1_cos_grad
                r_OH2_grad += k_a * (cosθ - cosθ_e) * r_OH2_cos_grad

                # bond-bond coupling gradients #
                r_OH1_grad += k_bb * (r_OH2_mag - r_e) * r_OH1 / r_OH1_mag
                r_OH2_grad += k_bb * (r_OH1_mag - r_e) * r_OH2 / r_OH2_mag

                # bond-angle coupling gradients #
                r_OH1_grad += k_ba * (
                    (r_OH1_mag - r_e) * r_OH1_cos_grad +
                    (r_OH2_mag - r_e) * r_OH1_cos_grad +
                    (cosθ - cosθ_e) * r_OH1 / r_OH1_mag
                )
                r_OH2_grad += k_ba * (
                    (r_OH2_mag - r_e) * r_OH2_cos_grad +
                    (r_OH1_mag - r_e) * r_OH2_cos_grad +
                    (cosθ - cosθ_e) * r_OH2 / r_OH2_mag
                )
                grads[i_H1] += r_OH1_grad
                grads[i_H2] += r_OH2_grad
                grads[i_O]  -= r_OH1_grad + r_OH2_grad
            end
        else
            # currently just assuming this is an ion! #
            energy += 0.0
        end
    end
    return energy
end