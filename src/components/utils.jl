
function get_polarization_energy(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    # matrix vector version
    return x' * (0.5 * A * x - b)
end

function get_total_dipole_moment(coords::Vector{MVector{3, Float64}}, multipoles::Vector{Multipole1})
    @assert length(coords) == length(multipoles) "Coordinates and multipoles aren't the same length! Is there supposed to be virtual site data in the coordinates?"
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += multipoles[i].q * coords[i]
        μ += multipoles[i].μ
    end
    return μ
end

function get_total_dipole_moment(coords::Vector{MVector{3, Float64}}, multipoles::Vector{Multipole2})
    @assert length(coords) == length(multipoles) "Coordinates and multipoles aren't the same length! Is there supposed to be virtual site data in the coordinates?"
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += multipoles[i].q * coords[i]
        μ += multipoles[i].μ
    end
    return μ
end

function get_total_dipole_moment(coords::Vector{MVector{3, Float64}}, multipoles::Vector{CSMultipole1})
    @assert length(coords) == length(multipoles) "Coordinates and multipoles aren't the same length! Is there supposed to be virtual site data in the coordinates?"
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += (multipoles[i].q_shell + multipoles[i].Z) * coords[i]
        μ += multipoles[i].μ
    end
    return μ
end

function get_total_dipole_moment(coords::Vector{MVector{3, Float64}}, multipoles::Vector{CSMultipole2})
    @assert length(coords) == length(multipoles) "Coordinates and multipoles aren't the same length! Is there supposed to be virtual site data in the coordinates?"

    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += (multipoles[i].q_shell + multipoles[i].Z) * coords[i]
        μ += multipoles[i].μ
    end
    return μ
end

function get_total_dipole_moment(coords::Vector{MVector{3, Float64}}, multipoles::Vector{CSMultipole2}, induced_multipoles::Vector{CSMultipole1})
    @assert length(coords) == length(multipoles) "Coordinates and multipoles aren't the same length! Is there supposed to be virtual site data in the coordinates?"
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += (multipoles[i].Z + multipoles[i].q_shell + induced_multipoles[i].q_shell) * coords[i]
        μ += multipoles[i].μ + induced_multipoles[i].μ
    end
    return μ
end

function get_total_induced_dipole_moment(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, induced_multipoles::Vector{Multipole1})
    
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += induced_multipoles[i].q * coords[i]
        μ += induced_multipoles[i].μ
    end
    return μ
end

function get_total_induced_dipole_moment(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, induced_multipoles::Vector{CSMultipole1}) 
    μ = @MVector zeros(3)
    for i in eachindex(coords)
        μ += (induced_multipoles[i].q_shell + induced_multipoles[i].Z) * coords[i]
        μ += induced_multipoles[i].μ
    end
    return μ
end

function fluctuating_charge_molecular_polarizability(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, fragment_indices::Vector{Vector{Int}}, ff::AbstractForceField, field_strength::Float64=1e-5)
    α = zeros(3, 3)
    for w in 1:3
        μ_ind = zeros(3)
        # apply field in +w direction
        ff.storage.applied_field[w] += field_strength
        evaluate!(coords, labels, fragment_indices, ff)
        μ_plus_w = get_total_induced_dipole_moment(coords, labels, ff.storage.induced_multipoles)

        # apply field in -w direction
        ff.storage.applied_field[w] -= 2 * field_strength
        evaluate!(coords, labels, fragment_indices, ff)
        μ_minus_w = get_total_induced_dipole_moment(coords, labels, ff.storage.induced_multipoles)

        ff.storage.applied_field[w] = 0.0

        # compute finite difference change in dipole for each direction
        dμ = (μ_plus_w - μ_minus_w) / (2 * field_strength)
        α[w, 1] += dμ[1]
        α[w, 2] += dμ[2]
        α[w, 3] += dμ[3]
    end
    return α
end

function polarizability_derivatives(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, fragment_indices::Vector{Vector{Int}}, ff::AbstractForceField, step_size::Float64=1e-5)
    # Store the polarizability derivatives with respect to the upper-triangle
    # components of the polarizability. The polarizability is always symmetric.
    # Derivatives are with respect to each cartesian coordinate (3N of these).
    α_deriv = zeros(3 * length(coords), 6)
    
    #              xx, xy, yy, xz, yz, zz
    pol_indices = [1,  4,  5,  7,  8,  9]
    for i in eachindex(coords)
        for (j, pol_index) in enumerate(pol_indices)
            for w in 1:3
                coords[i][w] += step_size
                f_plus_h = fluctuating_charge_molecular_polarizability(coords, labels, fragment_indices, ff)

                coords[i][w] -= 2 * step_size
                f_minus_h = fluctuating_charge_molecular_polarizability(coords, labels, fragment_indices, ff)
                coords[i][w] += step_size
                α_deriv[3*(i-1)+w, j] = (f_plus_h[pol_index] - f_minus_h[pol_index]) / (2 * step_size)
            end
        end
    end
    return α_deriv
end

function dipole_derivatives(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, fragment_indices::Vector{Vector{Int}}, ff::AbstractForceField, step_size::Float64=1e-5)
    # Store the polarizability derivatives with respect to the upper-triangle
    # components of the polarizability. The polarizability is always symmetric.
    # Derivatives are with respect to each cartesian coordinate (3N of these).
    dipole_deriv = zeros(3, 3 * length(coords))
    
    for i in eachindex(coords)
        for w in 1:3
            coords[i][w] += step_size
            evaluate!(coords, labels, fragment_indices, ff)
            f_plus_h = get_total_dipole_moment(coords, ff.storage.multipoles)

            coords[i][w] -= 2 * step_size
            evaluate!(coords, labels, fragment_indices, ff)
            f_minus_h = get_total_dipole_moment(coords, ff.storage.multipoles)
            coords[i][w] += step_size
            dipole_deriv[:, 3*(i-1)+w] = (f_plus_h - f_minus_h) / (2 * step_size)
        end
    end
    return dipole_deriv
end

function dipole_by_finite_difference(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, fragment_indices::Vector{Vector{Int}}, ff::AbstractForceField, field_strength::Float64=1e-5)
    # Store the polarizability derivatives with respect to the upper-triangle
    # components of the polarizability. The polarizability is always symmetric.
    # Derivatives are with respect to each cartesian coordinate (3N of these).
    dipole = zeros(3)
    
    for w in 1:3
        # apply field in +w direction
        ff.storage.applied_field[w] += field_strength
        evaluate!(coords, labels, fragment_indices, ff)
        E_plus_h = ff.results.energies[:Total]

        # apply field in -w direction
        ff.storage.applied_field[w] -= 2 * field_strength
        evaluate!(coords, labels, fragment_indices, ff)
        E_minus_h = ff.results.energies[:Total]

        ff.storage.applied_field[w] = 0.0

        # compute finite difference change in dipole for each direction
        dipole[w] = (E_plus_h - E_minus_h) / (2 * field_strength)
    end
    return dipole
end

"""
Takes the sums over some property (probably energies or forces)
and applies the appropriate weights to each term, returning the final value.
"""
function get_mbe_data_from_subsystem_sums(data::AbstractArray, num_fragments::Int)
    mbe_data = [zero(data[begin]) for _ in 1:length(data)]
    mbe_data[begin] = data[begin]
    for i_mbe in 1:(length(data)-1)
        for i in 0:i_mbe
            mbe_data[i_mbe+1] += (-1)^i * binomial(num_fragments - (i_mbe + 1) + i, i) * data[i_mbe-i+1]
        end
    end
    return mbe_data
end