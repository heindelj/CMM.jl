
function update_model_multipoles_and_local_axes!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    multipoles::AbstractVector{CSMultipole2},
    local_axes::AbstractVector{LocalAxes},
    grads::Union{Nothing, AbstractVector{MVector{3,Float64}}}=nothing,
    charge_grads::Union{Nothing, AbstractVector{MArray{Tuple{3,3,3},Float64,3,27}}}=nothing
)
    charges = zeros(length(labels))
    #energy = get_one_body_energy_and_charges!(coords, labels, fragment_indices, charges, params, nothing, charge_grads)
    get_charges_with_charge_flux!(coords, labels, fragment_indices, charges, params, charge_grads)
    for i_frag in eachindex(fragment_indices)
        # H2O
        if (count(==("H"), labels[fragment_indices[i_frag]]) == 2 &&
            count(==("O"), labels[fragment_indices[i_frag]]) == 1)

            # NOTE: We assume OHH order for the water molecules!
            # Make sure that you properly sort the structure before
            # doing anything! X is the virtual site.
            i_O = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]

            # store charges, rotated dipoles, and local axis information for each water
            get_bisector_rotation_matrix_and_local_axis_system!(coords, i_O, i_H1, i_H2, local_axes[i_O])
            multipoles[i_O].Z = params[:O_Z]
            multipoles[i_O].q_shell = charges[i_O] - params[:O_Z]

            @views multipoles[i_O].μ[:] = local_axes[i_O].R * get_local_frame_dipoles(labels[i_O], params)
            @views multipoles[i_O].Q[:, :] = local_axes[i_O].R * get_local_frame_quadrupoles(labels[i_O], params) * local_axes[i_O].R'
            
            get_z_then_x_rotation_matrix_and_local_axis_system!(coords, i_H1, i_O, i_H2, local_axes[i_H1])
            multipoles[i_H1].Z = params[:H_Z]
            multipoles[i_H1].q_shell = charges[i_H1] - params[:H_Z]
            @views multipoles[i_H1].μ[:] = local_axes[i_H1].R * get_local_frame_dipoles(labels[i_H1], params)
            @views multipoles[i_H1].Q[:, :] = local_axes[i_H1].R * get_local_frame_quadrupoles(labels[i_H1], params) * local_axes[i_H1].R'
            
            get_z_then_x_rotation_matrix_and_local_axis_system!(coords, i_H2, i_O, i_H1, local_axes[i_H2])
            multipoles[i_H2].Z = params[:H_Z]
            multipoles[i_H2].q_shell = charges[i_H2] - params[:H_Z]
            @views multipoles[i_H2].μ[:] = local_axes[i_H2].R * get_local_frame_dipoles(labels[i_H2], params)
            @views multipoles[i_H2].Q[:, :] = local_axes[i_H2].R * get_local_frame_quadrupoles(labels[i_H2], params) * local_axes[i_H2].R'
        elseif length(fragment_indices[i_frag]) == 1 # For now this is treated as some kind of ion
            i = fragment_indices[i_frag][1]
            get_global_axis_system!(i, local_axes[i])
            multipoles[i].Z = params[Symbol(labels[i], :_Z)]
            multipoles[i].q_shell = params[Symbol(labels[i], :_q)] - params[Symbol(labels[i], :_Z)]
        end
    end
end

function get_local_frame_dipoles(label::String, params::Dict{Symbol,Float64})
    dipole_data = Dict(
        "O" => MVector{3,Float64}([0.0, 0.0, params[:μ_O]]),
        "H" => MVector{3,Float64}([params[:μx_H], 0.0, params[:μz_H]]),
    )
    if haskey(dipole_data, label)
        return dipole_data[label]
    end
    return @MVector zeros(3)
end

function get_local_frame_quadrupoles(label::String, params::Dict{Symbol,Float64})
    quadrupole_data = Dict(
        "O" => MVector{5, Float64}([params[:Q20_O], params[:Q21c_O], params[:Q21s_O], params[:Q22c_O], params[:Q22s_O]]),
        "H" => MVector{5, Float64}([params[:Q20_H], params[:Q21c_H], params[:Q21s_H], params[:Q22c_H], params[:Q22s_H]]),
    )
    if haskey(quadrupole_data, label) # otherwise just keep it at zero
        return convert_spherical_quadrupole_to_cartesian_quadrupole(quadrupole_data[label])
    end
    return @MMatrix zeros(3, 3)
end


function get_model_inverse_polarizabilities!(
    α_inv::Vector{MMatrix{3, 3, Float64, 9}},
    labels::AbstractVector{String},
    local_axes::AbstractVector{LocalAxes},
    params::Dict{Symbol, Float64}
)
    for i in eachindex(labels)
        α_local = get_dipole_polarizability(labels[i], params)
        α_inv[i] = inv(local_axes[i].R * α_local * local_axes[i].R')
    end
end

function get_model_inverse_polarizabilities_with_ion_ion_damping!(
    α_inv::Vector{MMatrix{3, 3, Float64, 9}},
    labels::AbstractVector{String},
    ϕ_in::Vector{Float64},
    local_axes::AbstractVector{LocalAxes},
    params::Dict{Symbol, Float64}
)
    for i in eachindex(labels)
        λ_damp = 0.0
        if is_ion(labels[i])
            α_free = params[Symbol(labels[i], :_α)]
            q_ion = params[Symbol(labels[i], :_q)]
            λ_max = params[Symbol(labels[i], :_α_max_damp_factor)]
            b_ϕ = params[Symbol(labels[i], :_α_damp_exponent)]
            
            # NOTE: There may be a better way of dealing with the sign of
            # the potential? It might just be that we should multiply all
            # the parameters by q_ion * ϕ. Will have to play around to make
            # sure this works how I think it will.
            # Currently, the way this works, we will only damp for attractive
            # interactions. That is, ϕ_stop and ϕ_max are positive. -q_ion*ϕ_in
            # is also positive if the energy is attractive. This assumes the
            # electric potential. It may very well be that we should use the
            # repulsion potential. In which case, the sign ambiguity goes away
            # since everything will be positive. The best approach remains to be
            # seen.
            #λ_damp = cosine_switch_on(-q_ion * ϕ_in[i], 0.05, ϕ_stop, ϕ_max)
            λ_damp = λ_max*(1 - exp(-b_ϕ * (ϕ_in[i]).^2))
            α_local = get_dipole_polarizability(labels[i], params) + q_ion * λ_damp * diagm([α_free, α_free, α_free])
        else
            α_local = get_dipole_polarizability(labels[i], params)
        end
        
        α_inv[i] = inv(local_axes[i].R * α_local * local_axes[i].R')
    end
end

function get_quadrupole_polarizabilities!(
    α_quad::AbstractVector{MMatrix{6, 6, Float64, 36}},
    labels::AbstractVector{String},
    local_axes::AbstractVector{LocalAxes},
    params::Dict{Symbol, Float64}
)
    # Quadrupole polarizabilities are stored in Mandel form as a 6x6 rank 2 tensor
    # Note that the vector containing the spherical quadurpole polarizability
    # parameters are stored in the order:
    # [
    #   α_20;20, α_20;21c,  α_20;21s,  α_20;22c,  α_20;22s,
    #            α_21c;21c, α_21c;21s, α_21c;22c, α_21c;22s,
    #                       α_21s;21s, α_21s;22c, α_21s;22s,
    #                                  α_22c;22c, α_22c;22s,
    #                                             α_22s;22s
    # ]

    # Index mapping for cartesian and spherical components
    xx = 1
    yy = 2
    zz = 3
    yz = 4
    xz = 5
    xy = 6
    a2020   = 1
    a2021c  = 2
    a2021s  = 3
    a2022c  = 4
    a2022s  = 5
    a21c21c = 6
    a21c21s = 7
    a21c22c = 8
    a21c22s = 9
    a21s21s = 10
    a21s22c = 11
    a21s22s = 12
    a22c22c = 13
    a22c22s = 14
    a22s22s = 15
 
    # Storage for the rotator which rotates the quadrupole polarizability
    # tensor from the local to global axis system
    R_α_quad = @MMatrix zeros(6, 6)
    for i in eachindex(labels)
        ### Fill cartesian representation of quad pol by transformation from
        ### the spherical components.
        α_sph = get_spherical_quadrupole_polarizability_components_as_vector(labels[i], params)
        
        # Upper 3x3 of quad pol matrix in Mandel form
        α_quad[i][xx,xx] = α_sph[a2020] - 2 * sqrt(3) * α_sph[a2022c] + 3 * α_sph[a22c22c]
        α_quad[i][yy,yy] = α_sph[a2020] + 2 * sqrt(3) * α_sph[a2022c] + 3 * α_sph[a22c22c]
        α_quad[i][zz,zz] = 4 * α_sph[a2020]
        α_quad[i][yy,xx] = α_quad[i][xx,yy] = α_sph[a2020] - 3 * α_sph[a22c22c]
        α_quad[i][zz,xx] = α_quad[i][xx,zz] =  2 * sqrt(3) * α_sph[a2022c] - 2 * α_sph[a2020]
        α_quad[i][zz,yy] = α_quad[i][yy,zz] = -2 * sqrt(3) * α_sph[a2022c] - 2 * α_sph[a2020]
        
        # Off-diagonal 3x3 blocks of quad pol matrix in Mandel form
        α_quad[i][xx,yz] = α_quad[i][yz,xx] = sqrt(2) * (-sqrt(3) * α_sph[a2021s] + 3 * α_sph[a21s22c])
        α_quad[i][xx,xz] = α_quad[i][xz,xx] = sqrt(2) * (-sqrt(3) * α_sph[a2021c] + 3 * α_sph[a21c22c])
        α_quad[i][xx,xy] = α_quad[i][xy,xx] = sqrt(2) * (-sqrt(3) * α_sph[a2022s] + 3 * α_sph[a22c22s])
        α_quad[i][yy,yz] = α_quad[i][yz,yy] = sqrt(2) * (-sqrt(3) * α_sph[a2021s] - 3 * α_sph[a21s22c])
        α_quad[i][yy,xz] = α_quad[i][xz,yy] = sqrt(2) * (-sqrt(3) * α_sph[a2021c] - 3 * α_sph[a21c22c])
        α_quad[i][yy,xy] = α_quad[i][xy,yy] = sqrt(2) * (-sqrt(3) * α_sph[a2022s] - 3 * α_sph[a22c22s])
        α_quad[i][zz,yz] = α_quad[i][yz,zz] = sqrt(2) * (2 * sqrt(3) * α_sph[a2021s])
        α_quad[i][zz,xz] = α_quad[i][xz,zz] = sqrt(2) * (2 * sqrt(3) * α_sph[a2021c])
        α_quad[i][zz,xy] = α_quad[i][xy,zz] = sqrt(2) * (2 * sqrt(3) * α_sph[a2022s])

        # Lower 3x3 diagonal block
        α_quad[i][yz,yz] = 2 * (3 * α_sph[a21s21s])
        α_quad[i][xz,xz] = 2 * (3 * α_sph[a21c21c])
        α_quad[i][xy,xy] = 2 * (3 * α_sph[a22s22s])
        α_quad[i][yz,xz] = α_quad[i][xz,yz] = 2 * (3 * α_sph[a21c21s])
        α_quad[i][xy,yz] = α_quad[i][yz,xy] = 2 * (3 * α_sph[a21s22s])
        α_quad[i][xy,xz] = α_quad[i][xz,xy] = 2 * (3 * α_sph[a21c22s])

        # In the future, these rotation matrices for the quadrupole should be built
        # when the axis system is constructed. The rotation matrices should really
        # be stored separately from the struct which should just basically tell you
        # which indices are involved and which type of axis system it is.

        # Fill the rotation matrix based on R for local axis system
        R = local_axes[i].R
        R_α_quad[1,1] = R[1,1]^2
        R_α_quad[2,1] = R[2,1]^2
        R_α_quad[3,1] = R[3,1]^2
        R_α_quad[4,1] = sqrt(2) * R[2,1] * R[3,1]
        R_α_quad[5,1] = sqrt(2) * R[1,1] * R[3,1]
        R_α_quad[6,1] = sqrt(2) * R[1,1] * R[2,1]

        R_α_quad[1,2] = R[1,2]^2
        R_α_quad[2,2] = R[2,2]^2
        R_α_quad[3,2] = R[3,2]^2
        R_α_quad[4,2] = sqrt(2) * R[2,2] * R[3,2]
        R_α_quad[5,2] = sqrt(2) * R[1,2] * R[3,2]
        R_α_quad[6,2] = sqrt(2) * R[1,2] * R[2,2]

        R_α_quad[1,3] = R[1,3]^2
        R_α_quad[2,3] = R[2,3]^2
        R_α_quad[3,3] = R[3,3]^2
        R_α_quad[4,3] = sqrt(2) * R[2,3] * R[3,3]
        R_α_quad[5,3] = sqrt(2) * R[1,3] * R[3,3]
        R_α_quad[6,3] = sqrt(2) * R[1,3] * R[2,3]

        R_α_quad[1,4] = sqrt(2) * R[1,2] * R[1,3]
        R_α_quad[2,4] = sqrt(2) * R[2,2] * R[2,3]
        R_α_quad[3,4] = sqrt(2) * R[3,2] * R[3,3]
        R_α_quad[4,4] = (R[2,3] * R[3,2] + R[2,2] * R[3,3])
        R_α_quad[5,4] = (R[1,3] * R[3,2] + R[1,2] * R[3,3])
        R_α_quad[6,4] = (R[1,3] * R[2,2] + R[1,2] * R[2,3])

        R_α_quad[1,5] = sqrt(2) * R[1,1] * R[1,3]
        R_α_quad[2,5] = sqrt(2) * R[2,1] * R[2,3]
        R_α_quad[3,5] = sqrt(2) * R[3,1] * R[3,3]
        R_α_quad[4,5] = (R[2,3] * R[3,1] + R[2,1] * R[3,3])
        R_α_quad[5,5] = (R[1,3] * R[3,1] + R[1,1] * R[3,3])
        R_α_quad[6,5] = (R[1,3] * R[2,1] + R[1,1] * R[2,3])

        R_α_quad[1,6] = sqrt(2) * R[1,1] * R[1,2]
        R_α_quad[2,6] = sqrt(2) * R[2,1] * R[2,2]
        R_α_quad[3,6] = sqrt(2) * R[3,1] * R[3,2]
        R_α_quad[4,6] = (R[2,2] * R[3,1] + R[2,1] * R[3,2])
        R_α_quad[5,6] = (R[1,2] * R[3,1] + R[1,1] * R[3,2])
        R_α_quad[6,6] = (R[1,2] * R[2,1] + R[1,1] * R[2,2])

        α_quad[i] = R_α_quad' * (α_quad[i] / 12.0) * R_α_quad

        @views R_α_quad[:, :] -= R_α_quad[:, :] # reset rotation matrix
    end
end

function get_dipole_polarizability(label::String, params::Dict{Symbol,Float64})
    dipole_polarizability = @MMatrix zeros(3, 3)
    symbol_cartesian = [:x, :y, :z] # maps 1, 2, and 3 to appropriate cartesian axis
    # only loop over unique pairs since the polarizability
    # has to be symmetric. This maps αyx to αxy automatically.
    for i in 1:3
        for j in i:3
            sym_ij = Symbol(label, :_α, symbol_cartesian[i], symbol_cartesian[j])
            if haskey(params, sym_ij)
                if i == j
                    dipole_polarizability[i, i] = abs(params[sym_ij])
                else
                    dipole_polarizability[i, j] = params[sym_ij]
                    dipole_polarizability[j, i] = params[sym_ij]
                end
            elseif haskey(params, Symbol(label, :_α))
                if i == j
                    dipole_polarizability[i, i] = abs(params[Symbol(label, :_α)])
                end
            end
        end
    end
    return dipole_polarizability
end

function get_spherical_quadrupole_polarizability_components_as_vector(label::String, params::Dict{Symbol, Float64})
    component_symbols = [:_20, :_21c, :_21s, :_22c, :_22s]

    quad_polarizability_components = zeros(15)
    i_pol = 1
    for i in eachindex(component_symbols)
        for j in i:length(component_symbols)
            quad_pol_component, _ = maybe_get_param_or_return_default(
                Symbol(label, :_α, component_symbols[i], component_symbols[j]), params
            )
            quad_polarizability_components[i_pol] = quad_pol_component
            i_pol += 1
        end
    end
    return quad_polarizability_components
end

"""
See: https://core.ac.uk/download/pdf/38938135.pdf eqs.(32)-(34)
"""
function get_dipole_polarizability_derivatives!(α_inv::MMatrix{3, 3, Float64}, dα_x::MMatrix{3, 3, Float64}, dα_y::MMatrix{3, 3, Float64}, dα_z::MMatrix{3, 3, Float64})
    x = 1
    y = 2
    z = 3
    α = inv(α_inv)

    @views dα_x[:, 1] = [0.0, α[z, x], -α[y, x]]
    @views dα_x[:, 2] = [α[x, z], α[y, z] + α[z, y], α[z,z] - α[y, y]]
    @views dα_x[:, 3] = [-α[y, x], α[z, z] - α[y, y], -α[y, z] - α[z, y]]

    @views dα_y[:, 1] = [-α[x, z] - α[z, x], -α[z, y], α[x, x] - α[z, z]]
    @views dα_y[:, 2] = [-α[y, z], 0.0, α[y, x]]
    @views dα_y[:, 3] = [α[x, x] - α[z, z], α[x, y], α[x, z] + α[z, x]]
    
    @views dα_z[:, 1] = [α[x, y] + α[y, x], α[y, y] - α[x, x], α[y, z]]
    @views dα_z[:, 2] = [α[y, y] - α[x, x], -α[x, y] - α[y, x], -α[x, z]]
    @views dα_z[:, 3] = [α[z, y], -α[z, x], 0.0]
end