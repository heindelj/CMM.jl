@inline function δ(i::Int, j::Int)
    return i == j ? 1 : 0
end

function ϵ_ijk(i::Int, j::Int, k::Int)
    perm_table = Dict(
        (1,2,3) =>  1.0,
        (2,3,1) =>  1.0,
        (3,1,2) =>  1.0,
        (3,2,1) => -1.0,
        (2,1,3) => -1.0,
        (1,3,2) => -1.0,
    )
    if i == j || i == k || j == k
        return 0.0
    end
    return perm_table[(i,j,k)]
end

"""
Almost all of the code in here is an implementation of the axis systems
and derivatives of the associated rotation matrices described in:
dx.doi.org/10.1021/ct401096t | J. Chem. Theory Comput. 2014, 10, 1638−1651
Without looking at that paper while reading this, it will be extremely
difficult to understand, in detail, what is going on. The function names
tell you exactly what is going on. I use identical notation to the extent
that is possible.
"""

mutable struct LocalAxes
    R::MMatrix{3, 3, Float64}
    # store the gradients with respect to each basis function
    # with respect to each index used in constructing the axis
    # system. We store them in the order: i, i_z, i_x, i_y
    dex::SVector{4, MMatrix{3, 3, Float64}}
    dey::SVector{4, MMatrix{3, 3, Float64}}
    dez::SVector{4, MMatrix{3, 3, Float64}}
    i::Int
    i_z::Int
    i_x::Int
    i_y::Int
    type::Symbol
end
LocalAxes() = LocalAxes(zeros(3, 3), [zeros(3, 3) for _ in 1:4], [zeros(3, 3) for _ in 1:4], [zeros(3, 3) for _ in 1:4], 0, 0, 0, 0, :none)
LocalAxes(R::MMatrix{3, 3, Float64}, i::Int, i_z::Int, i_x::Int, i_y::Int, type::Symbol) = LocalAxes(R, [zeros(3, 3) for _ in 1:4], [zeros(3, 3) for _ in 1:4], [zeros(3, 3) for _ in 1:4], i, i_z, i_x, i_y, type)

"""
Modifies the provided 3x3 matrix such that  R*src = dest. That is, this matrix aligns
the src axis to the dest axis. We assume the provided src and dest vectors are normalized.
See: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
"""
function get_rotation_matrix_to_align_two_unit_vectors!(R::Matrix{Float64}, src::Vector{Float64}, dest::Vector{Float64})
    @assert size(R) == (3, 3) "Provided matrix must be 3x3 since it is a rotation matrix (in 3D)."
    v = cross(src, dest)
    c = dot(src, dest)
    skew_symmetric_cross!(R, v)
    @views R[:, :] = diagm(ones(3)) + R + R * R * (1.0 / (1.0 + c))
    return
end

function get_rotation_matrix_to_align_two_unit_vectors(src::Vector{Float64}, dest::Vector{Float64})
    R = diagm(ones(3))
    v = cross(src, dest)
    c = dot(src, dest)
    skew_symmetric_cross!(R, v)
    @views R[:, :] = diagm(ones(3)) + R + R * R * (1.0 / (1.0 + c))
    return R
end

"""
Forms the skew-symmetric cross-product matrix from a vector v which is assumed
to be formed from the cross-product of two vectors.
"""
function skew_symmetric_cross!(A::Matrix{Float64}, v::Vector{Float64})
    A[1, 1] = 0.0
    A[2, 1] = v[3]
    A[3, 1] = -v[2]
    A[1, 2] = -v[3]
    A[2, 2] = 0.0
    A[3, 2] = v[1]
    A[1, 3] = v[2]
    A[2, 3] = -v[1]
    A[3, 3] = 0.0
end

"""
Converts torque on a dipole into a force on the atom containing the dipole. Does this
by multiplying the torques with the derivatives of components of the local to global
rotation matrix. These derivatives are all computed in this function rather than being
stored in a 3x3x3 tensor for instance.
"""
function torque_to_force!(coords::AbstractVector{MVector{3,Float64}}, local_axes::LocalAxes, grads::AbstractVector{MVector{3,Float64}}, torques::AbstractVector{MVector{3,Float64}})
    if local_axes.type == :Global
        return
    end
    
    # get vectors making up local axis frame
    u_vec = coords[local_axes.i_z] - coords[local_axes.i]
    v_vec = coords[local_axes.i_x] - coords[local_axes.i]
    u = norm(u_vec)
    v = norm(v_vec)
    normalize!(u_vec)
    normalize!(v_vec)
    w_vec = normalize!(cross(u_vec, v_vec))

    # get projection of torque vector onto each axis frame
    dtdu = torques[local_axes.i] ⋅ u_vec
    dtdv = torques[local_axes.i] ⋅ v_vec
    dtdw = torques[local_axes.i] ⋅ w_vec

    vxu = normalize!(cross(v_vec, u_vec))
    wxu = normalize!(cross(w_vec, u_vec))
    uvcos = u_vec ⋅ v_vec
    uvsin = sqrt(1 - uvcos^2)

    du = @MVector zeros(3)
    dv = @MVector zeros(3)

    # compute derivative of basis vector times torque projections
    if local_axes.type == :ZthenX
        du = vxu * dtdv / (u * uvsin) + wxu * dtdw / u
        dv = -vxu * dtdu / (v * uvsin)
    elseif local_axes.type == :Bisector
        wxv = normalize!(cross(w_vec, v_vec))
        du = vxu * dtdv / (u * uvsin) + 0.5 * wxu * dtdw / u
        dv = -vxu * dtdu / (v * uvsin) + 0.5 * wxv * dtdw / v
    else
        @assert false "Haven't implemented this type of local axis system yet."
    end

    grads[local_axes.i_z] -= du
    grads[local_axes.i_x] -= dv
    grads[local_axes.i]   += du + dv

end

"""
We store the rotation matrix from local to global coordinate systems in the
LocalAxes struct. The column vectors of R are therefore, just the unit vectors
along the local axes. From the z-axis, we can get both the angle of the
r with the z-axis and the angle of r to the x-y plane (i.e. ϕ and θ).
Note: r_ij is the vector from atom_i to atom_j.
axis_i is the local axis centered on atom i.
"""
@inline function get_spherical_angles(r_i::MVector{3, Float64}, r_j::MVector{3, Float64}, axis_i::LocalAxes, axis_j::LocalAxes)
    r_ij = r_j - r_i

    # note n is already normalized so we don't divide by its length
    @views e_x_i = axis_i.R[:, 1]
    @views e_y_i = axis_i.R[:, 2]
    @views e_z_i = axis_i.R[:, 3]
    @views e_x_j = axis_j.R[:, 1]
    @views e_y_j = axis_j.R[:, 2]
    @views e_z_j = axis_j.R[:, 3]
    dot_i = clamp( r_ij ⋅ e_z_i / norm(r_ij), -1.0, 1.0)
    dot_j = clamp(-r_ij ⋅ e_z_j / norm(r_ij), -1.0, 1.0)

    # use atan2 for avoiding small-angle instabilities and avoiding
    # singularities in derivatives
    ϕ_i = atan(sqrt(1-dot_i^2), dot_i)
    ϕ_j = atan(sqrt(1-dot_j^2), dot_j)

    r_proj_on_xy_plane_i =  r_ij - r_ij ⋅ e_z_i * e_z_i
    r_proj_on_xy_plane_j = -r_ij + r_ij ⋅ e_z_j * e_z_j
    dot_i = clamp(r_proj_on_xy_plane_i ⋅ e_x_i / norm(r_proj_on_xy_plane_i), -1.0, 1.0)
    dot_j = clamp(r_proj_on_xy_plane_j ⋅ e_x_j / norm(r_proj_on_xy_plane_j), -1.0, 1.0)
    θ_i = atan(sqrt(1-dot_i^2), dot_i)
    θ_j = atan(sqrt(1-dot_j^2), dot_j)
    
    # NOTE: Techincally, we should be mapping θ_i and θ_j to
    # to [0, 2π] by checking the sign of the dot product with
    # y-axis and when less than zero return 2π - θ.
    # But, we only use these angles to evaluate spherical harmonics
    # and we only ever use the cosine terms of the real spherical
    # harmonics which means it doesn't matter if we get the other
    # half of the angle since cos is 2π periodic.
    #
    # We may have to revisit the above assumption in the future.
    return θ_i, θ_j, ϕ_i, ϕ_j
end

"""
Note, we pass in a buffer used to store the gradients with respect to
θ_i, θ_j, ϕ_i, and ϕ_j. There will be gradients with respect to each
atom used to define the local axis system, and the other atom used in
computing r_ij. Since the gradient w.r.t. j of each angle is negative
the same gradient w.r.t. i, we only store the gradients relevant to
atoms in the local axis system. Therefore, there are a maximum of four
gradient contributions for each of the four angles. The number of nonzero
contributions depends on the particular local axis system.
"""
@inline function get_spherical_angles_and_gradients!(r_i::MVector{3, Float64}, r_j::MVector{3, Float64}, axis_i::LocalAxes, axis_j::LocalAxes, angle_grads::SVector{4, MMatrix{3, 4, Float64, 12}})
    # NOTE: When computing the gradients below, we default to zero
    # when the dot product is sufficiently close to 1. This is because at
    # that point the dot product gradient terms will all be zero, so the
    # overall gradient should be zero, but there is a divide by zero which
    # results in nans and infs. In practice, there is only a problem when
    # two vectors are prefectly colinear which is vanishingly rare in simulations
    # but quite common in hand-constructed configurations.
    
    r_ij = r_j - r_i

    # note basis vectors are already normalized so we don't divide by their length
    @views e_x_i = axis_i.R[:, 1]
    @views e_y_i = axis_i.R[:, 2]
    @views e_z_i = axis_i.R[:, 3]
    @views e_x_j = axis_j.R[:, 1]
    @views e_y_j = axis_j.R[:, 2]
    @views e_z_j = axis_j.R[:, 3]
    dot_i = clamp( r_ij ⋅ e_z_i / norm(r_ij), -1.0, 1.0)
    dot_j = clamp(-r_ij ⋅ e_z_j / norm(r_ij), -1.0, 1.0)
    ϕ_i = atan(sqrt(1-dot_i^2), dot_i)
    ϕ_j = atan(sqrt(1-dot_j^2), dot_j)

    # get gradients of ϕ w.r.t. each atom used in the local axis system
    # and the other atom at the end of the direction vector
    ### GRADIENT WRT INDEX i ###
    if abs(1.0 - dot_i^2) > 1e-13
        grad_dot_i_j = (norm(r_ij) * diagm(ones(3)) - r_ij * r_ij' / norm(r_ij)) * e_z_i / norm(r_ij)^2
        grad_dot_i_i = axis_i.dez[1] * r_ij / norm(r_ij) - grad_dot_i_j
        grad_dot_i_iz = axis_i.dez[2] * r_ij / norm(r_ij)
        grad_dot_i_ix = axis_i.dez[3] * r_ij / norm(r_ij)
        dϕ_i_i  = -dot_i^2 * grad_dot_i_i / sqrt(1-dot_i^2)  - sqrt(1-dot_i^2) * grad_dot_i_i
        dϕ_i_j  = -dot_i^2 * grad_dot_i_j / sqrt(1-dot_i^2)  - sqrt(1-dot_i^2) * grad_dot_i_j
        dϕ_i_iz = -dot_i^2 * grad_dot_i_iz / sqrt(1-dot_i^2) - sqrt(1-dot_i^2) * grad_dot_i_iz
        dϕ_i_ix = -dot_i^2 * grad_dot_i_ix / sqrt(1-dot_i^2) - sqrt(1-dot_i^2) * grad_dot_i_ix
    else
        dϕ_i_i  = 0.0
        dϕ_i_j  = 0.0
        dϕ_i_iz = 0.0
        dϕ_i_ix = 0.0
    end

    ### GRADIENT WRT INDEX j ###
    if abs(1.0 - dot_i^2) > 1e-13
        grad_dot_j_i = (norm(r_ij) * diagm(ones(3)) - r_ij * r_ij' / norm(r_ij)) * e_z_j / norm(r_ij)^2
        grad_dot_j_j =  -axis_j.dez[1] * r_ij / norm(r_ij) - grad_dot_j_i
        grad_dot_j_jz = -axis_j.dez[2] * r_ij / norm(r_ij)
        grad_dot_j_jx = -axis_j.dez[3] * r_ij / norm(r_ij)
        dϕ_j_i  = -dot_j^2 * grad_dot_j_i / sqrt(1-dot_j^2)  - sqrt(1-dot_j^2) * grad_dot_j_i
        dϕ_j_j  = -dot_j^2 * grad_dot_j_j / sqrt(1-dot_j^2)  - sqrt(1-dot_j^2) * grad_dot_j_j
        dϕ_j_jz = -dot_j^2 * grad_dot_j_jz / sqrt(1-dot_j^2) - sqrt(1-dot_j^2) * grad_dot_j_jz
        dϕ_j_jx = -dot_j^2 * grad_dot_j_jx / sqrt(1-dot_j^2) - sqrt(1-dot_j^2) * grad_dot_j_jx
    else
        dϕ_j_i  = 0.0
        dϕ_j_j  = 0.0
        dϕ_j_jz = 0.0
        dϕ_j_jx = 0.0
    end

    r_proj_on_xy_plane_i =  r_ij - r_ij ⋅ e_z_i * e_z_i
    r_proj_on_xy_plane_j = -r_ij + r_ij ⋅ e_z_j * e_z_j
    dot_i = clamp(r_proj_on_xy_plane_i ⋅ e_x_i / norm(r_proj_on_xy_plane_i), -1.0, 1.0)
    dot_j = clamp(r_proj_on_xy_plane_j ⋅ e_x_j / norm(r_proj_on_xy_plane_j), -1.0, 1.0)
    θ_i = atan(sqrt(1-dot_i^2), dot_i)
    θ_j = atan(sqrt(1-dot_j^2), dot_j)

    # get gradients of θ w.r.t. each atom used in the local axis system
    # and the other atom at the end of the direction vector
    ### GRADIENT WRT INDEX i ###
    if abs(1.0 - dot_i^2) > 1e-13
        grad_proj_vec_i = ((r_ij ⋅ e_z_i)^2 * e_z_i - (r_ij ⋅ e_z_i) * r_ij) / norm(r_proj_on_xy_plane_i)
        grad_norm_proj_vec_i  = axis_i.dez[1] * grad_proj_vec_i
        grad_norm_proj_vec_iz = axis_i.dez[2] * grad_proj_vec_i
        grad_norm_proj_vec_ix = axis_i.dez[3] * grad_proj_vec_i
    
        grad_dot_i_j = (norm(r_proj_on_xy_plane_i) * diagm(ones(3)) - r_proj_on_xy_plane_i * r_proj_on_xy_plane_i' / norm(r_proj_on_xy_plane_i)) * e_x_i / norm(r_proj_on_xy_plane_i)^2
        grad_dot_i_i  = (norm(r_proj_on_xy_plane_i) * axis_i.dex[1] * r_ij - r_proj_on_xy_plane_i ⋅ e_x_i * grad_norm_proj_vec_i)  / norm(r_proj_on_xy_plane_i)^2 - grad_dot_i_j
        grad_dot_i_iz = (norm(r_proj_on_xy_plane_i) * axis_i.dex[2] * r_ij - r_proj_on_xy_plane_i ⋅ e_x_i * grad_norm_proj_vec_iz) / norm(r_proj_on_xy_plane_i)^2
        grad_dot_i_ix = (norm(r_proj_on_xy_plane_i) * axis_i.dex[3] * r_ij - r_proj_on_xy_plane_i ⋅ e_x_i * grad_norm_proj_vec_ix) / norm(r_proj_on_xy_plane_i)^2

        dθ_i_i  = -dot_i^2 * grad_dot_i_i / sqrt(1-dot_i^2)  - sqrt(1-dot_i^2) * grad_dot_i_i
        dθ_i_j  = -dot_i^2 * grad_dot_i_j / sqrt(1-dot_i^2)  - sqrt(1-dot_i^2) * grad_dot_i_j
        dθ_i_iz = -dot_i^2 * grad_dot_i_iz / sqrt(1-dot_i^2) - sqrt(1-dot_i^2) * grad_dot_i_iz
        dθ_i_ix = -dot_i^2 * grad_dot_i_ix / sqrt(1-dot_i^2) - sqrt(1-dot_i^2) * grad_dot_i_ix
    else
        dθ_i_i  = 0.0
        dθ_i_j  = 0.0
        dθ_i_iz = 0.0
        dθ_i_ix = 0.0
    end
    ### GRADIENT WRT INDEX j ###
    if abs(1.0 - dot_j^2) > 1e-13
        grad_proj_vec_j = ((r_ij ⋅ e_z_j)^2 * e_z_j - (r_ij ⋅ e_z_j) * r_ij) / norm(r_proj_on_xy_plane_j)
        grad_norm_proj_vec_j  = axis_j.dez[1] * grad_proj_vec_j
        grad_norm_proj_vec_jz = axis_j.dez[2] * grad_proj_vec_j
        grad_norm_proj_vec_jx = axis_j.dez[3] * grad_proj_vec_j
    
        grad_dot_j_i = (norm(r_proj_on_xy_plane_j) * diagm(ones(3)) - r_proj_on_xy_plane_j * r_proj_on_xy_plane_j' / norm(r_proj_on_xy_plane_j)) * e_x_j / norm(r_proj_on_xy_plane_j)^2
        grad_dot_j_j  = (-norm(r_proj_on_xy_plane_j) * axis_j.dex[1] * r_ij - r_proj_on_xy_plane_j ⋅ e_x_j * grad_norm_proj_vec_j)  / norm(r_proj_on_xy_plane_j)^2 - grad_dot_j_i
        grad_dot_j_jz = (-norm(r_proj_on_xy_plane_j) * axis_j.dex[2] * r_ij - r_proj_on_xy_plane_j ⋅ e_x_j * grad_norm_proj_vec_jz) / norm(r_proj_on_xy_plane_j)^2
        grad_dot_j_jx = (-norm(r_proj_on_xy_plane_j) * axis_j.dex[3] * r_ij - r_proj_on_xy_plane_j ⋅ e_x_j * grad_norm_proj_vec_jx) / norm(r_proj_on_xy_plane_j)^2

        dθ_j_i  = -dot_j^2 * grad_dot_j_i / sqrt(1-dot_j^2)  - sqrt(1-dot_j^2) * grad_dot_j_i
        dθ_j_j  = -dot_j^2 * grad_dot_j_j / sqrt(1-dot_j^2)  - sqrt(1-dot_j^2) * grad_dot_j_j
        dθ_j_jz = -dot_j^2 * grad_dot_j_jz / sqrt(1-dot_j^2) - sqrt(1-dot_j^2) * grad_dot_j_jz
        dθ_j_jx = -dot_j^2 * grad_dot_j_jx / sqrt(1-dot_j^2) - sqrt(1-dot_j^2) * grad_dot_j_jx
    else
        dθ_j_i  = 0.0
        dθ_j_j  = 0.0
        dθ_j_jz = 0.0
        dθ_j_jx = 0.0
    end

    # NOTE: In principle we could also need to store gradients involving the y-basis vector
    # Currently, we don't actually use it (see below note), so we just don't store or calculate
    # those gradients. If we did calculate them, we would need to add one column to this matrix
    # and go from there.
    @views angle_grads[1][:, 1] = dθ_i_i
    @views angle_grads[1][:, 2] = dθ_i_iz
    @views angle_grads[1][:, 3] = dθ_i_ix
    @views angle_grads[1][:, 4] = dθ_i_j

    @views angle_grads[2][:, 1] = dθ_j_j
    @views angle_grads[2][:, 2] = dθ_j_jz
    @views angle_grads[2][:, 3] = dθ_j_jx
    @views angle_grads[2][:, 4] = dθ_j_i

    @views angle_grads[3][:, 1] = dϕ_i_i
    @views angle_grads[3][:, 2] = dϕ_i_iz
    @views angle_grads[3][:, 3] = dϕ_i_ix
    @views angle_grads[3][:, 4] = dϕ_i_j

    @views angle_grads[4][:, 1] = dϕ_j_j
    @views angle_grads[4][:, 2] = dϕ_j_jz
    @views angle_grads[4][:, 3] = dϕ_j_jx
    @views angle_grads[4][:, 4] = dϕ_j_i

    # NOTE: Technically, we should be mapping θ_i and θ_j to
    # to [0, 2π] by checking the sign of the dot product with
    # y-axis and when less than zero return 2π - θ.
    # But, we only use these angles to evaluate spherical harmonics
    # and we only ever use the cosine terms of the real spherical
    # harmonics which means it doesn't matter if we get the other
    # half of the angle since cos is 2π periodic.
    #
    # We may have to revisit the above assumption in the future.
    return θ_i, θ_j, ϕ_i, ϕ_j
end

function get_global_axis_system!(i_center::Int, local_axes::LocalAxes)

    local_axes.i = i_center
    local_axes.i_z = 0
    local_axes.i_x = 0
    local_axes.i_y = 0
    local_axes.type = :Global

    e_x = [1.0, 0.0, 0.0]
    e_y = [0.0, 1.0, 0.0]
    e_z = [0.0, 0.0, 1.0]

    @views local_axes.R[:, 1] = e_x
    @views local_axes.R[:, 2] = e_y
    @views local_axes.R[:, 3] = e_z
end

function get_z_then_x_rotation_matrix_and_local_axis_system!(coords::AbstractVector{MVector{3,Float64}}, i_center::Int, i_z::Int, i_x::Int, local_axes::LocalAxes)

    local_axes.i = i_center
    local_axes.i_z = i_z
    local_axes.i_x = i_x
    local_axes.i_y = 0

    ξ = coords[i_z] - coords[i_center]
    η = coords[i_x] - coords[i_center]
    u = ξ
    v = η
    e_z = normalize(ξ)
    e_x = normalize!(η - η ⋅ e_z * e_z)
    e_y = normalize!(cross(e_z, e_x))
    
    @views local_axes.R[:, 1] = e_x
    @views local_axes.R[:, 2] = e_y
    @views local_axes.R[:, 3] = e_z

    local_axes.type = :ZthenX

    #dez_i = @MMatrix zeros(3, 3)
    #dez_iz = @MMatrix zeros(3, 3)
    #dez_ix = @MMatrix zeros(3, 3)
    #dez_iy = @MMatrix zeros(3, 3)

    #dex_i = @MMatrix zeros(3, 3)
    #dex_iz = @MMatrix zeros(3, 3)
    #dex_ix = @MMatrix zeros(3, 3)
    #dex_iy = @MMatrix zeros(3, 3)

    #dey_i = @MMatrix zeros(3, 3)
    #dey_iz = @MMatrix zeros(3, 3)
    #dey_ix = @MMatrix zeros(3, 3)
    #dey_iy = @MMatrix zeros(3, 3)

    # get derivatives of each basis vector w.r.t. each atom
    # z-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dez_du = (δ(α, γ) / norm(u) - u[α] * u[γ] / norm(u)^3)
    #            dez_i[β, α]  -= dez_du * δ(γ, β) # sign of these if flipped from paper
    #            dez_iz[β, α] += dez_du * δ(γ, β) # sign of these if flipped from paper
    #        end
    #    end
    #end
    
    # x-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dex_dv = (
    #                (δ(α, γ) - e_z[α] * e_z[γ]) / sqrt(norm(v)^2 - (v ⋅ e_z)^2) -
    #                (v[α] - (v ⋅ e_z) * e_z[α]) * (v[γ] - (v ⋅ e_z) * e_z[γ]) / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2)
    #            )
    #            dex_dz = (
    #                (v[α] - (v ⋅ e_z) * e_z[α]) * (v ⋅ e_z) * v[γ] / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2) - 
    #                (v[γ] * e_z[α] + (v ⋅ e_z) * δ(α, γ)) / sqrt(norm(v)^2 - (v ⋅ e_z)^2)
    #            )
    #            dex_i[β, α]  += -dex_dv * δ(γ, β) + dex_dz * dez_i[γ, β]
    #            dex_iz[β, α] +=  dex_dz * dez_iz[γ, β]
    #            dex_ix[β, α] +=  dex_dv * δ(γ, β)
    #        end
    #    end
    #end

    # y-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dey_dx = 0.0
    #            dey_dz = 0.0
    #            for σ in 1:3
    #                for τ in 1:3
    #                    dey_dx += ϵ_ijk(α, σ, τ) * e_z[σ] * δ(γ, τ)
    #                    dey_dz += ϵ_ijk(α, σ, τ) * e_x[τ] * δ(γ, σ)
    #                end
    #            end
    #            dey_i[β, α]  += dey_dx * dex_i[β, γ]  + dey_dz * dez_i[β, γ]
    #            dey_iz[β, α] += dey_dx * dex_iz[β, γ] + dey_dz * dez_iz[β, γ]
    #            dey_ix[β, α] += dey_dx * dex_ix[β, γ] + dey_dz * dez_ix[β, γ]
    #        end
    #    end
    #end
end

function get_bisector_rotation_matrix_and_local_axis_system!(coords::AbstractVector{MVector{3,Float64}}, i_center::Int, i_z::Int, i_x::Int, local_axes::LocalAxes)
    
    local_axes.i = i_center
    local_axes.i_z = i_z
    local_axes.i_x = i_x
    local_axes.i_y = 0

    ξ = coords[i_z] - coords[i_center]
    η = coords[i_x] - coords[i_center]
    u = norm(η) * ξ + norm(ξ) * η
    v = η

    e_z = normalize(u)
    vez = sum(v .* e_z)
    e_x = normalize!(v - vez * e_z)
    e_y = normalize!(cross(e_z, e_x))

    @views local_axes.R[:, 1] = e_x
    @views local_axes.R[:, 2] = e_y
    @views local_axes.R[:, 3] = e_z

    local_axes.type = :Bisector

    #dez_i = @MMatrix zeros(3, 3)
    #dez_iz = @MMatrix zeros(3, 3)
    #dez_ix = @MMatrix zeros(3, 3)
    #dez_iy = @MMatrix zeros(3, 3)

    #dex_i = @MMatrix zeros(3, 3)
    #dex_iz = @MMatrix zeros(3, 3)
    #dex_ix = @MMatrix zeros(3, 3)
    #dex_iy = @MMatrix zeros(3, 3)

    #dey_i = @MMatrix zeros(3, 3)
    #dey_iz = @MMatrix zeros(3, 3)
    #dey_ix = @MMatrix zeros(3, 3)
    #dey_iy = @MMatrix zeros(3, 3)


    # get derivatives of each basis vector w.r.t. each atom
    # z-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dez_du = (δ(α, γ) / norm(u) - u[α] * u[γ] / norm(u)^3)
    #            dez_i[β, α]  -= dez_du * ((norm(ξ) + norm(η)) * δ(γ, β) + η[γ] * ξ[β] / norm(ξ) + ξ[γ] * η[β] / norm(η))
    #            dez_iz[β, α] += dez_du * (norm(η) * δ(γ, β) + η[γ] * ξ[β] / norm(ξ))
    #            dez_ix[β, α] += dez_du * (norm(ξ) * δ(γ, β) + ξ[γ] * η[β] / norm(η))
    #        end
    #    end
    #end
    
    # x-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dex_dv = (
    #                (δ(α, γ) - e_z[α] * e_z[γ]) / sqrt(norm(v)^2 - (v ⋅ e_z)^2) -
    #                (v[α] - (v ⋅ e_z) * e_z[α]) * (v[γ] - (v ⋅ e_z) * e_z[γ]) / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2)
    #            )
    #            dex_dz = (
    #                (v[α] - (v ⋅ e_z) * e_z[α]) * (v ⋅ e_z) * v[γ] / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2) - 
    #                (v[γ] * e_z[α] + (v ⋅ e_z) * δ(α, γ)) / sqrt(norm(v)^2 - (v ⋅ e_z)^2)
    #            )
    #            dex_i[β, α]  += -dex_dv * δ(γ, β) + dex_dz * dez_i[β, γ]
    #            dex_iz[β, α] +=  dex_dz * dez_iz[β, γ]
    #            dex_ix[β, α] +=  dex_dv * δ(γ, β) + dex_dz * dez_ix[β, γ]
    #        end
    #    end
    #end

    # y-basis vector #
    #for α in 1:3 # loop over component of basis vector
    #    for β in 1:3 # loop over directions of vector
    #        for γ in 1:3 # loop over all chain rule terms
    #            dey_dx = 0.0
    #            dey_dz = 0.0
    #            for σ in 1:3
    #                for τ in 1:3
    #                    dey_dx += ϵ_ijk(α, σ, τ) * e_z[σ] * δ(γ, τ)
    #                    dey_dz += ϵ_ijk(α, σ, τ) * e_x[τ] * δ(γ, σ)
    #                end
    #            end
    #            dey_i[β, α]  += dey_dx * dex_i[β, γ]  + dey_dz * dez_i[β, γ]
    #            dey_iz[β, α] += dey_dx * dex_iz[β, γ] + dey_dz * dez_iz[β, γ]
    #            dey_ix[β, α] += dey_dx * dex_ix[β, γ] + dey_dz * dez_ix[β, γ]
    #        end
    #    end
    #end
end

function get_z_then_x_rotation_matrix_and_local_axis_system(coords::AbstractVector{MVector{3,Float64}}, i_center::Int, i_z::Int, i_x::Int)
    R = @MMatrix zeros(3, 3)

    ξ = coords[i_z] - coords[i_center]
    η = coords[i_x] - coords[i_center]
    u = ξ
    v = η
    e_z = normalize(ξ)
    e_x = normalize!(η - η ⋅ e_z * e_z)
    e_y = normalize!(cross(e_z, e_x))
    
    @views R[:, 1] = e_x
    @views R[:, 2] = e_y
    @views R[:, 3] = e_z

    dez_i = @MMatrix zeros(3, 3)
    dez_iz = @MMatrix zeros(3, 3)
    dez_ix = @MMatrix zeros(3, 3)
    dez_iy = @MMatrix zeros(3, 3)

    dex_i = @MMatrix zeros(3, 3)
    dex_iz = @MMatrix zeros(3, 3)
    dex_ix = @MMatrix zeros(3, 3)
    dex_iy = @MMatrix zeros(3, 3)

    dey_i = @MMatrix zeros(3, 3)
    dey_iz = @MMatrix zeros(3, 3)
    dey_ix = @MMatrix zeros(3, 3)
    dey_iy = @MMatrix zeros(3, 3)

    # get derivatives of each basis vector w.r.t. each atom
    # z-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dez_du = (δ(α, γ) / norm(u) - u[α] * u[γ] / norm(u)^3)
                dez_i[β, α]  -= dez_du * δ(γ, β) # sign of these if flipped from paper
                dez_iz[β, α] += dez_du * δ(γ, β) # sign of these if flipped from paper
            end
        end
    end
    
    # x-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dex_dv = (
                    (δ(α, γ) - e_z[α] * e_z[γ]) / sqrt(norm(v)^2 - (v ⋅ e_z)^2) -
                    (v[α] - (v ⋅ e_z) * e_z[α]) * (v[γ] - (v ⋅ e_z) * e_z[γ]) / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2)
                )
                dex_dz = (
                    (v[α] - (v ⋅ e_z) * e_z[α]) * (v ⋅ e_z) * v[γ] / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2) - 
                    (v[γ] * e_z[α] + (v ⋅ e_z) * δ(α, γ)) / sqrt(norm(v)^2 - (v ⋅ e_z)^2)
                )
                dex_i[β, α]  += -dex_dv * δ(γ, β) + dex_dz * dez_i[γ, β]
                dex_iz[β, α] +=  dex_dz * dez_iz[γ, β]
                dex_ix[β, α] +=  dex_dv * δ(γ, β)
            end
        end
    end

    # y-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dey_dx = 0.0
                dey_dz = 0.0
                for σ in 1:3
                    for τ in 1:3
                        dey_dx += ϵ_ijk(α, σ, τ) * e_z[σ] * δ(γ, τ)
                        dey_dz += ϵ_ijk(α, σ, τ) * e_x[τ] * δ(γ, σ)
                    end
                end
                dey_i[β, α]  += dey_dx * dex_i[β, γ]  + dey_dz * dez_i[β, γ]
                dey_iz[β, α] += dey_dx * dex_iz[β, γ] + dey_dz * dez_iz[β, γ]
                dey_ix[β, α] += dey_dx * dex_ix[β, γ] + dey_dz * dez_ix[β, γ]
            end
        end
    end

    return LocalAxes(R, [dex_i, dex_iz, dex_ix, dex_iy], [dey_i, dey_iz, dey_ix, dey_iy], [dez_i, dez_iz, dez_ix, dez_iy], i_center, i_z, i_x, 0, :ZthenX)
end

function get_bisector_rotation_matrix_and_local_axis_system(coords::AbstractVector{MVector{3,Float64}}, i_center::Int, i_z::Int, i_x::Int)
    R = @MMatrix zeros(3, 3)

    ξ = coords[i_z] - coords[i_center]
    η = coords[i_x] - coords[i_center]
    u = norm(η) * ξ + norm(ξ) * η
    v = η

    e_z = normalize(u)
    vez = sum(v .* e_z)
    e_x = normalize!(v - vez * e_z)
    e_y = normalize!(cross(e_z, e_x))

    @views R[:, 1] = e_x
    @views R[:, 2] = e_y
    @views R[:, 3] = e_z

    dez_i = @MMatrix zeros(3, 3)
    dez_iz = @MMatrix zeros(3, 3)
    dez_ix = @MMatrix zeros(3, 3)
    dez_iy = @MMatrix zeros(3, 3)

    dex_i = @MMatrix zeros(3, 3)
    dex_iz = @MMatrix zeros(3, 3)
    dex_ix = @MMatrix zeros(3, 3)
    dex_iy = @MMatrix zeros(3, 3)

    dey_i = @MMatrix zeros(3, 3)
    dey_iz = @MMatrix zeros(3, 3)
    dey_ix = @MMatrix zeros(3, 3)
    dey_iy = @MMatrix zeros(3, 3)


    # get derivatives of each basis vector w.r.t. each atom
    # z-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dez_du = (δ(α, γ) / norm(u) - u[α] * u[γ] / norm(u)^3)
                dez_i[β, α]  -= dez_du * ((norm(ξ) + norm(η)) * δ(γ, β) + η[γ] * ξ[β] / norm(ξ) + ξ[γ] * η[β] / norm(η))
                dez_iz[β, α] += dez_du * (norm(η) * δ(γ, β) + η[γ] * ξ[β] / norm(ξ))
                dez_ix[β, α] += dez_du * (norm(ξ) * δ(γ, β) + ξ[γ] * η[β] / norm(η))
            end
        end
    end
    
    # x-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dex_dv = (
                    (δ(α, γ) - e_z[α] * e_z[γ]) / sqrt(norm(v)^2 - (v ⋅ e_z)^2) -
                    (v[α] - (v ⋅ e_z) * e_z[α]) * (v[γ] - (v ⋅ e_z) * e_z[γ]) / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2)
                )
                dex_dz = (
                    (v[α] - (v ⋅ e_z) * e_z[α]) * (v ⋅ e_z) * v[γ] / (norm(v)^2 - (v ⋅ e_z)^2)^(3/2) - 
                    (v[γ] * e_z[α] + (v ⋅ e_z) * δ(α, γ)) / sqrt(norm(v)^2 - (v ⋅ e_z)^2)
                )
                dex_i[β, α]  += -dex_dv * δ(γ, β) + dex_dz * dez_i[β, γ]
                dex_iz[β, α] +=  dex_dz * dez_iz[β, γ]
                dex_ix[β, α] +=  dex_dv * δ(γ, β) + dex_dz * dez_ix[β, γ]
            end
        end
    end

    # y-basis vector #
    for α in 1:3 # loop over component of basis vector
        for β in 1:3 # loop over directions of vector
            for γ in 1:3 # loop over all chain rule terms
                dey_dx = 0.0
                dey_dz = 0.0
                for σ in 1:3
                    for τ in 1:3
                        dey_dx += ϵ_ijk(α, σ, τ) * e_z[σ] * δ(γ, τ)
                        dey_dz += ϵ_ijk(α, σ, τ) * e_x[τ] * δ(γ, σ)
                    end
                end
                dey_i[β, α]  += dey_dx * dex_i[β, γ]  + dey_dz * dez_i[β, γ]
                dey_iz[β, α] += dey_dx * dex_iz[β, γ] + dey_dz * dez_iz[β, γ]
                dey_ix[β, α] += dey_dx * dex_ix[β, γ] + dey_dz * dez_ix[β, γ]
            end
        end
    end

    return LocalAxes(R, [dex_i, dex_iz, dex_ix, dex_iy], [dey_i, dey_iz, dey_ix, dey_iy], [dez_i, dez_iz, dez_ix, dez_iy], i_center, i_z, i_x, 0, :Bisector)
end

"""
Returns rotation matrix to rotate the x-axis (HOH bisector) onto the first hydrogen atom.
Then, the rotation matrix to rotate the x-axis onto the second hydrogen is simply
the transpose of this rotation matrix. This is needed for rotating atomic dipoles
into the local axis system correctly.
"""
function get_rotation_matrix_to_rotate_hydrogen_dipoles(coords::AbstractMatrix{Float64})
    r_oh1 = coords[:, 2] - coords[:, 1]
    r_oh2 = coords[:, 3] - coords[:, 1]

    bisector = normalize(norm(r_oh2) * r_oh1 + norm(r_oh1) * r_oh2) # bisector
    normalize!(r_oh1)
    normalize!(r_oh2)

    # need to include the translation so the z-component of the dipoles is properly removed
    R = get_rotation_matrix_to_align_two_unit_vectors(bisector, r_oh1)
    return R
end

"""
Returns a matrix to rotate a vector in the x-y plane by θ radians (which is defined as the
water plane) in this instance. Because of this, no coordinates are needed.
"""
function get_rotation_matrix_in_xy_by_θ(θ::Float64)
    return [[cos(θ), sin(θ), 0.0] [-sin(θ), cos(θ), 0.0] [0.0, 0.0, 1.0]]
end

function get_rotation_matrix_in_xz_by_θ(θ::Float64)
    return [[cos(θ), 0.0, -sin(θ)] [0.0, 1.0, 0.0] [sin(θ), 0.0, cos(θ)]]
end

function get_rotation_matrix_in_yz_by_θ(θ::Float64)
    return [[1.0, 0.0, 0.0] [0.0, cos(θ), sin(θ)] [0.0, -sin(θ), cos(θ)]]
end