import CMM.CMM_FF

function update_storage!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    natoms = length(coords)
    num_fragments = length(fragment_indices)
    # check if the storage matches the current coordinates.
    # If not, reallocate the storage.
    if num_fragments != ff.storage.num_fragments || (natoms) != length(ff.storage.multipoles) || length(ff.results.energies) == 1
        ff.storage = CMMStorage(
            num_fragments,
            SlaterShellDamping(),
            SlaterOverlapDamping(),
            [@MMatrix zeros(3, 3) for _ in 1:natoms], # α_inv
            [LocalAxes() for _ in 1:(natoms)], # local axes
            [CSMultipole2() for _ in 1:(natoms)], # perm multipoles
            [CSMultipole1() for _ in 1:natoms], # induced multipoles
            zeros(natoms), # ϕ_core
            zeros(natoms), # ϕ_shell
            zeros(natoms), # ϕ_induced
            zeros(natoms), # ϕ_exch_pol
            zeros(natoms), # ϕ_dispersion
            zeros(natoms), # ϕ_repulsion
            zeros(natoms), # ϕ_donor
            zeros(natoms), # ϕ_acceptor
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_core
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_shell
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_induced
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_exch_pol
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_dispersion
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_repulsion
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_donor
            [@MVector zeros(3) for _ in 1:(natoms)], # E_field_acceptor
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_core
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_shell
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_induced
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_exch_pol
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_dispersion
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_repulsion
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_donor
            [@MMatrix zeros(3, 3) for _ in 1:(natoms)], # E_field_gradients_acceptor
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_core
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_shell
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_induced
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_exch_pol
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_dispersion
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_repulsion
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_donor
            [@MArray zeros(3, 3, 3) for _ in 1:(natoms)], # E_field_gradient_gradients_acceptor
            zeros(natoms), # Δq_ct
            zeros(4 * natoms + num_fragments), # b_vec
            zeros(4 * natoms + num_fragments, 4 * natoms + num_fragments), # polarization matrix
            zeros(4 * natoms + num_fragments), # residual for conjugate gradient
            zeros(3), # applied field
            zeros(natoms), # η_fq
            ff.storage.include_exch_pol,
            ff.storage.include_charge_transfer,
            [@MArray zeros(3, 3, 3) for _ in 1:num_fragments], # variable charge gradients
            [@MVector zeros(3) for _ in 1:natoms], # torques
            [@MMatrix zeros(3, 3) for _ in 1:3], # dα
            [@MVector zeros(3) for _ in 1:natoms], # deformation gradients
            [@MVector zeros(3) for _ in 1:natoms], # pauli gradients
            [@MVector zeros(3) for _ in 1:natoms], # dispersion gradients
            [@MVector zeros(3) for _ in 1:natoms], # electrostatic gradients
            [@MVector zeros(3) for _ in 1:natoms], # polarization gradients
            [@MVector zeros(3) for _ in 1:natoms], # CT gradients
        )
        ff.results = ForceFieldResults(natoms)
    end
    return 0.0
end