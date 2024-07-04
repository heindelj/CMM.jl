
function build_cmm_model(
    include_gradients::Bool=false,
    include_exch_pol::Bool=true,
    include_charge_transfer::Bool=true,
    custom_terms::Union{Vector{Function}, Nothing}=nothing,
    custom_params::Union{Dict{Symbol, Float64}, Nothing}=nothing
)
    natoms = 0
    num_fragments = 0
    if include_exch_pol
        params = get_parameter_dict_with_exch_pol()
    else
        params = get_parameter_dict_induction()
    end

    if custom_params !== nothing
        params = custom_params
    end
    if include_gradients
        ff_terms = [
            reset_storage!, get_one_body_properties!, permanent_electrostatics!,
            short_range_energy_and_gradients!, polarization_energy_and_electrostatic_gradients!,
            charge_transfer_constraint_gradients!
        ]
    else
        ff_terms = [
            reset_storage!, get_one_body_properties!,
            permanent_electrostatics!, short_range_energy!,
            polarization_energy!
        ]
    end
    if custom_terms !== nothing
        ff_terms = custom_terms
    end

    ff_storage = CMMStorage(
        num_fragments,
        SlaterShellDamping(),
        SlaterOverlapDamping(),
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # α_inv
        [LocalAxes() for _ in 1:natoms], # local axes
        [CSMultipole2() for _ in 1:natoms], # perm multipoles
        [CSMultipole1() for _ in 1:natoms], # induced multipoles
        zeros(natoms), # ϕ_core
        zeros(natoms), # ϕ_shell
        zeros(natoms), # ϕ_induced
        zeros(natoms), # ϕ_exch_pol
        zeros(natoms), # ϕ_dispersion
        zeros(natoms), # ϕ_repulsion
        zeros(natoms), # ϕ_donor
        zeros(natoms), # ϕ_acceptor
        [@MVector zeros(3) for _ in 1:natoms], # E_field_core
        [@MVector zeros(3) for _ in 1:natoms], # E_field_shell
        [@MVector zeros(3) for _ in 1:natoms], # E_field_induced
        [@MVector zeros(3) for _ in 1:natoms], # E_field_exch_pol
        [@MVector zeros(3) for _ in 1:natoms], # E_field_dispersion
        [@MVector zeros(3) for _ in 1:natoms], # E_field_repulsion
        [@MVector zeros(3) for _ in 1:natoms], # E_field_donor
        [@MVector zeros(3) for _ in 1:natoms], # E_field_acceptor
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_core
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_shell
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_induced
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_exch_pol
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_dispersion
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_repulsion
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_donor
        [@MMatrix zeros(3, 3) for _ in 1:natoms], # E_field_gradients_acceptor
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_core
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_shell
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_induced
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_exch_pol
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_dispersion
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_repulsion
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_donor
        [@MArray zeros(3, 3, 3) for _ in 1:natoms], # E_field_gradient_gradients_acceptor
        zeros(natoms), # Δq_ct
        zeros(4 * natoms + num_fragments), # b_vec
        zeros(4 * natoms + num_fragments, 4 * natoms + num_fragments), # polarization matrix
        zeros(4 * natoms + num_fragments), # residual for conjugate gradient
        zeros(3), # applied field
        zeros(natoms), # η_fq
        include_exch_pol,
        include_charge_transfer,
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
    ff_results = ForceFieldResults()
    return CMM_FF(ff_terms, params, ff_storage, ff_results)
end