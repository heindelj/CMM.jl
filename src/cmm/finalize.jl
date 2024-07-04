
function finalize!(ff::AbstractForceField)
    # accumulate all component gradients into total gradients #
    if ff.results.grads !== nothing
        @views ff.results.grads[:] += (
            ff.storage.deformation_grads +
            ff.storage.pauli_grads +
            ff.storage.dispersion_grads +
            ff.storage.electrostatic_grads +
            ff.storage.polarization_grads +
            ff.storage.charge_transfer_grads
        )
    end

    # convert all energies for kcal/mol just for convenience
    for key in keys(ff.results.energies)
        ff.results.energies[key] *= 627.51
    end
    if haskey(ff.results.energies, :Distortion)
        ff.results.energies[:Interaction] = ff.results.energies[:Total] - ff.results.energies[:Distortion]
    end
    return
end