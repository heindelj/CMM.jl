using Optim, Combinatorics, CSV, DataFrames, StatsBase, ProgressMeter, Printf, StaticArrays, CMM

include("damping.jl")
include("multipoles.jl")
include("damped_multipoles.jl")
include("force_field.jl")
include("parameters.jl")
include("utils.jl")
#include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/harmonic_frequencies.jl")
include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/molecular_axes.jl")

function write_mbe_for_dataset_to_csv(
    geom_file::String,
    csv_outfile::String,
    ff::CMM.CMM_FF,
    ff_pol::CMM.CMM_FF
)

    _, labels, geoms = CMM.read_xyz(geom_file)

    all_mbe_terms_and_energies = Dict{Symbol, Vector{Float64}}[]

    fragment_patterns = Dict(
        2 => [[1], [2]],
        3 => [[1], [2], [3]],
        4 => [[1,2,3], [4]],
        5 => [[1,2,3], [4], [5]],
        6 => [[1,2,3], [4,5,6]],
        7 => [[1], [2,3,4], [5,6,7]],
        8 => [[1,2,3], [4,5,6], [7], [8]],
        9 => [[1,2,3], [4,5,6], [7,8,9]],
        12 => [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]],
        15 => [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13,14,15]],
    )

    fragment_indices = fragment_patterns[length(labels[1])]
    mbe_terms_and_max_order = Dict(
        :Distortion     => 1,
        :Dispersion     => 2,
        :Pauli          => 2,
        :Electrostatics => 2,
        :Polarization   => 5,
        :ChargeTransfer => 5,
        :Total          => 5
    )
    
    if length(fragment_indices) < 5
        mbe_terms_and_max_order[:Polarization] = length(fragment_indices)
        mbe_terms_and_max_order[:ChargeTransfer] = length(fragment_indices)
        mbe_terms_and_max_order[:Total] = length(fragment_indices)
    end

    for i in eachindex(geoms)
        mbe_terms_and_energies = Dict(
            :Distortion     => zeros(mbe_terms_and_max_order[:Distortion] + 1),
            :Dispersion     => zeros(mbe_terms_and_max_order[:Dispersion] + 1),
            :Pauli          => zeros(mbe_terms_and_max_order[:Pauli] + 1),
            :Electrostatics => zeros(mbe_terms_and_max_order[:Electrostatics] + 1),
            :Polarization   => zeros(mbe_terms_and_max_order[:Polarization] + 1),
            :ChargeTransfer => zeros(mbe_terms_and_max_order[:ChargeTransfer] + 1),
            :Total          => zeros(mbe_terms_and_max_order[:Total] + 1)
        )
        for key in keys(mbe_terms_and_max_order)
            energies = mbe(geoms[i] / .529177, labels[i], fragment_indices, ff, key, mbe_terms_and_max_order[key])
            evaluate!(geoms[i] / .529177, labels[i], fragment_indices, ff)
            evaluate!(geoms[i] / .529177, labels[i], fragment_indices, ff_pol)
            for i_energy in eachindex(energies)
                mbe_terms_and_energies[key][i_energy] = energies[i_energy]
            end
            if key == :ChargeTransfer
                mbe_terms_and_energies[key][end] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
            elseif key == :Polarization
                mbe_terms_and_energies[key][end] = ff_pol.results.energies[key]
            else
                mbe_terms_and_energies[key][end] = ff.results.energies[key]
            end
        end
        push!(all_mbe_terms_and_energies, mbe_terms_and_energies)
    end

    all_keys = Symbol[]
    for key in keys(mbe_terms_and_max_order)
        if mbe_terms_and_max_order[key] > 1
            push!(all_keys, Symbol(key, "2B"))
            if mbe_terms_and_max_order[key] >= 3
                push!(all_keys, Symbol(key, "3B"))
            end
            if mbe_terms_and_max_order[key] >= 4
                push!(all_keys, Symbol(key, "4B"))
            end
            if mbe_terms_and_max_order[key] >= 5
                push!(all_keys, Symbol(key, "5B"))
            end
        end
        push!(all_keys, key)
    end

    all_data = Vector{Float64}[]
    for i in eachindex(geoms)
        mbe_data = Float64[]
        for key in keys(all_mbe_terms_and_energies[i])
            if mbe_terms_and_max_order[key] == 2
                append!(mbe_data, all_mbe_terms_and_energies[i][key][2])
                append!(mbe_data, all_mbe_terms_and_energies[i][key][end])
            elseif mbe_terms_and_max_order[key] >= 3
                append!(mbe_data, all_mbe_terms_and_energies[i][key][2:end])
            elseif mbe_terms_and_max_order[key] == 1
                append!(mbe_data, all_mbe_terms_and_energies[i][key][end])
            end
        end
        push!(all_data, mbe_data * 4.184) # convert to KJ/mol for consistency with Q-Chem
    end

    df = DataFrame(mapreduce(permutedims, vcat, all_data), all_keys)

    CSV.write(csv_outfile, df)
    return
end

function mae_for_dataset(
    geom_file::String,
    eda_file::String,
    fragment_indices::Vector{Vector{Int}},
    ff::CMM.CMM_FF,
    ff_no_ct::CMM.CMM_FF,
)
    _, labels, geoms = read_xyz(geom_file)
    eda_data = CSV.File(eda_file) |> DataFrame

    all_exch_data = zeros(length(geoms))
    all_disp_data = zeros(length(geoms))
    all_elec_data = zeros(length(geoms))
    all_pol_data  = zeros(length(geoms))
    all_ct_data   = zeros(length(geoms))

    @showprogress for i in eachindex(geoms)
        coords = [MVector{3,Float64}(geoms[i][:, j] / .529177) for j in eachindex(eachcol(geoms[i]))]
        evaluate!(coords, labels[i], fragment_indices, ff)
        evaluate!(coords, labels[i], fragment_indices, ff_no_ct)
        all_exch_data[i] = ff.results.energies[:Pauli]
        all_disp_data[i] = ff.results.energies[:Dispersion]
        all_elec_data[i] = ff.results.energies[:Electrostatics]
        all_pol_data[i] = ff_no_ct.results.energies[:Polarization]
        all_ct_data[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_no_ct.results.energies[:Polarization])
    end

    mae_dict = Dict(
        :Pauli => meanad(all_exch_data, eda_data[!, :mod_pauli] / 4.184),
        :Dispersion => meanad(all_disp_data, eda_data[!, :disp] / 4.184),
        :Polarization => meanad(all_pol_data, eda_data[!, :pol] / 4.184),
        :Electrostatics => meanad(all_elec_data, eda_data[!, :cls_elec] / 4.184),
        :ChargeTransfer => meanad(all_ct_data, eda_data[!, :ct] / 4.184)
    )

    return mae_dict
end

function optimize_xyz(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::CMM.CMM_FF,
    iterations::Int=2000,
    f_tol::Float64=1e-8,
    g_tol::Float64=1e-6,
    x_tol::Float64=1e-4,
    show_trace::Bool=true,
    show_every::Int=5,
    warn_about_units::Bool=true
)
    for i in eachindex(eachcol(coords))
        for j in eachindex(eachcol(coords))
            if warn_about_units
                if i != j && (norm(coords[i] - coords[j]) < 1.0)
                    @warn "Found short intermatomic distance of less than 1.0.
                    Did you pass the coordinates in angstrom instead of bohr?"
                    warn_about_units = false
                    break
                end
            end
        end
    end

    function fg!(F, G, x)
        coords = [MVector{3,Float64}(x[:, i]) for i in eachindex(eachcol(x))]
        energy = 0.0
        if G !== nothing
            switch_to_gradients!(ff)
            evaluate!(coords, labels, fragment_indices, ff)
            for i in eachindex(ff.results.grads)
                @views G[(3*i-2):3*i] = ff.results.grads[i]
            end
        end
        if F !== nothing
            if energy != 0.0
                return energy
            end
            switch_to_energies_only!(ff)
            evaluate!(coords, labels, fragment_indices, ff)
            return ff.results.energies[:Total] / 627.51
        end
    end

    results = optimize(
        Optim.only_fg!(fg!),
        reduce(hcat, coords),
        LBFGS(linesearch=Optim.LineSearches.HagerZhang()),
        Optim.Options(
            f_tol=f_tol,
            g_tol=g_tol,
            x_tol=x_tol,
            show_trace=show_trace,
            show_every=show_every,
            iterations=iterations))
    final_geom = Optim.minimizer(results)
    return (Optim.minimum(results) * 627.51, [MVector{3, Float64}(final_geom[:, i]) for i in eachindex(eachcol(final_geom))])
end

function optimize_xyz_by_fd(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::CMM.CMM_FF,
    iterations::Int=2000,
    f_tol::Float64=1e-8,
    g_tol::Float64=1e-6,
    x_tol::Float64=1e-4,
    show_trace::Bool=true,
    show_every::Int=5,
    warn_about_units::Bool=true
)
    for i in eachindex(coords)
        for j in eachindex(coords)
            if warn_about_units
                if i != j && (norm(coords[i] - coords[j]) < 1.0)
                    @warn "Found short intermatomic distance of less than 1.0.
                    Did you pass the coordinates in angstrom instead of bohr?"
                    warn_about_units = false
                    break
                end
            end
        end
    end

    function fg!(F, G, x)
        coords = [MVector{3,Float64}(x[:, i]) for i in eachindex(eachcol(x))]
        energy = 0.0
        if G !== nothing
            fd_grads = finite_difference_forces(coords, labels, fragment_indices, ff, :Total) / 627.51
            for i in eachindex(fd_grads)
                @views G[(3*i-2):3*i] = fd_grads[i]
            end
        end
        if F !== nothing
            if energy != 0.0
                return energy
            end
            evaluate!(coords, labels, fragment_indices, ff)
            return ff.results.energies[:Total] / 627.51
        end
    end

    results = optimize(
        Optim.only_fg!(fg!),
        reduce(hcat, coords),
        LBFGS(linesearch=Optim.LineSearches.HagerZhang()),
        Optim.Options(
            f_tol=f_tol,
            g_tol=g_tol,
            x_tol=x_tol,
            show_trace=show_trace,
            show_every=show_every,
            iterations=iterations))
    final_geom = Optim.minimizer(results)
    return (Optim.minimum(results) * 627.51, [MVector{3, Float64}(final_geom[:, i]) for i in eachindex(eachcol(final_geom))])
end

function optimize_all_reference_structures_and_write_to_file(ff::CMM.CMM_FF)
    include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/molecular_axes.jl")
    _, labels, geoms = read_xyz("assets/xyz/all_clusters.xyz")
    opt_geoms = Matrix{Float64}[]
    opt_energies = Float64[]
    opt_rmsd = Float64[]
    for i in eachindex(geoms)
        geom = [MVector{3, Float64}(geoms[i][:, j] / .529177) for j in eachindex(labels[i])]
        fragment_indices = [[i,i+1,i+2] for i in 1:3:length(geom)]
        opt_energy, opt_coords = optimize_xyz_by_fd(geom, labels[i], fragment_indices, ff)
        push!(opt_geoms, reduce(hcat, opt_coords * .529177))
        push!(opt_energies, opt_energy)
        push!(opt_rmsd, kabsch_rmsd(geoms[i], reduce(hcat, opt_coords * .529177)))
        display(opt_rmsd[end])
        display(opt_energy)
    end
    display(mean(opt_rmsd))
    display(opt_energies)
    display(opt_rmsd)
    write_xyz("optimized_reference_clusters.xyz", [string(length(labels[i]), "\nE = ", opt_energies[i]) for i in eachindex(opt_energies)], labels, opt_geoms)
    return opt_energies, opt_rmsd
end

function write_csv_with_force_field_energies(
    geom_file::String,
    eda_outfile::String,
    ff::CMM.CMM_FF,
    ff_no_ct::CMM.CMM_FF
)
    _, labels, geoms = CMM.read_xyz(geom_file)

    fragment_patterns = Dict(
        2 => [[1], [2]],
        3 => [[1], [2], [3]],
        4 => [[1,2,3], [4]],
        5 => [[1,2,3], [4], [5]],
        6 => [[1,2,3], [4,5,6]],
        7 => [[1], [2,3,4], [5,6,7]],
        8 => [[1,2,3], [4,5,6], [7], [8]],
        9 => [[1,2,3], [4,5,6], [7,8,9]],
        12 => [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]],
        15 => [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13,14,15]],
    )

    fragment_indices = fragment_patterns[length(labels[1])]

    all_exch_data = zeros(length(geoms))
    all_disp_data = zeros(length(geoms))
    all_elec_data = zeros(length(geoms))
    all_pol_data  = zeros(length(geoms))
    all_ct_data   = zeros(length(geoms))

    @showprogress for i in eachindex(geoms)
        coords = geoms[i] / .529177
        evaluate!(coords, labels[i], fragment_indices, ff)
        evaluate!(coords, labels[i], fragment_indices, ff_no_ct)
        all_exch_data[i] = ff.results.energies[:Pauli]
        all_disp_data[i] = ff.results.energies[:Dispersion]
        all_elec_data[i] = ff.results.energies[:Electrostatics]
        all_pol_data[i] = ff_no_ct.results.energies[:Polarization]
        all_ct_data[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_no_ct.results.energies[:Polarization])
    end

    df = DataFrame(
        :cls_elec => all_elec_data * 4.184,
        :mod_pauli => all_exch_data * 4.184,
        :disp => all_disp_data * 4.184,
        :pol => all_pol_data * 4.184,
        :ct => all_ct_data * 4.184
    )
    CSV.write(eda_outfile, df)
    return
end

function get_force_components(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::CMM.CMM_FF,
    ff_no_ct::Union{Nothing, CMM.CMM_FF}
)
    force_dict = Dict{Symbol, Vector{MVector{3, Float64}}}(
        :Deformation => [@MVector zeros(3) for _ in eachindex(coords)],
        :Pauli => [@MVector zeros(3) for _ in eachindex(coords)],
        :Dispersion => [@MVector zeros(3) for _ in eachindex(coords)],
        :Electrostatics => [@MVector zeros(3) for _ in eachindex(coords)],
        :Polarization => [@MVector zeros(3) for _ in eachindex(coords)],
        :ChargeTransfer => [@MVector zeros(3) for _ in eachindex(coords)],
        :Total => [@MVector zeros(3) for _ in eachindex(coords)]
    )

    evaluate!(coords, labels, fragment_indices, ff)
    
    if ff_no_ct !== nothing
        evaluate!(coords, labels, fragment_indices, ff_no_ct)

        force_dict[:Deformation] = -ff.storage.deformation_grads * 627.51 * 4.184 / .529177
        force_dict[:Pauli] = -ff.storage.pauli_grads * 627.51 * 4.184 / .529177
        force_dict[:Dispersion] = -ff.storage.dispersion_grads * 627.51 * 4.184 / .529177
        force_dict[:Electrostatics] = -ff.storage.electrostatic_grads * 627.51 * 4.184 / .529177
        force_dict[:Polarization] = -ff_no_ct.storage.polarization_grads * 627.51 * 4.184 / .529177
        force_dict[:ChargeTransfer] = -(ff.storage.charge_transfer_grads + (ff.storage.polarization_grads - ff_no_ct.storage.polarization_grads)) * 627.51 * 4.184 / .529177
        force_dict[:Total] = -ff.results.grads * 627.51 * 4.184 / .529177
    else
        force_dict[:Deformation] = -ff.storage.deformation_grads * 627.51 * 4.184 / .529177
        force_dict[:Pauli] = -ff.storage.pauli_grads * 627.51 * 4.184 / .529177
        force_dict[:Dispersion] = -ff.storage.dispersion_grads * 627.51 * 4.184 / .529177
        force_dict[:Electrostatics] = -ff.storage.electrostatic_grads * 627.51 * 4.184 / .529177
        force_dict[:Polarization] = -ff.storage.polarization_grads * 627.51 * 4.184 / .529177
        force_dict[:ChargeTransfer] = -ff.storage.charge_transfer_grads * 627.51 * 4.184 / .529177
        force_dict[:Total] = -ff.results.grads * 627.51 * 4.184 / .529177
    end

    return force_dict
end

function harmonic_analysis_fqct(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, fragment_indices::Vector{Vector{Int}}, ff::CMM.CMM_FF)
    @assert ff.results.grads === nothing "Haven't implemented version based on gradients yet. Please pass force field with gradients disabled."
    function potential(new_coords::Matrix{Float64})
        @views static_coords = [MVector{3, Float64}(new_coords[:, i]) / 0.529177 for i in eachindex(eachcol(new_coords))]
        evaluate!(static_coords, labels, fragment_indices, ff)
        return ff.results.energies[:Total] / 627.51
    end
    evals, evecs, reduced_masses = harmonic_analysis(potential, reduce(hcat, coords * 0.529177), labels, false)
    return evals, evecs, reduced_masses
end

function find_hbonded_atoms_and_oh_distance_from_normal_modes(
    opt_geom::Vector{MVector{3, Float64}},
    labels::Vector{String},
    evals::Vector{ComplexF64},
    evecs::Vector{Vector{Float64}},
    minimum_hbond_frequency::Float64=2500.0, #cm^-1
    maximum_hbond_frequency::Float64=3860.0 #cm^-1
)
    # Expects input in bohr! Will be converted to angstrom when stored.

    @assert 3 * length(opt_geom) == length(evals) "Number of eigenvalues and number of degreees of freedom are not equal. Make sure the geometry and eigenvalues are paired up correctly."

    bond_and_freqs = Tuple{Float64, Float64}[]

    nearest_oxygen_distances = Dict{Int, Float64}()
    for i in eachindex(opt_geom)
        h_index = 0
        if labels[i] == "H"
            nearest_oxygen_distances[i] = 10000.0 # large default
            for j in eachindex(opt_geom)
                if labels[j] == "O"
                    dist = norm(opt_geom[i] - opt_geom[j]) * 0.529177
                    if dist < nearest_oxygen_distances[i]
                        nearest_oxygen_distances[i] = dist
                    end
                end
            end
        end
    end

    evals = [evals[i].re for i in eachindex(evals)]
    for i in eachindex(evals)
        if evals[i] > minimum_hbond_frequency && evals[i] < maximum_hbond_frequency
            evec_lengths = norm.(eachcol(reshape(evecs[i], (3, :))))
            _, atom_index = findmax(evec_lengths)
            @assert labels[atom_index] == "H" "Atom moving the most in h-bond wasn't a hydrogen!"
            push!(bond_and_freqs, (nearest_oxygen_distances[atom_index], evals[i]))
        end
    end
    return bond_and_freqs
end

function optimize_structures_and_write_bond_length_frequency_correlation(
    outfile::String,
    geom_file::String,
    ff_energy::CMM.CMM_FF,
    ff_with_grads::Union{CMM.CMM_FF, Nothing}
)
    _, labels, geoms = read_xyz(geom_file, static=true)
    opt_geoms = Vector{MVector{3, Float64}}[]
    @showprogress for i in eachindex(geoms)
        fragment_indices = [[i,i+1,i+2] for i in 1:3:length(geoms[i])]
        if ff_with_grads !== nothing
            opt_energy, opt_coords = optimize_xyz(geoms[i] / .529177, labels[i], fragment_indices, ff_with_grads)
        else
            opt_energy, opt_coords = optimize_xyz_by_fd(geoms[i] / .529177, labels[i], fragment_indices, ff_energy)
        end
        push!(opt_geoms, opt_coords)
    end
    all_bond_and_freqs = Tuple{Float64, Float64}[]
    println("Beginning Harmonic Frequncy Analyses:")
    @showprogress for i in eachindex(opt_geoms)
        fragment_indices = [[i,i+1,i+2] for i in 1:3:length(opt_geoms[i])]
        evals, evecs, _ = harmonic_analysis_fqct(opt_geoms[i], labels[i], fragment_indices, ff_energy)
        bond_and_freqs = find_hbonded_atoms_and_oh_distance_from_normal_modes(
            opt_geoms[i], labels[i], evals, evecs
        )
        display(bond_and_freqs)
        append!(all_bond_and_freqs, bond_and_freqs)
    end

    bond_lengths, freqs = collect(zip(all_bond_and_freqs...))
    bond_lengths = [bond_lengths...]
    freqs = [freqs...]
    df = DataFrame(
        :bond_lengths => bond_lengths,
        :freqs => freqs
    )
    CSV.write(outfile, df)
    write_xyz(string(splitext(outfile)[1], "_opt_geoms.xyz"), labels, [reduce(hcat, geom * 0.529177) for geom in opt_geoms])
    return
end

function mae_over_water_datasets_for_all_terms()
    mae_by_term = Dict(
        #:Distortion => zeros(4),
        :Pauli => zeros(4),
        :Electrostatics => zeros(4),
        :Dispersion => zeros(4),
        :Polarization => zeros(4),
        :ChargeTransfer => zeros(4),
        :Interaction => zeros(4),
    )

    skewness_by_term = Dict(
        #:Distortion => zeros(4),
        :Pauli => zeros(4),
        :Electrostatics => zeros(4),
        :Dispersion => zeros(4),
        :Polarization => zeros(4),
        :ChargeTransfer => zeros(4),
        :Interaction => zeros(4),
    )

    ff_pol = build_custom_fqct_aniso_model(
        false,
        [reset_storage!, get_one_body_properties!,
        permanent_electrostatics!, polarization_energy!],
        get_parameter_dict_aniso_v2()
    )
    ff = build_fqct_aniso_model(false)
    
    # Read in the datasets #
    w2_eda_meili = CSV.File("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w2_meili.csv") |> DataFrame
    _, w2_labels_meili, w2_geoms_meili = read_xyz("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w2_meili.xyz", static=true)

    w2_eda_sobol = CSV.File("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w2_sobol_dimers.csv") |> DataFrame
    _, w2_labels_sobol, w2_geoms_sobol = read_xyz("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w2_sobol_dimers.xyz", static=true)

    w3_eda_meili = CSV.File("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w3_meili.csv") |> DataFrame
    _, w3_labels_meili, w3_geoms_meili = read_xyz("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w3_meili.xyz", static=true)

    w4_eda_meili = CSV.File("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w4_meili.csv") |> DataFrame
    _, w4_labels_meili, w4_geoms_meili = read_xyz("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w4_meili.xyz", static=true)

    w5_eda_meili = CSV.File("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w5_meili.csv") |> DataFrame
    _, w5_labels_meili, w5_geoms_meili = read_xyz("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/w5_meili.xyz", static=true)

    # Get all non charge transfer contributions #
    all_dimer_elec_energies = zeros(length(w2_geoms_meili))
    all_dimer_pauli_energies = zeros(length(w2_geoms_meili))
    all_dimer_disp_energies = zeros(length(w2_geoms_meili))
    all_dimer_pol_energies = zeros(length(w2_geoms_meili))
    all_dimer_ct_energies = zeros(length(w2_geoms_meili))
    all_dimer_int_energies = zeros(length(w2_geoms_meili))

    all_trimer_elec_energies = zeros(length(w3_geoms_meili))
    all_trimer_pauli_energies = zeros(length(w3_geoms_meili))
    all_trimer_disp_energies = zeros(length(w3_geoms_meili))
    all_trimer_pol_energies = zeros(length(w3_geoms_meili))
    all_trimer_ct_energies = zeros(length(w3_geoms_meili))
    all_trimer_int_energies = zeros(length(w3_geoms_meili))

    all_tetramer_elec_energies = zeros(length(w4_geoms_meili))
    all_tetramer_pauli_energies = zeros(length(w4_geoms_meili))
    all_tetramer_disp_energies = zeros(length(w4_geoms_meili))
    all_tetramer_pol_energies = zeros(length(w4_geoms_meili))
    all_tetramer_ct_energies = zeros(length(w4_geoms_meili))
    all_tetramer_int_energies = zeros(length(w4_geoms_meili))

    all_pentamer_elec_energies = zeros(length(w5_geoms_meili))
    all_pentamer_pauli_energies = zeros(length(w5_geoms_meili))
    all_pentamer_disp_energies = zeros(length(w5_geoms_meili))
    all_pentamer_pol_energies = zeros(length(w5_geoms_meili))
    all_pentamer_ct_energies = zeros(length(w5_geoms_meili))
    all_pentamer_int_energies = zeros(length(w5_geoms_meili))

    for i in eachindex(w2_geoms_meili)
        evaluate!(
            w2_geoms_meili[i] / .529177, w2_labels_meili[i],
            [[1,2,3], [4,5,6]], ff_pol
        )
        evaluate!(
            w2_geoms_meili[i] / .529177, w2_labels_meili[i],
            [[1,2,3], [4,5,6]], ff
        )
        all_dimer_elec_energies[i] = ff_pol.results.energies[:Electrostatics]
        all_dimer_pauli_energies[i] = ff.results.energies[:Pauli]
        all_dimer_disp_energies[i] = ff.results.energies[:Dispersion]
        all_dimer_pol_energies[i] = ff_pol.results.energies[:Polarization]
        all_dimer_ct_energies[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        all_dimer_int_energies[i] = ff.results.energies[:Interaction]
    end

    for i in eachindex(w3_geoms_meili)
        evaluate!(
            w3_geoms_meili[i] / .529177, w3_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9]], ff_pol
        )
        evaluate!(
            w3_geoms_meili[i] / .529177, w3_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9]], ff
        )
        all_trimer_elec_energies[i] = ff_pol.results.energies[:Electrostatics]
        all_trimer_pauli_energies[i] = ff.results.energies[:Pauli]
        all_trimer_disp_energies[i] = ff.results.energies[:Dispersion]
        all_trimer_pol_energies[i] = ff_pol.results.energies[:Polarization]
        all_trimer_ct_energies[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        all_trimer_int_energies[i] = ff.results.energies[:Interaction]
    end

    for i in eachindex(w4_geoms_meili)
        evaluate!(
            w4_geoms_meili[i] / .529177, w4_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9], [10,11,12]], ff_pol
        )
        evaluate!(
            w4_geoms_meili[i] / .529177, w4_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9], [10,11,12]], ff
        )
        all_tetramer_elec_energies[i] = ff_pol.results.energies[:Electrostatics]
        all_tetramer_pauli_energies[i] = ff.results.energies[:Pauli]
        all_tetramer_disp_energies[i] = ff.results.energies[:Dispersion]
        all_tetramer_pol_energies[i] = ff_pol.results.energies[:Polarization]
        all_tetramer_ct_energies[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        all_tetramer_int_energies[i] = ff.results.energies[:Interaction]
    end

    for i in eachindex(w5_geoms_meili)
        evaluate!(
            w5_geoms_meili[i] / .529177, w5_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]], ff_pol
        )
        evaluate!(
            w5_geoms_meili[i] / .529177, w5_labels_meili[i],
            [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]], ff
        )
        all_pentamer_elec_energies[i] = ff_pol.results.energies[:Electrostatics]
        all_pentamer_pauli_energies[i] = ff.results.energies[:Pauli]
        all_pentamer_disp_energies[i] = ff.results.energies[:Dispersion]
        all_pentamer_pol_energies[i] = ff_pol.results.energies[:Polarization]
        all_pentamer_ct_energies[i] = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        all_pentamer_int_energies[i] = ff.results.energies[:Interaction]
    end

    mae_by_term[:Electrostatics][1] = meanad(all_dimer_elec_energies, w2_eda_meili[!, :cls_elec] / 4.184)
    mae_by_term[:Pauli][1] = meanad(all_dimer_pauli_energies, w2_eda_meili[!, :mod_pauli] / 4.184)
    mae_by_term[:Dispersion][1] = meanad(all_dimer_disp_energies, w2_eda_meili[!, :disp] / 4.184)
    mae_by_term[:Polarization][1] = meanad(all_dimer_pol_energies, w2_eda_meili[!, :pol] / 4.184)
    mae_by_term[:ChargeTransfer][1] = meanad(all_dimer_ct_energies, w2_eda_meili[!, :ct] / 4.184)
    mae_by_term[:Interaction][1] = meanad(all_dimer_int_energies, w2_eda_meili[!, :int] / 4.184)

    mae_by_term[:Electrostatics][2] = meanad(all_trimer_elec_energies, w3_eda_meili[!, :cls_elec] / 4.184)
    mae_by_term[:Pauli][2] = meanad(all_trimer_pauli_energies, w3_eda_meili[!, :mod_pauli] / 4.184)
    mae_by_term[:Dispersion][2] = meanad(all_trimer_disp_energies, w3_eda_meili[!, :disp] / 4.184)
    mae_by_term[:Polarization][2] = meanad(all_trimer_pol_energies, w3_eda_meili[!, :pol] / 4.184)
    mae_by_term[:ChargeTransfer][2] = meanad(all_trimer_ct_energies, w3_eda_meili[!, :ct] / 4.184)
    mae_by_term[:Interaction][2] = meanad(all_trimer_int_energies, w3_eda_meili[!, :int] / 4.184)

    mae_by_term[:Electrostatics][3] = meanad(all_tetramer_elec_energies, w4_eda_meili[!, :cls_elec] / 4.184)
    mae_by_term[:Pauli][3] = meanad(all_tetramer_pauli_energies, w4_eda_meili[!, :mod_pauli] / 4.184)
    mae_by_term[:Dispersion][3] = meanad(all_tetramer_disp_energies, w4_eda_meili[!, :disp] / 4.184)
    mae_by_term[:Polarization][3] = meanad(all_tetramer_pol_energies, w4_eda_meili[!, :pol] / 4.184)
    mae_by_term[:ChargeTransfer][3] = meanad(all_tetramer_ct_energies, w4_eda_meili[!, :ct] / 4.184)
    mae_by_term[:Interaction][3] = meanad(all_tetramer_int_energies, w4_eda_meili[!, :int] / 4.184)

    mae_by_term[:Electrostatics][4] = meanad(all_pentamer_elec_energies, w5_eda_meili[!, :cls_elec] / 4.184)
    mae_by_term[:Pauli][4] = meanad(all_pentamer_pauli_energies, w5_eda_meili[!, :mod_pauli] / 4.184)
    mae_by_term[:Dispersion][4] = meanad(all_pentamer_disp_energies, w5_eda_meili[!, :disp] / 4.184)
    mae_by_term[:Polarization][4] = meanad(all_pentamer_pol_energies, w5_eda_meili[!, :pol] / 4.184)
    mae_by_term[:ChargeTransfer][4] = meanad(all_pentamer_ct_energies, w5_eda_meili[!, :ct] / 4.184)
    mae_by_term[:Interaction][4] = meanad(all_pentamer_int_energies, w5_eda_meili[!, :int] / 4.184)

    skewness_by_term[:Electrostatics][1] = skewness(-(all_dimer_elec_energies, w2_eda_meili[!, :cls_elec] / 4.184))
    skewness_by_term[:Pauli][1] = skewness(-(all_dimer_pauli_energies, w2_eda_meili[!, :mod_pauli] / 4.184))
    skewness_by_term[:Dispersion][1] = skewness(-(all_dimer_disp_energies, w2_eda_meili[!, :disp] / 4.184))
    skewness_by_term[:Polarization][1] = skewness(-(all_dimer_pol_energies, w2_eda_meili[!, :pol] / 4.184))
    skewness_by_term[:ChargeTransfer][1] = skewness(-(all_dimer_ct_energies, w2_eda_meili[!, :ct] / 4.184))
    skewness_by_term[:Interaction][1] = skewness(-(all_dimer_int_energies, w2_eda_meili[!, :int] / 4.184))

    skewness_by_term[:Electrostatics][2] = skewness(-(all_trimer_elec_energies, w3_eda_meili[!, :cls_elec] / 4.184))
    skewness_by_term[:Pauli][2] = skewness(-(all_trimer_pauli_energies, w3_eda_meili[!, :mod_pauli] / 4.184))
    skewness_by_term[:Dispersion][2] = skewness(-(all_trimer_disp_energies, w3_eda_meili[!, :disp] / 4.184))
    skewness_by_term[:Polarization][2] = skewness(-(all_trimer_pol_energies, w3_eda_meili[!, :pol] / 4.184))
    skewness_by_term[:ChargeTransfer][2] = skewness(-(all_trimer_ct_energies, w3_eda_meili[!, :ct] / 4.184))
    skewness_by_term[:Interaction][2] = skewness(-(all_trimer_int_energies, w3_eda_meili[!, :int] / 4.184))

    skewness_by_term[:Electrostatics][3] = skewness(-(all_tetramer_elec_energies, w4_eda_meili[!, :cls_elec] / 4.184))
    skewness_by_term[:Pauli][3] = skewness(-(all_tetramer_pauli_energies, w4_eda_meili[!, :mod_pauli] / 4.184))
    skewness_by_term[:Dispersion][3] = skewness(-(all_tetramer_disp_energies, w4_eda_meili[!, :disp] / 4.184))
    skewness_by_term[:Polarization][3] = skewness(-(all_tetramer_pol_energies, w4_eda_meili[!, :pol] / 4.184))
    skewness_by_term[:ChargeTransfer][3] = skewness(-(all_tetramer_ct_energies, w4_eda_meili[!, :ct] / 4.184))
    skewness_by_term[:Interaction][3] = skewness(-(all_tetramer_int_energies, w4_eda_meili[!, :int] / 4.184))

    skewness_by_term[:Electrostatics][4] = skewness(-(all_pentamer_elec_energies, w5_eda_meili[!, :cls_elec] / 4.184))
    skewness_by_term[:Pauli][4] = skewness(-(all_pentamer_pauli_energies, w5_eda_meili[!, :mod_pauli] / 4.184))
    skewness_by_term[:Dispersion][4] = skewness(-(all_pentamer_disp_energies, w5_eda_meili[!, :disp] / 4.184))
    skewness_by_term[:Polarization][4] = skewness(-(all_pentamer_pol_energies, w5_eda_meili[!, :pol] / 4.184))
    skewness_by_term[:ChargeTransfer][4] = skewness(-(all_pentamer_ct_energies, w5_eda_meili[!, :ct] / 4.184))
    skewness_by_term[:Interaction][4] = skewness(-(all_pentamer_int_energies, w5_eda_meili[!, :int] / 4.184))

    return mae_by_term, skewness_by_term
end

function get_all_mbe_terms_for_xyz_file(cluster_file::String)
    ff = build_cmm_model()
    ff_pol = build_cmm_model(
               false, true, false,
               [reset_storage!, get_one_body_properties!,
               permanent_electrostatics!, polarization_energy!]
           )
    _, labels, geoms = read_xyz(cluster_file, static=true)
    
    total_elec_energies      = zeros(length(geoms))
    many_body_elec_energies  = zeros(length(geoms))
    two_body_elec_energies   = zeros(length(geoms))
    total_pauli_energies     = zeros(length(geoms))
    many_body_pauli_energies = zeros(length(geoms))
    two_body_pauli_energies  = zeros(length(geoms))
    total_disp_energies      = zeros(length(geoms))
    many_body_disp_energies  = zeros(length(geoms))
    two_body_disp_energies   = zeros(length(geoms))
    total_pol_energies       = zeros(length(geoms))
    many_body_pol_energies   = zeros(length(geoms))
    two_body_pol_energies    = zeros(length(geoms))
    total_ct_energies        = zeros(length(geoms))
    many_body_ct_energies    = zeros(length(geoms))
    two_body_ct_energies     = zeros(length(geoms))
    total_int_energies = zeros(length(geoms))
    many_body_int_energies = zeros(length(geoms))
    two_body_int_energies = zeros(length(geoms))
    
    all_results = Dict{Symbol, Vector{Float64}}[]

    for i in eachindex(geoms)
        results = Dict{Symbol, Vector{Float64}}(
            :Electrostatics => zeros(3),
            :Pauli => zeros(3),
            :Dispersion => zeros(3),
            :Polarization => zeros(3),
            :ChargeTransfer => zeros(3),
        )
        evaluate!(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff)
        evaluate!(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff_pol)
        total_elec_energies[i]  = ff.results.energies[:Electrostatics]
        total_pauli_energies[i] = ff.results.energies[:Pauli]
        total_disp_energies[i]  = ff.results.energies[:Dispersion]
        total_pol_energies[i]   = ff_pol.results.energies[:Polarization]
        total_ct_energies[i]    = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        total_int_energies[i]  = ff.results.energies[:Interaction]

        two_body_elec_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :Electrostatics, 2)[2]
        two_body_pauli_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :Pauli, 2)[2]
        two_body_disp_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :Dispersion, 2)[2]
        two_body_pol_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :Polarization, 2)[2]
        two_body_ct_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :ChargeTransfer, 2)[2]
        two_body_int_energies[i] = mbe(geoms[i] / .529177, labels[i], [[j, j+1, j+2] for j in 1:3:length(labels[i])], ff, :Interaction, 2)[2]

        many_body_elec_energies[i] = total_elec_energies[i] - two_body_elec_energies[i]
        many_body_pauli_energies[i] = total_pauli_energies[i] - two_body_pauli_energies[i]
        many_body_disp_energies[i] = total_disp_energies[i] - two_body_disp_energies[i]
        many_body_pol_energies[i] = total_pol_energies[i] - two_body_pol_energies[i]
        many_body_ct_energies[i] = total_ct_energies[i] - two_body_ct_energies[i]
        many_body_int_energies[i] = total_int_energies[i] - two_body_int_energies[i]

        results[:Electrostatics] = [two_body_elec_energies[i], many_body_elec_energies[i], total_elec_energies[i]]
        results[:Pauli] = [two_body_pauli_energies[i], many_body_pauli_energies[i], total_pauli_energies[i]]
        results[:Dispersion] = [two_body_disp_energies[i], many_body_disp_energies[i], total_disp_energies[i]]
        results[:Polarization] = [two_body_pol_energies[i], many_body_pol_energies[i], total_pol_energies[i]]
        results[:ChargeTransfer] = [two_body_ct_energies[i], many_body_ct_energies[i], total_ct_energies[i]]
        results[:Interaction] = [two_body_int_energies[i], many_body_int_energies[i], total_int_energies[i]]
        push!(all_results, results)
    end

    return all_results
end

function get_mbe_terms_for_table_in_paper(cluster_file::String)
    ff = build_fqct_aniso_model(false)
    ff_pol = build_custom_fqct_aniso_model(
               false,
               [reset_storage!, get_one_body_properties!,
               permanent_electrostatics!, polarization_energy!],
               get_parameter_dict_aniso_v2()
           )
    _, labels, geoms = read_xyz(cluster_file, static=true)
    cluster_indices = [2, 5, 6, 13, 15, 21]
    
    total_elec_energies      = zeros(length(cluster_indices))
    many_body_elec_energies  = zeros(length(cluster_indices))
    two_body_elec_energies   = zeros(length(cluster_indices))
    total_pauli_energies     = zeros(length(cluster_indices))
    many_body_pauli_energies = zeros(length(cluster_indices))
    two_body_pauli_energies  = zeros(length(cluster_indices))
    total_disp_energies      = zeros(length(cluster_indices))
    many_body_disp_energies  = zeros(length(cluster_indices))
    two_body_disp_energies   = zeros(length(cluster_indices))
    total_pol_energies       = zeros(length(cluster_indices))
    many_body_pol_energies   = zeros(length(cluster_indices))
    two_body_pol_energies    = zeros(length(cluster_indices))
    total_ct_energies        = zeros(length(cluster_indices))
    many_body_ct_energies    = zeros(length(cluster_indices))
    two_body_ct_energies     = zeros(length(cluster_indices))
    total_int_energies = zeros(length(cluster_indices))
    many_body_int_energies = zeros(length(cluster_indices))
    two_body_int_energies = zeros(length(cluster_indices))
    
    all_results = Dict{Symbol, Vector{Float64}}[]

    for i in eachindex(cluster_indices)
        results = Dict{Symbol, Vector{Float64}}(
            :Electrostatics => zeros(3),
            :Pauli => zeros(3),
            :Dispersion => zeros(3),
            :Polarization => zeros(3),
            :ChargeTransfer => zeros(3),
        )
        i_geom = cluster_indices[i]
        evaluate!(geoms[i_geom] / .529177, labels[i_geom], [[j, j+1, j+2] for j in 1:3:length(labels[i_geom])], ff)
        evaluate!(geoms[i_geom] / .529177, labels[i_geom], [[j, j+1, j+2] for j in 1:3:length(labels[i_geom])], ff_pol)
        total_elec_energies[i]  = ff.results.energies[:Electrostatics]
        total_pauli_energies[i] = ff.results.energies[:Pauli]
        total_disp_energies[i]  = ff.results.energies[:Dispersion]
        total_pol_energies[i]   = ff_pol.results.energies[:Polarization]
        total_ct_energies[i]    = ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_pol.results.energies[:Polarization])
        total_int_energies[i]  = ff.results.energies[:Interaction]

        two_body_elec_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :Electrostatics, 2)[2]
        two_body_pauli_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :Pauli, 2)[2]
        two_body_disp_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :Dispersion, 2)[2]
        two_body_pol_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :Polarization, 2)[2]
        two_body_ct_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :ChargeTransfer, 2)[2]
        two_body_int_energies[i] = mbe(geoms[i_geom] / .529177, labels[i_geom], [[i, i+1, i+2] for i in 1:3:length(labels[i_geom])], ff, :Interaction, 2)[2]

        many_body_elec_energies[i] = total_elec_energies[i] - two_body_elec_energies[i]
        many_body_pauli_energies[i] = total_pauli_energies[i] - two_body_pauli_energies[i]
        many_body_disp_energies[i] = total_disp_energies[i] - two_body_disp_energies[i]
        many_body_pol_energies[i] = total_pol_energies[i] - two_body_pol_energies[i]
        many_body_ct_energies[i] = total_ct_energies[i] - two_body_ct_energies[i]
        many_body_int_energies[i] = total_int_energies[i] - two_body_int_energies[i]

        results[:Electrostatics] = [two_body_elec_energies[i], many_body_elec_energies[i], total_elec_energies[i]]
        results[:Pauli] = [two_body_pauli_energies[i], many_body_pauli_energies[i], total_pauli_energies[i]]
        results[:Dispersion] = [two_body_disp_energies[i], many_body_disp_energies[i], total_disp_energies[i]]
        results[:Polarization] = [two_body_pol_energies[i], many_body_pol_energies[i], total_pol_energies[i]]
        results[:ChargeTransfer] = [two_body_ct_energies[i], many_body_ct_energies[i], total_ct_energies[i]]
        results[:Interaction] = [two_body_int_energies[i], many_body_int_energies[i], total_int_energies[i]]
        push!(all_results, results)
    end

    return all_results
end

function ion_water_optimization_and_harmonic_frequencies()
    ff = build_fqct_aniso_model(false)
    ions = ["f", "cl", "br", "i", "li", "na", "k", "rb", "cs"]
    all_freqs = zeros(length(ions), 6)
    all_binding_energies = zeros(length(ions))
    for (i, ion) in enumerate(ions)
        _, h2o_ion_labels_scan, h2o_ion_geoms_scan = read_xyz(
            string("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/scans/", "h2o_", ion, "_scan.xyz"),
            static=true
        )
        ion_water_de, opt_ion_geom = optimize_xyz_by_fd(
            h2o_ion_geoms_scan[7] / .529177, h2o_ion_labels_scan[7], [[1,2,3], [4]], ff
        )
        evals, _, _ = harmonic_analysis_fqct(opt_ion_geom, h2o_ion_labels_scan[1], [[1,2,3], [4]], ff)
        all_binding_energies[i] = ion_water_de
        all_freqs[i, :] = Real.(evals[end-5:end])
    end

    df = DataFrame(
        :ion => ions,
        :NM1 => Int.(round.(all_freqs[:, 1])),
        :NM2 => Int.(round.(all_freqs[:, 2])),
        :NM3 => Int.(round.(all_freqs[:, 3])),
        :NM4 => Int.(round.(all_freqs[:, 4])),
        :NM5 => Int.(round.(all_freqs[:, 5])),
        :NM6 => Int.(round.(all_freqs[:, 6])),
        :De  => all_binding_energies
    )
    CSV.write("optimized_ion_frequencies_and_binding_energies.csv", df)
end

function ion_water_dimer_polarizability_comparison()
    ff = build_fqct_aniso_model(false)
    ions = ["f", "cl", "br", "i", "li", "na", "k", "rb", "cs"]
    for (i, ion) in enumerate(ions)
        _, h2o_ion_labels, h2o_ion_geom = read_xyz(
            string("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/polarizabilities/", "h2o_", ion, "_wb97xv_qzvppd.xyz"),
            static=true
        )
        polarizability_ref = readdlm(string("/home/heindelj/dev/julia_development/reactive_force_fields/notebooks/data/polarizabilities/", "h2o_", ion, "_wb97xv_qzvppd_polarizability.txt"))
        pol_components_ref, pol_vecs_ref = eigen(polarizability_ref)
        
        polarizability_cmm = fluctuating_charge_molecular_polarizability(h2o_ion_geom[1] / .529177, h2o_ion_labels[1], [[1,2,3], [4]], ff)
        pol_components_cmm, pol_vecs_cmm = eigen(polarizability_cmm)
        @printf "Ion-Water Polarizability: %s\n" titlecase(ion)
        @printf "wB97XV: %.2lf %.2lf %.2lf | Iso. Pol. = %.2lf\n" pol_components_ref[1] pol_components_ref[2] pol_components_ref[3] mean(pol_components_ref)
        @printf "CMM:    %.2lf %.2lf %.2lf | Iso. Pol. = %.2lf\n\n" pol_components_cmm[1] pol_components_cmm[2] pol_components_cmm[3] mean(pol_components_cmm)
        #@printf "Rel. Angle: %.3lf %.3lf %.3lf\n" pol_vecs_ref[:,1] ⋅ pol_vecs_cmm[:,1] pol_vecs_ref[:,2] ⋅ pol_vecs_cmm[:,2] pol_vecs_ref[:,3] ⋅ pol_vecs_cmm[:,3]
    end
end