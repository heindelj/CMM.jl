using Optim, Combinatorics, CSV, DataFrames, ProgressMeter, StaticArrays, CMM

function write_mbe_for_dataset_to_csv(
    geom_file::String,
    csv_outfile::String,
    ff::CMM.CMM_FF,
    ff_pol::CMM.CMM_FF
)

    _, labels, geoms = CMM.read_xyz(geom_file)

    all_mbe_terms_and_energies = Dict{Symbol, Vector{Float64}}[]

    function fragment_builder(labels::Vector{String})
        fragment_indices = Vector{Int}[]
        for i in eachindex(labels)
            if labels[i] == "O"
                push!(fragment_indices, [i, i+1, i+2])
                i += 2
            elseif CMM.is_ion(labels[i])
                push!(fragment_indices, [i])
            end
        end
        return fragment_indices
    end

    fragment_indices = fragment_builder(labels[1])
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

    @showprogress for i in eachindex(geoms)
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

function main(xyz_file::String)
    _, labels, geoms = CMM.read_xyz(xyz_file)
    ff = build_cmm_model(false, custom_params=CMM.get_parameter_dict_with_exch_pol_from_original_submission())
    ff_pol = build_cmm_model_POL(custom_params=CMM.get_parameter_dict_with_exch_pol_from_original_submission())

    csv_file = string(splitext(xyz_file)[1], ".csv")
    write_mbe_for_dataset_to_csv(xyz_file, csv_file, ff, ff_pol)
end

if length(ARGS) != 1
    @assert false, "Did not receive an input file. Usage: julia get_mbe.jl [xyz_file]"
else
    main(ARGS[1])
end