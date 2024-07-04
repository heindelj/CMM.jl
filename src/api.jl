import CMM.CMM_FF

function evaluate!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    ff::CMM_FF
)
    energy = 0.0
    update_storage!(coords, labels, fragment_indices, ff)
    for i in eachindex(ff.terms)
        energy += ff.terms[i](coords, labels, fragment_indices, ff)
    end
    finalize!(ff)
    return energy
end

function mbe(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::AbstractForceField,
    term::Symbol=:Total,
    order::Int=5
)
    num_fragments = count(>(0), length.(fragment_indices))
    if order > num_fragments
        order = num_fragments
    end

    energies = zeros(order)

    ff_no_ct = typeof(ff)(copy(ff.terms), copy(ff.params), ff.storage, typeof(ff.results)(copy(ff.results.energies), nothing))
    if term == :Polarization
        for i in length(ff_no_ct.terms):-1:1
            if ff_no_ct.terms[i] == short_range_energy! || ff_no_ct.terms[i] == short_range_energy_and_gradients!
                popat!(ff_no_ct.terms, i)
            end
        end
    end

    if term == :ChargeTransfer
        for i in length(ff_no_ct.terms):-1:1
            if ff_no_ct.terms[i] == short_range_energy! || ff_no_ct.terms[i] == short_range_energy_and_gradients!
                popat!(ff_no_ct.terms, i)
            end
        end
    end

    for i_mbe in 1:order
        for fragment_indices_combination in combinations(fragment_indices, i_mbe)
            flat_indices = reduce(vcat, fragment_indices_combination)
            @views subsystem_coords = coords[flat_indices]
            @views subsystem_labels = labels[flat_indices]
            shifted_indices = [zero(fragment_indices_combination[i]) for i in eachindex(fragment_indices_combination)]
            index = 1
            for i_frag in eachindex(shifted_indices)
                for i in eachindex(shifted_indices[i_frag])
                    shifted_indices[i_frag][i] = index
                    index += 1
                end
            end
            if term == :ChargeTransfer
                evaluate!(subsystem_coords, subsystem_labels, shifted_indices, ff)
                evaluate!(subsystem_coords, subsystem_labels, shifted_indices, ff_no_ct)
                energies[i_mbe] += ff.results.energies[:CT_direct] + (ff.results.energies[:Polarization] - ff_no_ct.results.energies[:Polarization])
            elseif term == :Polarization
                evaluate!(subsystem_coords, subsystem_labels, shifted_indices, ff_no_ct)
                energies[i_mbe] += ff_no_ct.results.energies[:Polarization]
            else
                evaluate!(subsystem_coords, subsystem_labels, shifted_indices, ff)
                energies[i_mbe] += ff.results.energies[term]
            end
        end
    end

    return get_mbe_data_from_subsystem_sums(energies, num_fragments)
end

function finite_difference_forces(
    coords::Vector{MVector{3, Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::AbstractForceField,
    term::Symbol,
    step_size = 1e-5
)
    grads = [@MVector zeros(3) for _ in eachindex(coords)]

    for i in eachindex(coords)
        for w in 1:3
            coords[i][w] += step_size
            evaluate!(coords, labels, fragment_indices, ff)
            f_plus_h = ff.results.energies[term]

            coords[i][w] -= 2 * step_size
            evaluate!(coords, labels, fragment_indices, ff)
            f_minus_h = ff.results.energies[term]
            coords[i][w] += step_size
            grads[i][w] = (f_plus_h - f_minus_h) / (2 * step_size)
        end
    end
    return grads
end

function optimize_xyz(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    ff::AbstractForceField,
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
    ff::AbstractForceField,
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


"""
Reads in an xyz file of possibly multiple geometries, returning the header, atom labels, 
and coordinates as arrays of strings and Vector{MVector{3, Float64}} for the coordinates.
The coordinates can be optionally converted to SVectors instead.
"""
function read_xyz(ifile::String; to_svector=false)
    header = Vector{String}()
    atom_labels = Vector{Vector{String}}()
    
    geoms = Vector{Vector{MVector{3, Float64}}}()
    successfully_parsed = true
    open(ifile, "r") do io
        for line in eachline(io)
            if isa(tryparse(Int, line), Int)
                # allocate the geometry for this frame
                N = parse(Int, line)
                # store the header for this frame
                head = string(line, "\n", readline(io))
                # loop through the geometry storing the vectors and atom labels as you go
                new_data = fill(Vector{SubString{String}}(), N)
                geom = [@MVector zeros(3) for _ in 1:N]
                labels = fill("", N)
                for j = 1:N
                    new_data[j] = split(readline(io))
                end
                for j = 1:N
                    try
                        labels[j] = new_data[j][1]
                        geom[j][1] = parse(Float64, new_data[j][2])
                        geom[j][2] = parse(Float64, new_data[j][3])
                        geom[j][3] = parse(Float64, new_data[j][4])
                    catch
                        @assert false "Failed to parse xyz file. Exiting."
                    end
                end
                # Store parsed structure
                push!(header, head)
                push!(geoms, geom)
                push!(atom_labels, labels)
            end
        end
    end
    if to_svector
        new_geoms = Vector{Vector{SVector{3, Float64}}}()
        for i in eachindex(geoms)
            push!(new_geoms, [SVector{3, Float64}(geoms[i][j]) for j in eachindex(geoms[i])])
        end
        return header, atom_labels, new_geoms
    end
    return header, atom_labels, geoms
end

function write_xyz(outfile::AbstractString, header::AbstractArray, labels::AbstractArray, geoms::AbstractArray; append::Bool=false, skip_atom_labels::Vector{String}=String[], directory::AbstractString="")
    """
    Writes an xyz file with the elements returned by read_xyz.
    """
    if isdir(directory)
        outfile = string(directory, "/", outfile)
    elseif directory == ""
        
    else
        mkdir(directory)
        outfile = string(directory, "/", outfile)
    end
    mode = "w"
    if append
        mode = "a"
    end
    if length(header) != length(geoms)
        header = [header[1] for i in 1:length(geoms)]
    end
    if length(labels) != length(geoms)
        labels = [labels[1] for i in 1:length(geoms)]
    end
    open(outfile, mode) do io
        for (i_geom, head) in enumerate(header)
            write(io, string(head, "\n"))
            for (i_coord, atom_label) in enumerate(labels[i_geom])
                if atom_label ∉ skip_atom_labels
                    write(io, string(atom_label, " ", join(string.(geoms[i_geom][:,i_coord]), " "), "\n"))
                end
            end
        end
    end
end

function write_xyz(outfile::AbstractString, header::AbstractString, labels::AbstractVector{String}, geoms::AbstractVector{Matrix{Float64}}; append::Bool=false, skip_atom_labels::Vector{String}=String[], directory::AbstractString="")
    write_xyz(outfile, [header for _ in eachindex(geoms)], [labels for _ in eachindex(geoms)], geoms, append=append, skip_atom_labels=skip_atom_labels, directory=directory)
end

function write_xyz(outfile::AbstractString, labels::AbstractVector{String}, geom::Matrix{Float64}; append::Bool=false, skip_atom_labels::Vector{String}=String[], directory::AbstractString="")
    num_atoms_to_skip = 0
    for i in eachindex(labels)
        if labels[i] in skip_atom_labels
            num_atoms_to_skip += 1
        end
    end
    write_xyz(outfile, [string(length(labels) - num_atoms_to_skip, "\n")], [labels], [geom], append=append, skip_atom_labels=skip_atom_labels, directory=directory)
end

function write_xyz(outfile::AbstractString, labels::AbstractVector{Vector{String}}, geoms::AbstractVector{Matrix{Float64}}; append::Bool=false, skip_atom_labels::Vector{String}=String[], directory::AbstractString="")
    num_atoms_to_skip = 0
    for i in eachindex(labels[1])
        if labels[1][i] in skip_atom_labels
            num_atoms_to_skip += 1
        end
    end
    write_xyz(outfile, [string(length(labels[i]) - num_atoms_to_skip, "\n") for i in eachindex(labels)], labels, geoms, append=append, skip_atom_labels=skip_atom_labels, directory=directory)
end