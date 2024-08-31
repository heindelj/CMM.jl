using CMM, Profile

function profile_ion_water_energy(n::Int)
    header, labels, geoms = CMM.read_xyz("benchmark/assets/w4_na_cl.xyz")
    fragment_inidces = [[[i,i+1,i+2] for i in 1:3:length(labels[1])-2]..., [length(labels[1])-1], [length(labels[1])]]

    ff = build_cmm_model()
    for i in 1:n
        evaluate!(geoms[1] / .529177, labels[1], fragment_inidces, ff)
    end
end

@profview profile_ion_water_energy(1)
@profview profile_ion_water_energy(10)