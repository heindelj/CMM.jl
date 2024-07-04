
function conversion(unit1::Symbol, unit2::Symbol)
    units = Dict(
        # energy conversions
        :hartree      => 1.0,
        :kcal         => 627.509608031,
        :wavenumbers  => 219474.63,
        :ev           => 27.2115985349,
        # distances
        :bohr         => 1.0,
        :angstrom     => 0.52917721092,
        :pm           => 100 * 0.52917721092,
        # masses
        :amu          => 1.0,
        :au_mass      => 1822.88839,
        # temperature
        :au_temperature => 1.0,
        :kelvin         => 315774.659,
        # time
        :fs           => 0.02418884254,
        :au_time      => 1.0
    )
    return units[unit2] / units[unit1]
end
