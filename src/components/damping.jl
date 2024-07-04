
abstract type AbstractDamping end

struct TholeDamping <: AbstractDamping end
struct SlaterShellDamping <: AbstractDamping end
struct SlaterOverlapDamping <: AbstractDamping end
struct SlaterPolarizationDamping <: AbstractDamping end
struct SlaterAnionDamping <: AbstractDamping end
struct SlaterRepulsionDamping <: AbstractDamping end
struct TangToenniesDamping <: AbstractDamping end

include("thole_damping.jl")
include("slater_damping.jl")
include("tang_toennies_damping.jl")

function is_ion(label::String)
    # This function is used to turn off mutual polarization
    # for ions since they seem to cause divergences
    if label == "O" || label == "H"
        return false
    elseif label == "F"
        return true
    elseif label == "Cl"
        return true
    elseif label == "Br"
        return true
    elseif label == "I"
        return true
    elseif label == "Li"
        return true
    elseif label == "Na"
        return true
    elseif label == "K"
        return true
    elseif label == "Rb"
        return true
    elseif label == "Cs"
        return true
    elseif label == "Mg"
        return true
    elseif label == "Ca"
        return true
    end
    return false
end