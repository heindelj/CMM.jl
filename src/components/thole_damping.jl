# This file contains functions for returning the value of thole
# smearing functions, which are used to modify the various multipolar
# interactions. All terms are related by a recursion relationship
# See: J. Chem. Phys. 133, 234101 (2010)
# Also: J. Phys. Chem. B, Vol. 107, No. 24, 2003

@inline function get_λ1(u::Float64, a::Float64, disambiguate::TholeDamping)
    return 1.0 - exp(-a * u^3) + a^(1/3) * u * gamma(2/3, a * u^3)
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::TholeDamping)
    return 1 - exp(-a * u^3)
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::TholeDamping)
    return 1 - (1 + a * u^3) * exp(-a * u^3)
end

@inline function get_λ7(u::Float64, a::Float64, disambiguate::TholeDamping)
    return 1 - (1 + a * u^3 + 3/5 * a^2 * u^6) * exp(-a * u^3)
end

@inline function get_λ9(u::Float64, a::Float64, disambiguate::TholeDamping)
    return 1 - (1 + a * u^3 + (18 * a^2 * u^6 + 9 * a^3 * u^9) / 35) * exp(-a * u^3)
end

function get_mean_polarizability(label::String, params::Dict{Symbol,Float64})
    if haskey(params, Symbol(label, :_αxx))
        return (params[Symbol(label, :_αxx)] + params[Symbol(label, :_αyy)] + params[Symbol(label, :_αzz)]) / 3.0
    end
    return abs(params[Symbol(label, :_α)])
end

@inline function get_u(r_ij::Float64, label_i::String, label_j::String, params::Dict{Symbol, Float64}, disambiguate::TholeDamping)
    return u = r_ij / (get_mean_polarizability(label_i, params) * get_mean_polarizability(label_j, params))^(1/6)
end

@inline function get_a(params::Dict{Symbol, Float64}, disambiguate::TholeDamping)
    return params[:a_thole]
end