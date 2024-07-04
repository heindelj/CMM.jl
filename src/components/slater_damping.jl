function get_u(r_ij::Float64, b_i::Float64, params::Dict{Symbol, Float64}, disambiguate::SlaterShellDamping)
    return b_i * r_ij
end

@inline function get_λ1(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + 0.5 * u) * exp(-u)
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + u + 0.5 * u^2) * exp(-u)
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0) * exp(-u)
end

@inline function get_λ7(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + u^4 / 30.0) * exp(-u)
end

@inline function get_λ9(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + (4.0 / 105.0) * u^4 + u^5 / 210.0) * exp(-u)
end

# Coefficients come from: https://github.com/TinkerTools/tinker/blob/8a8098d10864348ecfcd062561e4ec4405bac3b1/source/damping.f
# Specifically valence-valence damping for Gordon f2.
@inline function get_λ11(u::Float64, a::Float64, disambiguate::SlaterShellDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + (5.0 / 126.0) * u^4 + (2.0 / 315.0) * u^5 + (1.0 / 1890.0) * u^6) * exp(-u)
end

@inline function get_λ1(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + 11/16 * u + 3/16 * u^2 + 1/48 * u^3) * exp(-u)
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + u + 0.5 * u^2 + 7/48 * u^3 + 1/48 * u^4) * exp(-u)
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + u + 0.5 * u^2 + 1/6 * u^3 + 1/24 * u^4 + 1/144 * u^5) * exp(-u)
end

@inline function get_λ7(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6) * exp(-u)
end

@inline function get_λ9(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6 + 1/5040 * u^7) * exp(-u)
end

@inline function get_λ11(u::Float64, a::Float64, disambiguate::SlaterOverlapDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6 + 1/5040 * u^7 + 1/45360 * u^8) * exp(-u)
end

@inline function get_λ1(u::Float64, a::Float64, disambiguate::SlaterPolarizationDamping)
    return 1 - sum([1, 1/9, 1/11, 1/13, 1/15] .* [1, u, u^2, u^3, u^4]) * exp(-u)
    # ^^^ This is what I have been using for the exch_pol model
    
    #return 1 - sum([1, 1/11, 1/13, 1/15, 1/17] .* [1, u, u^2, u^3, u^4]) * exp(-u)
    # ^^^ This is what I have been using for the induction model
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::SlaterPolarizationDamping)
    return 1 - sum([1, 1, 2/99, -9/143, -8/65, 1/15] .* [1, u, u^2, u^3, u^4, u^5]) * exp(-u)
    # ^^^ This is what I have been using for the exch_pol model
    
    #return 1 - sum([1, 1, 2/143, -11/195, -28/255, 1/17] .* [1, u, u^2, u^3, u^4, u^5]) * exp(-u)
    # ^^^ This is what I have been using for the induction model
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::SlaterPolarizationDamping)
    return 1 - sum([1, 1, 101/297, 2/297, 43/2145, -10/117, 1/45] .* [1, u, u^2, u^3, u^4, u^5, u^6]) * exp(-u)
    # ^^^ This is what I have been using for the exch_pol model
    
    #return 1 - sum([1, 1, 145/429, 2/429, 59/3315, -58/765, 1/51] .* [1, u, u^2, u^3, u^4, u^5, u^6]) * exp(-u)
    # ^^^ This is what I have been using for the induction model
end

# NOTE(JOE): This damping function is currently wrong! It is not actually used
# for any energies but is needed for polarization gradients. I put it here right
# now cause I know I'll need it in the future. This is only called in the
# collection of induced potentials, fields, etc. I have commented out the relevant
# calls to ensure no mistakes are made on accident.
@inline function get_λ7(u::Float64, a::Float64, disambiguate::SlaterPolarizationDamping)
    return 1 - sum([1, 1, 101/297, 2/297, 43/2145, -10/117, 1/45, 1/90] .* [1, u, u^2, u^3, u^4, u^5, u^6, u^7]) * exp(-u)
    #return 1 - (1 + u + 25/48 * u^2 + 3/16 * u^3 + 5/128 * u^4 - 1/384 * u^5 + 1/384 * u^6) * exp(-u)
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::SlaterAnionDamping)
    return 1 - sum([1, 1, 1/2, 0, 0, 0] .* [1, u, u^2, u^3, u^4, u^5]) * exp(-u)
    #return 1 - (1 + u + 9/16 * u^2 + 1/8 * u^3 + 1/128 * u^4 + 1/128 * u^5) * exp(-u)
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::SlaterAnionDamping)
    return 1 - sum([1, 1, 5/12, 0, 0, 0, 0] .* [1, u, u^2, u^3, u^4, u^5, u^6]) * exp(-u)
    #return 1 - (1 + u + 25/48 * u^2 + 3/16 * u^3 + 5/128 * u^4 - 1/384 * u^5 + 1/384 * u^6) * exp(-u)
end

@inline function get_λ1(u::Float64, a::Float64, disambiguate::TangToenniesDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6) * exp(-u)
end

@inline function get_λ3(u::Float64, a::Float64, disambiguate::TangToenniesDamping)
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6 + 1/720 * u^7) * exp(-u)
end

@inline function get_λ5(u::Float64, a::Float64, disambiguate::TangToenniesDamping)
    # I am at least mildly concerned that there is a minues sign on u^7.
    return 1 - (1 + u + 0.5 * u^2 + u^3 / 6.0 + 1/24 * u^4 + 1/120 * u^5 + 1/720 * u^6 - 1/720 * u^7 + 1/2160 * u^8) * exp(-u)
end

function get_u(r_ij::Float64, label_i::String, label_j::String, params::Dict{Symbol, Float64}, disambiguate::SlaterOverlapDamping)
    b_ij_elec, has_pairwise = maybe_get_pairwise_parameter(label_i, label_j, :_b_elec, params)
    b_ij_elec = abs(b_ij_elec)
    if !has_pairwise
        b_ij_elec = sqrt(abs(params[Symbol(label_i, :_b_elec)]) * abs(params[Symbol(label_j, :_b_elec)]))
    end
    return b_ij_elec * r_ij
end

function get_a(params::Dict{Symbol, Float64}, disambiguate::SlaterOverlapDamping)
    return 1.0
end

function get_a(params::Dict{Symbol, Float64}, disambiguate::SlaterShellDamping)
    return 1.0
end