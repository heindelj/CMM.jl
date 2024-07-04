# This file contains hard-coded versions of some renormalized spherical
# harmonics defined in J. Chem. Theory Comput. 2018, 14, 739−758
# Also see here: https://en.citizendium.org/wiki/Spherical_harmonics
# Briefly, these are equal to:
# Ylm = sqrt(2) * ((l-m)!/(l+m)!)^(1/2) * Plm(cos(ϕ)) * cos(mθ)
# where Plm(x) are the associated Legendre polynomials
# See: https://en.wikipedia.org/wiki/Associated_Legendre_polynomials

# For the gradient terms, any term with a nonzero m value returns the
# function value, θ derivative, then the ϕ dervative (three numbers in total)
# Terms with m=0 just return the function value and then the derivative

@inline function Y10(ϕ::Float64)
    return cos(ϕ)
end

@inline function Y10_and_grad(ϕ::Float64)
    return cos(ϕ), -sin(ϕ)
end

@inline function Y11(θ::Float64, ϕ::Float64)
    return -sqrt(1 - cos(ϕ)^2) * cos(θ)
end

@inline function Y11_and_grad(θ::Float64, ϕ::Float64)
    if abs(ϕ) < 1e-6
        return -sqrt(1 - cos(ϕ)^2) * cos(θ), sqrt(1 - cos(ϕ)^2) * sin(θ), -cos(θ) * cos(ϕ)
    end
    return -sqrt(1 - cos(ϕ)^2) * cos(θ), sqrt(1 - cos(ϕ)^2) * sin(θ), -cos(θ) * cos(ϕ) * sin(ϕ) / sqrt(1 - cos(ϕ)^2)
end

@inline function Y20(ϕ::Float64)
    return (3.0 * cos(ϕ)^2 - 1.0) / sqrt(2.0)
end

@inline function Y20_and_grad(ϕ::Float64)
    return (3.0 * cos(ϕ)^2 - 1.0) / sqrt(2.0), -6.0 * cos(ϕ) * sin(ϕ) / sqrt(2.0)
end

@inline function Y21(θ::Float64, ϕ::Float64)
    return -sqrt(3) * cos(ϕ) * sqrt(1.0 - cos(ϕ)^2) * cos(θ)
end

@inline function Y21_and_grad(θ::Float64, ϕ::Float64)
    if abs(ϕ) < 1e-6
        return -sqrt(3) * cos(ϕ) * sqrt(1.0 - cos(ϕ)^2) * cos(θ),
        sqrt(3) * cos(ϕ) * sqrt(1.0 - cos(ϕ)^2) * sin(θ), 
        -sqrt(3) * cos(θ) * (1.0 - sqrt(1.0 - cos(ϕ)^2) * sin(ϕ))
    end
    return -sqrt(3) * cos(ϕ) * sqrt(1.0 - cos(ϕ)^2) * cos(θ),
    sqrt(3) * cos(ϕ) * sqrt(1.0 - cos(ϕ)^2) * sin(θ), 
    -sqrt(3) * cos(θ) * (cos(ϕ)^2 / sqrt(1.0 - cos(ϕ)^2) * sin(ϕ) - sqrt(1.0 - cos(ϕ)^2) * sin(ϕ))
end

@inline function Y22(θ::Float64, ϕ::Float64)
    return sqrt(0.75) * (1.0 - cos(ϕ)^2) * cos(2.0 * θ)
end

@inline function Y22_and_grad(θ::Float64, ϕ::Float64)
    return sqrt(0.75) * (1.0 - cos(ϕ)^2) * cos(2.0 * θ),
    -2.0 * sqrt(0.75) * (1.0 - cos(ϕ)^2) * sin(2.0 * θ),
    2.0 * sqrt(0.75) * cos(2.0 * θ) * cos(ϕ) * sin(ϕ)
end

@inline function Y30(ϕ::Float64)
    return (5.0 * cos(ϕ)^3 - 3.0 * cos(ϕ)) / sqrt(2.0)
end