
#
# Note: The incomplete gamma function below and its derivatives are only
# correct for n >= 1 as implemented. There are modifications that need to
# be made for the case when n=0. This case never arises in these models,
# so we jus avoid the branch for now. If you request n=0 for any of the
# derivatives, it will fail with a factorial domain error right now.
#
function inc_gamma(x::Float64, n::Int)
    sum = 0.0
    @inbounds for k in 0:n
        sum += x^k / factorial(k)
    end
    return 1.0 - exp(-x) * sum
end

@inline function inc_gamma_derivative(x::Float64, n::Int)
    x^n / factorial(n) * exp(-x)
end

@inline function inc_gamma_second_derivative(x::Float64, n::Int)
    inc_gamma_derivative(x, n-1) - inc_gamma_derivative(x, n)
end

"""
For details about this and the dispersion function below see:
J. Chem. Theory Comput. 2016, 12, 3851−3870
"""
@inline function slater_damping_value(r_ij::Float64, b_ij::Float64)
    return (b_ij - (2 * b_ij^2 * r_ij + 3 * b_ij) / (b_ij^2 * r_ij^2 + 3 * b_ij * r_ij + 3)) * r_ij
end

@inline function slater_damping_value_gradient(r_ij_vec::MVector{3, Float64}, r_ij::Float64, b_ij::Float64)
    return (
        b_ij - (
                (b_ij^2 * r_ij^2 + 3 * b_ij * r_ij + 3) * (4 * b_ij^2 * r_ij + 3 * b_ij) -
                (2 * b_ij^2 * r_ij^2 + 3 * b_ij * r_ij) * (2 * b_ij^2 * r_ij + 3 * b_ij)
            ) / (b_ij^2 * r_ij^2 + 3 * b_ij * r_ij+ 3)^2
        ) * -r_ij_vec / r_ij
end

@inline function born_mayer_damping_value(r_ij::Float64, b_ij::Float64, scaling::Float64=0.84)
    return scaling * b_ij * r_ij
end

@inline function born_mayer_damping_value_gradient(r_ij_vec::MVector{3, Float64}, r_ij::Float64, b_ij::Float64, scaling::Float64=0.84)
    return -scaling * b_ij * r_ij_vec / r_ij
end

@inline function born_mayer_damping_value_hessian(r_ij_vec::MVector{3, Float64}, r_ij::Float64, b_ij::Float64, scaling::Float64=0.84)
    return -scaling * b_ij * (diagm(ones(3)) / r_ij - r_ij_vec * r_ij_vec' / r_ij^3)
end
