
function double_factorial(n::Float64)
    if n == 0.0
        return 1.0
    end
    if n > 0.0
        res = 1.0
        while n > 0.0
            res *= n
            n -= 2
        end
        return res
    end
    if n < 0.0 && (n % 2) == -1.0
        res = 1.0
        while n < 0.0
            res /= (n + 2)
            n += 2
        end
        return res
    else
        return Inf
    end
end

"""
Compute the double factorial for an integer.
Only accepts positive integers and n=-1 which equals 1.
"""
function double_factorial(n::Int)
    if n == 0
        return 1
    end
    if n > 0
        res = 1
        while n > 0
            res *= n
            n -= 2
        end
        return res
    end
    if n == -1
        return 1
    end
    @assert false "We don't return anything for double factorials of input less than -1. Try with a float for full implementation."
end

"""
Computes an element of the cartesian interaction tensor, Tαβγδ
by recursion relations described in:
M. Challacombe et al. Chemical Physics Letters 241 (1995) 67-72
See Eqs. 21-24.
r_ji is r_i - r_j.

Note: α, β, and γ are the number of indices into that cartesian index.
So, for the interaction the x-component of a dipole with the xy-component
of a quadrupole, you would call get_T_αβγ(1, 2, 0).
Notice, that there is a default δ parameter as a fourth argument. This is
used in the recursion to construct the interaction tensor elements but is
not needed when retriving a particular element.
"""
function get_T_αβγ(r_ji::MVector{3, Float64}, α::Int, β::Int, γ::Int, δ::Int=0)
    if α < 0 || β < 0 || γ < 0 || δ < 0
        return 0.0
    end
    if α == 0 && β == 0 && γ == 0
        return get_T_000δ(r_ji, δ)
    elseif β == 0 && γ == 0
        return get_T_α00δ(r_ji, α, δ)
    elseif γ == 0
        return get_T_αβ0δ(r_ji, α, β, δ)
    end

    return r_ji[3] * get_T_αβγ(r_ji, α, β, γ-1, δ+1) + (γ-1) * get_T_αβγ(r_ji, α, β, γ-2, δ+1)
end

@inline function get_T_000δ(r_ji::MVector{3, Float64}, δ::Int)
    return (-1)^δ * double_factorial(2 * δ - 1) * norm(r_ji)^-(2*δ+1)
end

@inline function get_T_α00δ(r_ji::MVector{3, Float64}, α::Int, δ::Int)
    return r_ji[1] * get_T_αβγ(r_ji, α-1, 0, 0, δ+1) + (α-1) * get_T_αβγ(r_ji, α-2, 0, 0, δ+1)
end

@inline function get_T_αβ0δ(r_ji::MVector{3,Float64}, α::Int, β::Int, δ::Int)
    return r_ji[2] * get_T_αβγ(r_ji, α, β-1, 0, δ+1) + (β-1) * get_T_αβγ(r_ji, α, β-2, 0, δ+1)
end