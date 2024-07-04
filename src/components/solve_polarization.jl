
"""
Solve system of equations by conjugate gradient. Specifically, we expect
this to be the polarization equation, but the code is generic. The result
is stored in the x vector.
It is possible to do this without explicitly constrcuting A, but for now
we just construct and do the matrix vector products directly since it is
simpler.
See: https://en.wikipedia.org/wiki/Conjugate_gradient_method
for explanation and notation used here.
"""
function conjugate_gradient!(A::Matrix{Float64}, x::Vector{Float64}, b::Vector{Float64}, tolerance::Float64=1e-13, maxiter::Int=10000)
    r_k = b - A * x # residual
    if norm(r_k) / length(r_k) < tolerance
        return
    end
    r_k_last = zero(r_k)
    p_k = copy(r_k)
    for k in 1:maxiter
        α_k = (r_k ⋅ r_k) / (p_k ⋅ (A * p_k))
        @views x[:] += α_k * p_k
        copy!(r_k_last, r_k)
        @views r_k[:] -= α_k * A * p_k
        if norm(r_k) / length(r_k) < tolerance
            return
        end
        β_k = (r_k ⋅ r_k) / (r_k_last ⋅ r_k_last)
        @views p_k[:] = r_k + β_k * p_k
    end
    @warn "Failed to converge polarization in max number iterations!"
    return
end