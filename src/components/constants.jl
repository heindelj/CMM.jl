using Parameters

####### convergence constants #######
@inline function chargecon()
    return 18.22261720426243437986
end

@inline function debye()
    return 4.8033324 # eÅ
end

abstract type TTM_Constants end

@with_kw struct TTM21_Constants <: TTM_Constants
    # vdw
    vdwA::Float64 = -1.329565985e+6
    vdwB::Float64 =  3.632560798e+5
    vdwC::Float64 = -2.147141323e+3
    vdwD::Float64 =  1.0e+13
    vdwE::Float64 =  13.2

    # M-site positioning
    γ_M::Float64 = 0.426706882
    γ_1::Float64 = 1.0 - γ_M
    γ_2::Float64 = γ_M / 2

    # polarizability
    α_O::Float64 = 0.837
    α_H::Float64 = 0.496
    α_M::Float64 = 0.0

    # Thole damping factors
    damping_factor_O::Float64 = α_O
    damping_factor_H::Float64 = α_H
    damping_factor_M::Float64 = α_O

    name::Symbol = :ttm21
end

@with_kw struct TTM22_Constants <: TTM_Constants
    # vdw
    vdwA::Float64 = -1.329565985e+6
    vdwB::Float64 =  3.632560798e+5
    vdwC::Float64 = -2.147141323e+3
    vdwD::Float64 =  1.0e+13
    vdwE::Float64 =  13.2
    
    # dms (start with the ttm3 modified dms)
    dms_param1::Float64 = 0.5
    dms_param2::Float64 = 0.9578
    dms_param3::Float64 = 0.012

    # M-site positioning
    γ_M::Float64 = 0.426706882
    γ_1::Float64 = 1.0 - γ_M
    γ_2::Float64 = γ_M / 2

    # polarizability
    α_O::Float64 = 0.837
    α_H::Float64 = 0.496
    α_M::Float64 = 0.0

    # Thole damping factors
    damping_factor_O::Float64 = α_O
    damping_factor_H::Float64 = α_H
    damping_factor_M::Float64 = α_O

    name::Symbol = :ttm22
end

@with_kw struct TTM3_Constants <: TTM_Constants
    # vdw
    vdwC::Float64 = -0.72298855E+03
    vdwD::Float64 =  0.10211829E+06
    vdwE::Float64 =  0.37170376E+01

    # dms
    dms_param1::Float64 = 0.5
    dms_param2::Float64 = 0.9578
    dms_param3::Float64 = 0.012

    # M-site positioning
    γ_M::Float64 = 0.46
    γ_1::Float64 = 1.0 - γ_M
    γ_2::Float64 = γ_M / 2

    # polarizability
    α_O::Float64 = 0.0
    α_H::Float64 = 0.0
    α_M::Float64 = 1.444

    # Thole damping factors
    damping_factor_O::Float64 = 0.837
    damping_factor_H::Float64 = 0.496
    damping_factor_M::Float64 = 0.837

    name::Symbol = :ttm3
end

@with_kw struct TTM4_Constants <: TTM_Constants
    # vdw
    A6::Float64 = -0.126503e+5
    A8::Float64 =  0.526347e+6
    A10::Float64 = -0.964270e+7
    A12::Float64 =  0.877792e+8
    A14::Float64 = -0.367476e+9
    A16::Float64 =  0.572395e+9

    # M-site positioning
    γ_M = 0.426706882
    γ_1 = 1.0 - gammaM
    γ_2 = gammaM / 2

    # polarizability
    α_O::Float64 = 1.310
    α_H::Float64 = 0.294
    α_M::Float64 = 0.0

    # Thole damping factors
    damping_factor_O::Float64 = α_O
    damping_factor_H::Float64 = α_H
    damping_factor_M::Float64 = α_O
    
    name::Symbol = :ttm4
end

# probably factor all of the qtip4pf stuff into it's own directory
@with_kw struct qtip4pf_Constants
    qM::Float64 = -1.1128 * chargecon()
    qH::Float64 = -qM/2

    rOHeq::Float64= 0.9419 # A
    aHOHeq::Float64 = 107.4 # degree

    epsilon::Float64 = 0.1852 # kcal/mol
    sigma::Float64 = 3.1589 # A

    alpha_r::Float64 = 2.287 # A-1
    D_r::Float64 = 116.09 # kcal/mol
    k_theta::Float64 = 87.85 # kcal/mol/rad^2

    gammaM::Float64 = 0.73612
    gamma1::Float64 = (1.0 - gammaM)/2
end

function get_constants_type(potential::Symbol)
    if potential == :ttm3
        return TTM3_Constants()
    elseif potential == :ttm21
        return TTM21_Constants()
    elseif potential == :ttm4
        return TTM4_Constants()
    else
        @assert false "Didn't receive a valid potential. Pass as a symbol :ttm3, :ttm21, or :ttm4."
    end
end
