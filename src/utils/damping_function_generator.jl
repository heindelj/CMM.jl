using SimplePolynomials

function get_P3(P1::SimplePolynomial)
    d_P1 = derivative(P1)
    return P1 - (d_P1 - P1) * SimplePolynomial(0, 1)
end

function get_P5(P1::SimplePolynomial)
    d_P1  = derivative(P1)
    d2_P1 = derivative(d_P1)
    P3 = get_P3(P1)
    return P3 - 1 // 3 * (2 * d_P1 - d2_P1 - P1) * SimplePolynomial(0, 0, 1)
end

function guess_P5_rule(P1::SimplePolynomial)
    d1_P1  = derivative(P1)
    d2_P1 = derivative(d1_P1)
    P3 = get_P3(P1)
    # Assume the polynomial looks like the previous order
    # minus a coefficient times a polynomial in derivatives
    # of the lowest order one times the order n basis polynomial.
    c0_numerators = [1, 2, 3]
    c0_denominators = [1, 3, 15, 105] # pretty sure this will be 1, 3, 15, 105, ...
    c_P1s    = [-10:10...]
    c_d1_P1s  = [-10:10...]
    c_d2_P1s = [-10:10...]
    P5_known = SimplePolynomial(1, 1, 1//2, 1//6, 1//24, 1//144)
    for c0_numerator in c0_numerators
        for c0_denominator in c0_denominators
            for c_P1 in c_P1s
                for c_d1_P1 in c_d1_P1s
                    for c_d2_P1 in c_d2_P1s
                        P_guess = P3 - c0_numerator // c0_denominator * (
                            c_P1 * P1 + c_d1_P1 * d1_P1 + c_d2_P1 * d2_P1
                        ) * SimplePolynomial(0, 0, 1)
                        if (P_guess - P5_known) == 0
                            return true, [c0_numerator // c0_denominator, c_P1, c_d1_P1, c_d2_P1]
                        end
                    end
                end
            end
        end
    end
    return false, BigInt[]
end

function guess_P7_rule(P1::SimplePolynomial)
    d1_P1  = derivative(P1)
    d2_P1 = derivative(d1_P1)
    d3_P1 = derivative(d2_P1)
    P5 = get_P5(P1)
    # Assume the polynomial looks like the previous order
    # minus a coefficient times a polynomial in derivatives
    # of the lowest order one times the order n basis polynomial.
    c0_numerators = [1]
    c0_denominators = [3, 5, 15] # pretty sure this will be 1, 3, 15, 105, ...
    c_P1s    = [-5:5...]
    c_d1_P1s  = [-5:5...]
    c_d2_P1s = [-5:5...]
    c_d3_P1s = [-5:5...]
    P7_known = SimplePolynomial(1, 1, 1//2, 1//6, 1//24, 1//120, 1//720)
    for c0_numerator in c0_numerators
        for c0_denominator in c0_denominators
            for c_P1 in c_P1s
                for c_d1_P1 in c_d1_P1s
                    for c_d2_P1 in c_d2_P1s
                        for c_d3_P1 in c_d3_P1s
                            P_guess = P5 - c0_numerator // c0_denominator * (
                                c_P1 * P1 + c_d1_P1 * d1_P1 + c_d2_P1 * d2_P1 +
                                c_d3_P1 * d3_P1
                            ) * SimplePolynomial(0, 0, 0, 1)
                            if (P_guess - P7_known) == 0
                                return true, [c0_numerator // c0_denominator, c_P1, c_d1_P1, c_d2_P1, c_d3_P1]
                            end
                        end
                    end
                end
            end
        end
    end
    return false, BigInt[]
end