
let 
    global gammln, gammq

    FPMIN::Float64 = 2.22507e-308 / eps()
    ngau::Int = 18

    y::Vector{Float64} = [0.0021695375159141994,
    0.011413521097787704,0.027972308950302116,0.051727015600492421,
    0.082502225484340941, 0.12007019910960293,0.16415283300752470,
    0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
    0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
    0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
    0.87126389619061517, 0.95698180152629142]

    w::Vector{Float64} = [0.0055657196642445571,
    0.012915947284065419,0.020181515297735382,0.027298621498568734,
    0.034213810770299537,0.040875750923643261,0.047235083490265582,
    0.053244713977759692,0.058860144245324798,0.064039797355015485,
    0.068745323835736408,0.072941885005653087,0.076598410645870640,
    0.079687828912071670,0.082187266704339706,0.084078218979661945,
    0.085346685739338721,0.085983275670394821]

    cof::Vector{Float64} = [57.1562356658629235,-59.5979603554754912,
        14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
        .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
        -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
        .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5]

    function gammpapprox(a::Float64, x::Float64, psig::Int)
        gln::Float64    = gammln(a)
        a1::Float64     = a - 1.0
        lna1::Float64   = log(a1)
        sqrta1::Float64 = sqrt(a1)

        xu::Float64 = 0.0
        t::Float64 = 0.0
        sum__::Float64 = 0.0
        ans::Float64 = 0.0

        if (x > a1)
            xu = max(a1 + 11.5 * sqrta1, x + 6.0 * sqrta1)
        else
            xu = max(0.0, min(a1 - 7.5 * sqrta1, x - 5.0 * sqrta1))
        end

        for j in 1:ngau
            t = x + (xu - x) * y[j]
            sum__ += w[j] * exp(-(t - a1) + a1*(log(t) - lna1))
        end

        ans = sum*(xu - x)*exp(a1*(lna1 - 1.0) - gln)

        return (psig ? (ans > 0.0 ? 1.0 - ans : -ans)
                    : (ans >= 0.0 ? ans : 1.0 + ans))
    end

    function gser(a::Float64, x::Float64)
        gln::Float64 = gammln(a)

        ap::Float64  = a
        sum__::Float64 = 1.0 / a
        del::Float64 = sum__

        while true
            ap += 1
            del *= x/ap
            sum__ += del
            if (abs(del) < abs(sum__) * eps())
                return sum__ * exp(-x + a * log(x) - gln)
            end
        end
    end

    function gcf(a::Float64, x::Float64)
        gln::Float64 = gammln(a)

        b::Float64 = x + 1.0 - a
        c::Float64 = 1.0/FPMIN
        d::Float64 = 1.0/b
        h::Float64 = d

        i::Int = 1
        while i < typemax(Int)
            an::Float64 = -i * (i - a)

            b += 2.0
            d = an * d + b
            if (abs(d) < FPMIN)
                d = FPMIN
            end

            c = b + an/c

            if (abs(c) < FPMIN)
                c = FPMIN
            end

            d = 1.0/d
            del::Float64 = d*c
            h *= del

            if (abs(del - 1.0) <= eps())
                break
            end
            i += 1
        end

        return h * exp( -x + a * log(x) - gln)
    end

    function gammq(a::Float64, x::Float64)
        """
        Implements the q-gamma function which seemingly isn't available
        in SpecialFunctions.jl.
        """
        ASWITCH::Int = 100

        @assert (x >= 0.0 && a > 0.0) "Negative numbers are not valid for gamma function!"

        if (x == 0.0)
            return 1.0
        elseif (Int(round(a)) >= ASWITCH)
            return gammpapprox(a, x, 0)
        elseif (x < a + 1.0)
            return 1.0 - gser(a,x)
        else
            return gcf(a,x)
        end
    end

    function gammln(xx::Float64)
        """
        Log gamma function
        """
        @assert xx > 0.0 "Input must be greater than 0.0."

        x::Float64 = xx
        z::Float64 = xx

        tmp::Float64 = x + 5.24218750000000000
        tmp = (x + 0.5) * log(tmp) - tmp

        ser = 0.999999999999997092
        for j in 1:length(cof)
            z += 1
            ser += cof[j] / z
        end

        return tmp + log(2.5066282746310005 * ser / x)
    end
    
end