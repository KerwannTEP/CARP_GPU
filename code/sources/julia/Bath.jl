# using HypergeometricFunctions
# using SpecialFunctions
# using StaticArrays # To have access to static arrays
# using Interpolations # To have access to interpolation functions




# https://cuda.juliagpu.org/stable/tutorials/custom_structs/
import Adapt 

struct HG_Interpolate{T}

    tabxInt::T
    tabHG1Int::T
    tabHG2Int::T 

end
Adapt.@adapt_structure HG_Interpolate

function HG_Interpolate_init_GPU(nbxInt::Int64=1000)

    xminInt = 0.0
    xmaxInt = 1.0
    #####
    rangexInt = range(xminInt,length=nbxInt,xmaxInt)
    tabxInt = collect(rangexInt)
    tabHG1Int = zeros(Float64,nbxInt)
    tabHG2Int = zeros(Float64,nbxInt)

    #####
    for indx=2:nbxInt-1
        xloc = tabxInt[indx]
        hg1loc = _₂F₁(qCalc/2,qCalc-3.5,1.0,xloc)
        hg2loc = _₂F₁(qCalc/2,qCalc/2,4.5-qCalc/2,xloc)
        tabHG1Int[indx] = hg1loc
        tabHG2Int[indx] = hg2loc
    end
    # x=0
    tabHG1Int[1] = 1.0
    tabHG2Int[1] = 1.0

    # x=1
    tabHG1Int[nbxInt] = gamma(4.5-1.5*qCalc)/(gamma(1-qCalc/2)*gamma(4.5-qCalc))
    tabHG2Int[nbxInt] = gamma(4.5-qCalc/2)*gamma(4.5-1.5*qCalc)/(gamma(4.5-qCalc)*gamma(4.5-qCalc))

    return HG_Interpolate(CuArray(tabxInt), CuArray(tabHG1Int), CuArray(tabHG2Int))

end

function H_1(hg_int::HG_Interpolate, x::Float64)

    i = searchsortedfirst(hg_int.tabxInt, x)
    i = clamp(i, firstindex(hg_int.tabHG1Int), lastindex(hg_int.tabHG1Int))
    @inbounds hg_int.tabHG1Int[i]

end

function H_2(hg_int::HG_Interpolate, x::Float64)

    i = searchsortedfirst(hg_int.tabxInt, x)
    i = clamp(i, firstindex(hg_int.tabHG2Int), lastindex(hg_int.tabHG2Int))
    @inbounds hg_int.tabHG2Int[i]

end

const hg_int_default = HG_Interpolate_init_GPU()




##################################################
# Distribution function in (E,L) for a Plummer sphere
##################################################

# https://discourse.julialang.org/t/defining-function-inside-a-macro/9139/2
# https://groups.google.com/g/julia-users/c/OEXxlaeZFoU

macro make_H()

    fn = Symbol("_H")

    structn = HG_Interpolate
    typef = Float64

    if (qCalc == 1.0)

        quote
            function $(esc(fn))(hg_int::$structn, x::$typef)
                
                if (x <= 1)
                    pref = 1/(GAMMA_ca)
                    HG   = H_1(hg_int,x)
                    return pref*HG
                else
                    pref = 1/(GAMMA_db*GAMMA_bc)
                    HG   = H_2(hg_int,1/x)
                    return pref*1/sqrt(x)*HG
                end  
                
            end
        end


    elseif (qCalc == -6.0)

        quote
            function $(esc(fn))(hg_int::$structn, x::$typef)
                
                if (x <= 1)
                    pref = 1/(GAMMA_ca)
                    HG   = H_1(hg_int,x)
                    return pref*HG
                else
                    pref = 1/(GAMMA_db*GAMMA_bc)
                    HG   = H_2(hg_int,1/x)
                    return pref*x^3*HG
                end
                
            end
        end

    else

        quote
            function $(esc(fn))(hg_int::$structn, x::$typef)
                
                if (x <= 1)
                    pref = 1/(GAMMA_ca)
                    HG   = H_1(hg_int,x)
                    return pref*HG
                else
                    pref = 1/(GAMMA_db*GAMMA_bc)
                    HG   = H_2(hg_int,1/x)
                    return pref*x^(-qCalc/2)*HG
                end
                
            end
        end

    end


end

@make_H()


macro make_tF()

    fn = Symbol("_tF")

    structn = HG_Interpolate
    typef = Float64

    if (qCalc == 0.0)

        quote
            function $(esc(fn))(hg_int::$structn, tE::$typef, tL::$typef)

                if (tE < 0.0) # If E or L are negative, the DF vanishes
                    return 0.0
                end

                return 3.0/(7.0*PI^3) * (2.0*tE)^(3)*sqrt(2.0*tE)
                
            end
        end


    elseif (qCalc == 2.0)

        quote
            function $(esc(fn))(hg_int::$structn, tE::$typef, tL::$typef)
                
                if (tE < 0.0 || tL < 0.0) # If E or L are negative, the DF vanishes
                    return 0.0
                end

                # If E and L are positive
                x = tL^2/(2.0*tE)
                if (x <= 1)
                    return 6.0/(2.0*PI)^3 * (2.0*tE - tL^2)^(3/2)
                end
            
                return 0.0
                
            end
        end


    elseif (qCalc == 1.0)

        quote
            function $(esc(fn))(hg_int::$structn, tE::$typef, tL::$typef)
                
                if (tE < 0.0 || tL < 0.0) # If E or L are negative, the DF vanishes
                    return 0.0
                end

                # If E and L are positive
                x = tL^2/(2.0*tE)
                return (3.0*GAMMA_6q/(2.0*(2.0*PI)^(5/2)) *
                           tE*tE*sqrt(tE) * _H(hg_int,x))
                
            end
        end

    elseif (qCalc == -6.0)

        quote
            function $(esc(fn))(hg_int::$structn, tE::$typef, tL::$typef)
                
                if (tE < 0.0 || tL < 0.0) # If E or L are negative, the DF vanishes
                    return 0.0
                end
                # If E and L are positive
                x = tL^2/(2.0*tE)
                return (3.0*GAMMA_6q/(2.0*(2.0*PI)^(5/2)) *
                           tE^2*tE^2*tE^2*tE^2*tE*sqrt(tE) * _H(hg_int,x))
                
            end
        end

    else 

        quote
            function $(esc(fn))(hg_int::$structn, tE::$typef, tL::$typef)
                
                if (tE < 0.0 || tL < 0.0) # If E or L are negative, the DF vanishes
                    return 0.0
                end
                # If E and L are positive
                x = tL^2/(2.0*tE)
                return (3.0*GAMMA_6q/(2.0*(2.0*PI)^(5/2)) *
                           tE^(3.5-qCalc) * _H(hg_int,x))
                
            end
        end

    end


end

@make_tF()


function _F(hg_int::HG_Interpolate, E::Float64, L::Float64)
    tE = _tE(E)
    tL = _tL(L)
    DF  = _tF(hg_int,tE,tL)
    return _M*_F0*DF
end


##################################################
# Adding rotation to the DF
##################################################


function _Frot(hg_int::HG_Interpolate, E::Float64, L::Float64, Lz::Float64, alpha::Float64=alphaRot)
    Ftot = _F(hg_int,E,L)
    Frot = Ftot*(1.0 + alpha*sign(Lz/L))
    return Frot
end

# Normalized to M = int dJr dL dcosI _Frot_cosI(Jr,L,cosI)
function _Frot_cosI(hg_int::HG_Interpolate, E::Float64, L::Float64, cosI::Float64, alpha::Float64=alphaRot)
    Ftot = _F(hg_int,E,L)
    Frot = L*Ftot*(1.0 + alpha*sign(cosI))
    return Frot
end
