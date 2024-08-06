# Computes the 2D flux in (Jr,L) space
function flux2D_JrL(Jr::Float64, L::Float64, cosI::Float64, m_field::Float64, 
            alpha::Float64=alphaRot, nbAvr::Int64=nbAvr_default,
            nbw::Int64=nbw_default,
            nbvartheta::Int64=nbvartheta_default, nbphi::Int64=nbphi_default,
            nbu::Int64=nbu0, eps::Float64=10^(-5), m_test::Float64=m_field, hg_int::HG_Interpolate=hg_int_default)

    # F_i = D_i Frot - 0.5 * d/dJk [D_ik Frot]

    Jr_p = Jr + eps*_L0
    Jr_m = Jr - eps*_L0
    L_p = L + eps*_L0
    L_m = L - eps*_L0



    E = E_from_Jr_L(Jr,L,nbu)
    E_Jrp = E_from_Jr_L(Jr_p,L,nbu)
    E_Jrm = E_from_Jr_L(Jr_m,L,nbu)
    E_Lp = E_from_Jr_L(Jr,L_p,nbu)
    E_Lm = E_from_Jr_L(Jr,L_m,nbu)

    dJr, dL, dLz, dJrJr, dLL, dLzLz, dJrL, dJrLz, dLLz = orbitAverageActionCoeffs(Jr,L,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,m_test,hg_int)


    Frot = _Frot_cosI(hg_int,E,L,cosI,alpha)


    # Partial derivatives

    dJr_Jrp, dL_Jrp, dLz_Jrp, dJrJr_Jrp, dLL_Jrp, dLzLz_Jrp, dJrL_Jrp, dJrLz_Jrp, dLLz_Jrp = orbitAverageActionCoeffs(Jr_p,L,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,m_test,hg_int)
    Frot_Jrp = _Frot_cosI(hg_int,E_Jrp,L,cosI,alpha)
    dJr_Jrm, dL_Jrm, dLz_Jrm, dJrJr_Jrm, dLL_Jrm, dLzLz_Jrm, dJrL_Jrm, dJrLz_Jrm, dLLz_Jrm = orbitAverageActionCoeffs(Jr_m,L,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,m_test,hg_int)
    Frot_Jrm = _Frot_cosI(hg_int,E_Jrm,L,cosI,alpha)

    dJr_Lp, dL_Lp, dLz_Lp, dJrJr_Lp, dLL_Lp, dLzLz_Lp, dJrL_Lp, dJrLz_Lp, dLLz_Lp = orbitAverageActionCoeffs(Jr,L_p,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,m_test,hg_int)
    Frot_Lp = _Frot_cosI(hg_int,E_Lp,L_p,cosI,alpha)
    dJr_Lm, dL_Lm, dLz_Lm, dJrJr_Lm, dLL_Lm, dLzLz_Lm, dJrL_Lm, dJrLz_Lm, dLLz_Lm = orbitAverageActionCoeffs(Jr,L_m,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,m_test,hg_int)
    Frot_Lm = _Frot_cosI(hg_int,E_Lm,L_m,cosI,alpha)

    # Jr-component

    DJrF = dJr*Frot

    dJrJrF_Jr = (dJrJr_Jrp*Frot_Jrp-dJrJr_Jrm*Frot_Jrm)/(2.0*eps*_L0)
    dJrLF_L = (dJrL_Lp*Frot_Lp-dJrL_Lm*Frot_Lm)/(2.0*eps*_L0)

    fluxJr = DJrF - 0.5*(dJrJrF_Jr+dJrLF_L)

    # L-component

    DLF = dL*Frot

    dJrLF_Jr = (dJrL_Jrp*Frot_Jrp-dJrL_Jrm*Frot_Jrm)/(2.0*eps*_L0)
    dLLF_L = (dLL_Lp*Frot_Lp-dLL_Lm*Frot_Lm)/(2.0*eps*_L0)

    fluxL = DLF - 0.5*(dJrLF_Jr+dLLF_L)

    return fluxJr, fluxL
end


# https://cuda.juliagpu.org/stable/usage/multitasking/
# https://stackoverflow.com/questions/61905127/what-is-the-difference-between-threads-spawn-and-threads-threads
# https://stackoverflow.com/questions/55447363/julia-spawn-computing-jobs-sequentially-instead-of-parallel

# Computes the 2D-diffusion rate dF/dt in (Jr,L) space
function dFdt2D_JrL(Jr::Float64, L::Float64, m_field::Float64,
            alpha::Float64=alphaRot, nbCosI::Int64=50, nbAvr::Int64=nbAvr_default,
            nbw::Int64=nbw_default,
            nbvartheta::Int64=nbvartheta_default, nbphi::Int64=nbphi_default,
            nbu::Int64=nbu0, eps::Float64=10^(-5), m_test::Float64=m_field, hg_int::HG_Interpolate=hg_int_default)

    sumJr = 0.0
    sumL = 0.0


    sumJr_t = zeros(Float64, Threads.nthreads())
    sumL_t = zeros(Float64, Threads.nthreads())


    Threads.@threads for i=1:nbCosI
        cosI = -1.0 + 2.0/nbCosI*(i-0.5)

        fJr_p, _ = flux2D_JrL(Jr+eps,L,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,eps,m_test,hg_int)
        fJr_m, _ = flux2D_JrL(Jr-eps,L,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,eps,m_test,hg_int)

        dJr = (fJr_p-fJr_m)/(2.0*eps)

        _, fL_p  = flux2D_JrL(Jr,L+eps,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,eps,m_test,hg_int)
        _, fL_m  = flux2D_JrL(Jr,L-eps,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,nbu,eps,m_test,hg_int)

        dL = (fL_p-fL_m)/(2.0*eps)

        it = Threads.threadid()

        sumJr_t[it] += dJr
        sumL_t[it]  += dL
    end


    for it=1:Threads.nthreads()

        sumJr += sumJr_t[it]
        sumL += sumL_t[it]   
    end 

    sumJr *= 2.0/nbCosI
    sumL *= 2.0/nbCosI

    return -(sumJr+sumL)
end