


function orbitAverageEnergyCoeffs(sp::Float64, sa::Float64, cosI::Float64,
                m_field::Float64, alpha::Float64=alphaRot, nbAvr::Int64=nbAvr_default,
                nbw::Int64=nbw_default, nbvartheta::Int64=nbvartheta_default, 
                nbphi::Int64=nbphi_default, m_test::Float64=m_field, hg_int::HG_Interpolate=hg_int_default)


    E, L = E_L_from_sp_sa(sp,sa)

    sma, ecc = sma_ecc_from_sp_sa(sp,sa)
    sinI = sqrt(abs(1.0 - cosI^2))

    nint = nbAvr*nbw*nbvartheta*nbphi 


    # nbThreadsPerBlocks = 384 #1024
    numblocks = min(40, ceil(Int64, nint/nbThreadsPerBlocks))

    avrDE = 0.0
    avrDL = 0.0
    avrDEE = 0.0
    avrDEL = 0.0
    avrDLL = 0.0
    avrDLz = 0.0
    avrDLzLz = 0.0
    avrDLLz = 0.0
    avrDELz = 0.0

    list_coeffs_block = zeros(Float64, numblocks, 9)
    dev_list_coeffs_block = CuArray(list_coeffs_block)

    # https://cuda.juliagpu.org/stable/tutorials/introduction/
    
    @cuda threads=nbThreadsPerBlocks blocks=numblocks shmem=nbThreadsPerBlocks*9*sizeof(Float64) orbit_average!(dev_list_coeffs_block, E, L, 
                                                                                                                cosI, sinI, sma, ecc, sp, sa, 
                                                                                                                m_field, alpha, nbAvr, nbw, nbvartheta,
                                                                                                                nbphi, nint, nbThreadsPerBlocks, m_test, hg_int)

    list_coeffs_block = Array(dev_list_coeffs_block)

    for ib=1:numblocks

        avrDE += list_coeffs_block[ib,1]
        avrDL += list_coeffs_block[ib,2]
        avrDEE += list_coeffs_block[ib,3]
        avrDEL += list_coeffs_block[ib,4]
        avrDLL += list_coeffs_block[ib,5]
        avrDLz += list_coeffs_block[ib,6]
        avrDLzLz += list_coeffs_block[ib,7]
        avrDLLz += list_coeffs_block[ib,8]
        avrDELz += list_coeffs_block[ib,9]

    end
    
    halfperiod = 0.0


    for iu=1:nbAvr

        uloc = -1+2*(iu-0.5)/nbAvr
        jac_loc = Theta(uloc,sp,sa)

        
        halfperiod += jac_loc

    end

    avrDE /= halfperiod
    avrDL /= halfperiod
    avrDEE /= halfperiod
    avrDEL /= halfperiod
    avrDLL /= halfperiod
    avrDLz /= halfperiod
    avrDLzLz /= halfperiod
    avrDLLz /= halfperiod
    avrDELz /= halfperiod

    
    return avrDE, avrDL, avrDLz, avrDEE, avrDLL, avrDLzLz, avrDEL, avrDELz, avrDLLz
end



##################################################
# Orbit-averaged (Jr,L,Lz)-diffusion coefficients
##################################################

# Diffusion coefficients (orbit-averaged) in (Jr,L,Lz) space
function orbitAverageActionCoeffs(Jr::Float64, L::Float64, cosI::Float64, m_field::Float64,
                                alpha::Float64=alphaRot, nbAvr::Int64=nbAvr_default,
                                nbw::Int64=nbw_default,
                                nbvartheta::Int64=nbvartheta_default, nbphi::Int64=nbphi_default,
                                nbu::Int64=nbu0, m_test::Float64=m_field, hg_int::HG_Interpolate=hg_int_default)

    E = E_from_Jr_L(Jr,L,nbu)
    if (Jr > 0.0)
        sp, sa = sp_sa_from_E_L(E,L)
    else
        sc = _sc(E/_E0)
        sp, sa = sc, sc
    end

    avrDE, avrDL, avrDLz, avrDEE, avrDLL, avrDLzLz, avrDEL, avrDELz, avrDLLz = orbitAverageEnergyCoeffs(sp,sa,cosI,m_field,alpha,nbAvr,nbw,nbvartheta,nbphi,m_test,hg_int)

    dJrdE, dJrdL, d2JrdE2, d2JrdEL, d2JrdL2 = grad_Jr_E_L(E,L,nbu)

    avrDJr = dJrdE*avrDE+dJrdL*avrDL+(1/2)*d2JrdE2*avrDEE+(1/2)*d2JrdL2*avrDLL+d2JrdEL*avrDEL
    avrDJrJr = dJrdE^2*avrDEE+dJrdL^2*avrDLL+2*dJrdE*dJrdL*avrDEL
    avrDJrL = dJrdE*avrDEL+dJrdL*avrDLL
    avrDJrLz = dJrdE*avrDELz+dJrdL*avrDLLz

    return avrDJr, avrDL, avrDLz, avrDJrJr, avrDLL, avrDLzLz, avrDJrL, avrDJrLz, avrDLLz
end






