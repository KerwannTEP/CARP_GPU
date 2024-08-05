


function orbitAverageEnergyCoeffs(sp::Float64, sa::Float64, cosI::Float64,
                m_field::Float64, alpha::Float64=alphaRot, nbAvr::Int64=nbAvr_default,
                nbw::Int64=nbw_default, nbvartheta::Int64=nbvartheta_default, 
                nbphi::Int64=nbphi_default, m_test::Float64=m_field)


    E, L = E_L_from_sp_sa(sp,sa)
    Lz = L * cosI
    sma, ecc = sma_ecc_from_sp_sa(sp,sa)
    sinI = sqrt(abs(1.0 - cosI^2))

    nbThreadsPerBlocks = 1024
    nint = nbAvr*nbw*nbvartheta*nbphi 
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

    @cuda threads=nbThreadsPerBlocks blocks=numblocks orbit_average!(dev_list_coeffs_block, E, L, Lz, cosI, sinI, sma, ecc, sp, sa,
                                                                    m_field, alpha, nbAvr, nbw, nbvartheta,
                                                                    nbphi, nint, nbThreadsPerBlocks, m_test)

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
    

    halfperiod_threads = [0.0 for it=Threads.nthreads()]

    Threads.@threads for iu=1:nbAvr

        ithread = Threads.threadid()

        uloc = -1+2*(iu-0.5)/nbAvr
        jac_loc = Theta(uloc,sp,sa)

        halfperiod += jac_loc
        
        halfperiod_threads[ithread] += jac_loc

    end

    halfperiod = 0.0

    for it=1:Threads.nthreads()

        halfperiod += halfperiod_threads[it]

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






    




