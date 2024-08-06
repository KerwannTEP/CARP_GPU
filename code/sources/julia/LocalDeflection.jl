function orbit_average!(list_coeffs_block::CuDeviceArray{T}, E::Float64, L::Float64, cosI::Float64, 
                        sinI::Float64, sma::Float64, ecc::Float64, sp::Float64, sa::Float64, 
                        m_field::Float64, alpha::Float64, nbAvr::Int64, nbw::Int64, nbvartheta::Int64,
                        nbphi::Int64, nint::Int64, nbThreadsPerBlocks::Int64, m_test::Float64, hg_int::HG_Interpolate) where T


    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    tid = threadIdx().x

    nbv = nbw*nbvartheta*nbphi 

    # https://cuda.juliagpu.org/stable/development/kernel/
    # list_coeffs_threads = CuStaticSharedArray(T, (512, 9))
    list_coeffs_threads = CuDynamicSharedArray(T, (nbThreadsPerBlocks, 9))


    list_coeffs_threads[tid,1] = 0.0
    list_coeffs_threads[tid,2] = 0.0
    list_coeffs_threads[tid,3] = 0.0
    list_coeffs_threads[tid,4] = 0.0
    list_coeffs_threads[tid,5] = 0.0
    list_coeffs_threads[tid,6] = 0.0
    list_coeffs_threads[tid,7] = 0.0
    list_coeffs_threads[tid,8] = 0.0
    list_coeffs_threads[tid,9] = 0.0

    

    
    pref = 2.0*pi^2/(nbw*nbvartheta*nbphi)
    pref *= 2.0*pi*_G^2*logCoulomb

    # @assert (0 < index && index <= nint)

    while (index <= nint)


        # Velocity indices 
        # index - 1 = iu - 1 + (iv-1)*(nbAvr)


        iv = div(index-1,nbAvr) + 1
        iu = index - (iv-1)*nbAvr


        # @assert (0 < iu && iu <= nbAvr)

        # v = (w,theta,phi)
        # iv-1 = iw-1 + (iang-1)*(nbw)
        iang = floor(Int64, (iv-1)/nbw) + 1
        iw = iv - (iang-1)*nbw

        # iang - 1 = iphi - 1 + (ivartheta-1)*nbphi
        ivartheta = floor(Int64, (iang-1)/nbphi) + 1
        iphi = iang - (ivartheta-1)*nbphi


        # Radial index + radial average

        uloc = -1+2*(iu-0.5)/nbAvr
        sloc = s_from_u_sma_ecc(uloc,sma,ecc)
        rloc = r_from_s(sloc)
        jac_loc = Theta(uloc,sp,sa)

        vr = sqrt(2.0*abs(E - psiEff(rloc,L)))
        vt = L/rloc
        vSq = vr^2 + vt^2
        v = sqrt(vSq)
        vr_v = vr/v
        vt_v = vt/v


        # Velocity integral + physical angle average

        vartheta = pi*(ivartheta-0.5)/nbvartheta
        phi = 2.0*pi*(iphi-0.5)/nbphi


        sinvartheta = sin(vartheta)
        cosvartheta = cos(vartheta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        wmax = v*cosvartheta + sqrt(abs(vSq*cosvartheta^2 - 2.0*E))

        w = wmax*(iw-0.5)/nbw

        w1 = w*cosvartheta
        w2 = w*sinvartheta*cosphi
        w3 = w*sinvartheta*sinphi

        Ep = E + 0.5*w^2 - v*w1

        v1p = v-w1
        v2p = -w2
        v3p = -w3

        Lp = rloc * sqrt(abs(v2p^2 + (vr_v*v3p-vt_v*v1p)^2 ))

        Ftot = _F(hg_int,Ep,Lp)


        nu = -(v1p*vt_v-v3p*vr_v)*cosI

        sum_g =  0.0
        sum_sinSqg = 0.0

        if (v2p == 0.0)
            sum_g = sign(-nu)
            sum_sinSqg = 0.5*sum_g#sign(-nu)
        else
            mu = nu/(v2p*sinI)
            if (abs(mu) >= 1.0)
                sum_g = sign(-nu)
                sum_sinSqg = 0.5*sum_g#sign(-nu)/2.0
            else
                sum_g = -2.0*asin(nu/(abs(v2p)*sinI))/pi
                # sum_sinSqg = 0.5*(-2.0*asin(nu/(abs(v2p)*sinI))/pi + 2.0/pi*(nu/(abs(v2p)*sinI))*sqrt(1.0-(nu/(abs(v2p)*sinI))^2))
                sum_sinSqg = 0.5*(sum_g + 2.0/pi*(nu/(abs(v2p)*sinI))*sqrt(1.0-(nu/(abs(v2p)*sinI))^2))
            end
        end

        dvPar = -pref * (m_field + m_test) * 2.0*sinvartheta*cosvartheta* wmax * Ftot * (1.0 + alpha*sum_g)
        dvPar2 = 2.0*pref * m_field * sinvartheta^3* wmax * w*Ftot * (1.0 + alpha*sum_g)
        dvPerp2 = 2.0*pref * m_field * sinvartheta*(1.0+cosvartheta^2)* wmax * w*Ftot * (1.0 + alpha*sum_g)
        sinSqdvPerSq = 2.0*pref * m_field * sinvartheta*(1.0+cosvartheta^2)* wmax * w*Ftot * (0.5 + alpha*sum_sinSqg)


        list_coeffs_threads[tid,1] += jac_loc * (0.5*dvPar2 + 0.5*dvPerp2 + v*dvPar)
        list_coeffs_threads[tid,2] += jac_loc * (rloc*vt_v*dvPar + 0.25*(rloc/vt)*dvPerp2)
        list_coeffs_threads[tid,3] += jac_loc * (v^2* dvPar2)
        list_coeffs_threads[tid,4] += jac_loc * (rloc*vt*dvPar2)
        list_coeffs_threads[tid,5] += jac_loc * (rloc^2*vt_v^2*dvPar2 + 0.5*rloc^2*vr_v^2*dvPerp2)
        list_coeffs_threads[tid,6] += jac_loc * (rloc*vt_v*cosI*dvPar)
        list_coeffs_threads[tid,7] += jac_loc * (rloc^2*cosI^2*(vt_v^2*dvPar2 + 0.5*vr_v^2*dvPerp2) + 0.5*rloc^2*sinSqdvPerSq*sinI^2)
        list_coeffs_threads[tid,8] += jac_loc * (rloc^2*cosI*(vt_v^2*dvPar2 + 0.5*vr_v^2*dvPerp2))
        list_coeffs_threads[tid,9] += jac_loc * (rloc*vt*cosI*dvPar2)

  
        # Increment index

        index += blockDim().x * gridDim().x

    end


    sync_threads()

    # Reduction
    i = div(blockDim().x,2);

    while (i != 0)
        if (tid <= i)
            list_coeffs_threads[tid,1] += list_coeffs_threads[tid+i,1]
            list_coeffs_threads[tid,2] += list_coeffs_threads[tid+i,2]
            list_coeffs_threads[tid,3] += list_coeffs_threads[tid+i,3]
            list_coeffs_threads[tid,4] += list_coeffs_threads[tid+i,4]
            list_coeffs_threads[tid,5] += list_coeffs_threads[tid+i,5]
            list_coeffs_threads[tid,6] += list_coeffs_threads[tid+i,6]
            list_coeffs_threads[tid,7] += list_coeffs_threads[tid+i,7]
            list_coeffs_threads[tid,8] += list_coeffs_threads[tid+i,8]
            list_coeffs_threads[tid,9] += list_coeffs_threads[tid+i,9]
        end
        i =  div(i,2)
        sync_threads()
    end

    if (tid == 1)
        list_coeffs_block[blockIdx().x,1] = list_coeffs_threads[1,1]
        list_coeffs_block[blockIdx().x,2] = list_coeffs_threads[1,2]
        list_coeffs_block[blockIdx().x,3] = list_coeffs_threads[1,3]
        list_coeffs_block[blockIdx().x,4] = list_coeffs_threads[1,4]
        list_coeffs_block[blockIdx().x,5] = list_coeffs_threads[1,5]
        list_coeffs_block[blockIdx().x,6] = list_coeffs_threads[1,6]
        list_coeffs_block[blockIdx().x,7] = list_coeffs_threads[1,7]
        list_coeffs_block[blockIdx().x,8] = list_coeffs_threads[1,8]
        list_coeffs_block[blockIdx().x,9] = list_coeffs_threads[1,9]
    end

    return nothing

end