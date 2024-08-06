include("../sources/julia/Main.jl")

const Jr = 0.1
const L = 0.5
const cosI = 0.0


@time diff_coeffs = orbitAverageActionCoeffs(Jr,L,cosI,m_field)
@time diff_coeffs = orbitAverageActionCoeffs(Jr,L,cosI,m_field)

display(diff_coeffs)

const nbCosI = 20

@time dFdtJrL = dFdt2D_JrL(Jr, L, m_field, alphaRot, nbCosI)
@time dFdtJrL = dFdt2D_JrL(Jr, L, m_field, alphaRot, nbCosI)

println("dF/dt (Jr,L) = ", dFdtJrL)


