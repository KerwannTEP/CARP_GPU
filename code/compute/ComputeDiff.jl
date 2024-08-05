include("../sources/julia/Main.jl")

const Jr = 0.1
const L = 0.5
const cosI = 0.0


@time diff_coeffs = orbitAverageActionCoeffs(Jr,L,cosI,m_field)

display(diff_coeffs)