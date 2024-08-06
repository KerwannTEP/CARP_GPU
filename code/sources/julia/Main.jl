
println("Compiling libraries...")

using HypergeometricFunctions
using SpecialFunctions
using StaticArrays # To have access to static arrays
using Interpolations # To have access to interpolation functions
using ArgParse

println("Compiling CUDA...")

@time using CUDA 


include("Args.jl") # Parsing the command-line
##################################################
# General parameters
##################################################

"""
    nbAvr_default

Default number of sampling points for the orbit-averaging integrals.
"""
const nbAvr_default = parsed_args["nbAvr"]

"""
    qCalc

Anisotropy parameter q for the Plummer model.
"""
const qCalc = parsed_args["q"]

"""
    alpha

Rotation parameter for the Lynden-Bell Demon.
Frot(E,L,Lz) = F(E,L) * (1 + alpha * g(Lz/L))
"""
const alphaRot = parsed_args["a"]

const nbu0 = parsed_args["nbu"]
const nbw_default = parsed_args["nbw"]
const nbphi_default = parsed_args["nbphi"]
const nbvartheta_default = parsed_args["nbvartheta"]

const nbThreadsPerBlocks = parsed_args["nbThreadsPerBlocks"]

##################################################
##################################################

include("Constants.jl")
include("Mean.jl")
include("EffectiveAnomaly.jl")

include("OrbitParameters.jl")
include("Bath.jl")
include("LocalDeflection.jl")
include("Grad_Jr.jl")
include("OrbitAverage.jl")

