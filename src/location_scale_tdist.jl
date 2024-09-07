module location_scale_tdist

export LocationScaleT, pdf

using BayesBase
using LinearAlgebra
using Distributions
using SpecialFunctions


struct LocationScaleT <: ContinuousUnivariateDistribution
    ν  ::Real # Degrees-of-freedom
    μ  ::Real # Location parameter 
    σ  ::Real # Scale parameter (standard deviation)

    function LocationScaleT(ν::Float64, μ::Float64, σ::Float64)
        if ν <= 0.0; error("Degrees of freedom parameter must be positive."); end
        if σ <= 0.0; error("Standard deviation parameter must be positive."); end
        return new(ν, μ, σ)
    end
end

struct MvLocationScaleT{T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}} <: ContinuousMultivariateDistribution
    ν::N # Degrees-of-freedom
    μ::M # Mean vector
    Σ::S # Covariance matrix

    function MvLocationScaleT(ν::N, μ::M, Σ::S) where {T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}}
        if ν <= 0.0; error("Degrees of freedom parameter must be positive."); end
        if length(μ) !== size(Σ,1); error("Dimensionalities of mean and covariance matrix don't match."); end

        return new{T,N,M,S}(ν, μ, Σ)
    end
end

BayesBase.params(p::LocationScaleT) = (p.ν, p.μ, p.σ)
BayesBase.params(p::MvLocationScaleT) = (p.ν, p.μ, p.Σ)
BayesBase.dim(p::MvLocationScaleT) = length(p.μ)
BayesBase.mean(p::LocationScaleT) = p.μ
BayesBase.std(p::LocationScaleT) = sqrt(p.ν/(p.ν-2))*p.σ
BayesBase.var(p::LocationScaleT) = p.ν/(p.ν-2)*p.σ^2
BayesBase.precision(p::LocationScaleT) = inv(var(p))

function pdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return gamma( (ν+1)/2 ) / ( gamma(ν/2) *sqrt(π*ν)*σ ) * ( 1 + (x-μ)^2/(ν*σ^2) )^( -(ν+1)/2 )
end

function pdf(p::MvLocationScaleT, x)
    d = dims(p)
    ν, μ, Σ = params(p)
    return sqrt(1/( (ν*π)^d*det(Σ) )) * gamma((ν+d)/2)/gamma(ν/2) * (1 + 1/ν*(x-μ)'*inv(Σ)*(x-μ))^(-(ν+d)/2)
end

function logpdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return loggamma( (ν+1)/2 ) - loggamma(ν/2) - 1/2*log(πν) - log(σ) + ( -(ν+1)/2 )*log( 1 + (x-μ)^2/(ν*σ^2) )
end

function logpdf(p::MvLocationScaleT, x)
    d = dims(p)
    ν, μ, Σ = params(p)
    return -d/2*log(ν*π) - 1/2*logdet(Σ) +loggamma((ν+d)/2) -loggamma(ν/2) -(ν+d)/2*log(1 + 1/ν*(x-μ)'*inv(Σ)*(x-μ))
end

end
