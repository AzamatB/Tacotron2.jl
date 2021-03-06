using Zygote: @adjoint
### helper functions for working with array types
addparent(A::Type{<:AbstractArray}, P′::DataType = A) = A
addparent(A::Type{CuArray{T,N,Nothing}}, P′::DataType = A) where {T,N} = CuArray{T,N,P′}
addparent(A::Type{CuArray{T,N,P}}, P′::DataType) where {T,N,P} = CuArray{T,N,P′}

vectorof(::Type{<:Array{T}}) where T = Vector{T}
matrixof(::Type{<:Array{T}}) where T = Matrix{T}
tensor₃of(::Type{<:Array{T}}) where T = Array{T,3}

vectorof(::Array{T}) where T = Vector{T}
matrixof(::Array{T}) where T = Matrix{T}
tensor₃of(::Array{T}) where T = Array{T,3}

vectorof(::Type{CuArray{T,N,P}}) where {T,N,P} = CuVector{T,P}
matrixof(::Type{CuArray{T,N,P}}) where {T,N,P} = CuMatrix{T,P}
tensor₃of(::Type{CuArray{T,N,P}}) where {T,N,P} = CuArray{T,3,P}

vectorof(::CuArray{T,N,P}) where {T,N,P} = CuVector{T,P}
matrixof(::CuArray{T,N,P}) where {T,N,P} = CuMatrix{T,P}
tensor₃of(::CuArray{T,N,P}) where {T,N,P} = CuArray{T,3,P}

### Work around for github.com/FluxML/Flux.jl/issues/1084
function Flux.dropout(x, p)
   q = 1 - p
   y = rand!(similar(x))
   y .= Flux._dropout_kernel.(y, p, q)
   x .* y
end

@adjoint function Flux.dropout(x, p)
   q = 1 - p
   y = rand!(similar(x))
   y .= Flux._dropout_kernel.(y, p, q)
   return x .* y, Δ -> (Δ .* y, nothing)
end

### Custom adjoint definitions
@adjoint function Base.reduce(::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 2))
   return reduce(hcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_horz(sz, Δ, A), cumsizes, As))
end

@adjoint function Base.reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 1))
   return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_vert(sz, Δ, A), cumsizes, As))
end

veclength(::Tuple) = 1
veclength(x::AbstractVector{<:Tuple}) = length(x)
pull_block_vert(idx, ȳ, x::Tuple) = ȳ[idx]
pull_block_vert(lastidx, ȳ, x::AbstractVector{<:Tuple}) = ȳ[(lastidx-length(x)+1):lastidx]

@adjoint function Base.vcat(xs::Union{AbstractVector{T},T}...) where T <: Tuple
   lastidx = Ref(0)
   lastidxs = ntuple(length(xs)) do i
      @inbounds lastidx[] += veclength(xs[i])
   end
   return vcat(xs...), function (ȳ)
      (map((lastidx, x) -> pull_block_vert(lastidx, ȳ, x), lastidxs, xs)...,)
   end
end

pull_block_horz(lastidx, ȳ, x::AbstractArray{<:Number,3}) = ȳ[:, (lastidx-size(x,2)+1):lastidx, :]

@adjoint function Base.hcat(xs::T₃...) where T₃ <: AbstractArray{<:Number,3}
   lastidx = Ref(0)
   lastidxs = ntuple(length(xs)) do i
      @inbounds lastidx[] += size(xs[i], 2)
   end
   return hcat(xs...), function (ȳ)
      (map((lastidx, x) -> pull_block_horz(lastidx, ȳ, x), lastidxs, xs)...,)
   end
end

Zygote.refresh()
