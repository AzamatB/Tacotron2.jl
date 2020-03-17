using Zygote: @adjoint

vectorof(::Type{<:Array{T}}) where T = Vector{T}
matrixof(::Type{<:Array{T}}) where T = Matrix{T}
tensor₃of(::Type{<:Array{T}}) where T = Array{T,3}

vectorof(::Type{<:CuArray{T}}) where T = CuVector{T,Nothing}
matrixof(::Type{<:CuArray{T}}) where T = CuMatrix{T,Nothing}
tensor₃of(::Type{<:CuArray{T}}) where T = CuArray{T,3,Nothing}

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

@adjoint function reduce(::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 2))
   return reduce(hcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_horz(sz, Δ, A), cumsizes, As))
end

@adjoint function reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 1))
   return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_vert(sz, Δ, A), cumsizes, As))
end
