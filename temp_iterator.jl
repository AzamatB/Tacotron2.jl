# Warning: adjoint of the `collect(x::Decodings)` defined below is likely wrong
struct Decodings{T, M}
   m :: M
   values   :: T
   keys     :: T
   time_out :: Int
end

@adjoint function Decodings(m::Tacotron2, values::T, keys::T, time_out::Int) where T <: DenseArray{<:Real,3}
   return Decodings(m, values, keys, time_out), function (d̄)
      return d̄.m, d̄.values, d̄.keys, nothing
   end
end

# this adjoint is seems wrong
@adjoint function collect(x::Decodings)
   collect(x), function(ȳ)
      return (Decodings(first(ȳ)[3:end]..., length(ȳ)),)
   end
end

Flux.trainable(d::Decodings) = (d.m,)
@functor Decodings

Base.eltype(::Type{<:Decodings{T,M}}) where {T <: DenseArray{<:Real,3}, M <: Tacotron2} = Tuple{matrixof(T), vectorof(T), M, T, T}
Base.length(itr::Decodings) = itr.time_out

function Base.iterate(itr::Decodings)
   (itr.time_out <= 0) && (return nothing)
   #initialize dimensions
   _, time_in, batchsize = size(itr.values)
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # initialize parameters
   query   = gpu(zeros(Float32, querydim, batchsize))
   weights = gpu(zeros(Float32, time_in, 1, batchsize))
   frame   = gpu(zeros(Float32, nmelfeatures, batchsize))

   context, weights = itr.m.attention(itr.values, itr.keys, query, weights, weights)
   prenetoutput = itr.m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   query = itr.m.lstms(prenetoutput_context)
   query_context = [query; context]
   frame = itr.m.frameproj(query_context)
   σ⁻¹pstopᵀ = itr.m.stopproj(query_context)
   σ⁻¹pstop = reshape(σ⁻¹pstopᵀ, Val(1))
   i = (frame, σ⁻¹pstop, m, values, keys)
   state = (1, query, weights, weights, frame)
   return i, state
end

function Base.iterate(itr::Decodings, state::Tuple{Int,M,T,T,M}) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
   (t, query, weights, Σweights, frame) = state
   (t == itr.time_out) && (return nothing)
   t += 1
   context, weights = itr.m.attention(itr.values, itr.keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = itr.m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   query = itr.m.lstms(prenetoutput_context)
   query_context = [query; context]
   frame = itr.m.frameproj(query_context)
   σ⁻¹pstopᵀ = itr.m.stopproj(query_context)
   σ⁻¹pstop = reshape(σ⁻¹pstopᵀ, Val(1))
   i = (frame, σ⁻¹pstop, m, values, keys)
   state = (t, query, weights, Σweights, frame)
   return i, state
end

###
function (m::Tacotron2)(textindices::DenseMatrix{<:Integer}, time_out::Integer)
   # dimensions
   batchsize = size(textindices, 2)
   nmelfeatures = length(m.frameproj.b)
   # encoding stage
   chex = m.che(textindices)
   convblock₃x = m.convblock₃(chex)
   values = m.blstm(convblock₃x)
   # @ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
   keys = einsum(EinCode{((1,2), (2,3,4)), (1,3,4)}(), (m.attention.V, values))
   # #=check=# Vh ≈ reduce(hcat, [reshape(m.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
   # decodings′ = Zygote.bufferfrom(Vector{Tuple{Matrix{Float32},Vector{Float32}}}(undef, time_out))
   # decodings′ .= Decodings(m, values, keys, time_out)
   # decodings = copy(decodings′)
   # decodings = [decoding for decoding ∈ decodings′]
   decodings = collect(Decodings(m, values, keys, time_out))
   frames = first.(decodings)
   σ⁻¹pstops = getindex.(decodings, 2)
   σ⁻¹stoprobs = reduce(hcat, σ⁻¹pstops)
   prediction = reshape(reduce(hcat, frames), nmelfeatures, batchsize, time_out)
   # #=check=# prediction == cat(frames...; dims=3)
   melprediction = permutedims(prediction, (3,1,2))
   melprediction⁺residual = melprediction + m.postnet(melprediction)
   return melprediction, melprediction⁺residual, σ⁻¹stoprobs
end
