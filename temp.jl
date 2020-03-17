struct Decodings{T <: DenseArray{<:Real,3}, M <: Tacotron2}
   m :: M
   values :: T
   keys   :: T
   batchsize    :: Int
   querydim     :: Int
   nmelfeatures :: Int
   time_in      :: Int
   time_out     :: Int
end

Flux.trainable(d::Decodings) = (d.m,)
@functor Decodings

Base.eltype(::Type{<:Decodings{T}}) where T <: DenseArray{<:Real,3} = Tuple{matrixof(T), vectorof(T)}
Base.length(itr::Decodings) = itr.time_out

function Base.iterate(itr::Decodings)
    (itr.time_out <= 0) && (return nothing)
    batchsize, time_in = itr.batchsize, itr.time_in
    # initialize parameters
    query    = gpu(zeros(Float32, itr.querydim, batchsize))
    weights  = gpu(zeros(Float32, time_in, 1, batchsize))
    Σweights = weights
    frame    = gpu(zeros(Float32, itr.nmelfeatures, batchsize))

    context, weights = itr.m.attention(itr.values, itr.keys, query, weights, Σweights)
    Σweights = weights
    prenetoutput = itr.m.prenet(frame)
    query = itr.m.lstms([prenetoutput; context])
    query_context = [query; context]
    frame = itr.m.frameproj(query_context)
    σ⁻¹pstopᵀ = itr.m.stopproj(query_context)
    σ⁻¹pstop = reshape(σ⁻¹pstopᵀ, Val(1))
    i = (frame, σ⁻¹pstop)
    state = (1, query, weights, Σweights, frame)
    return i, state
end

function Base.iterate(itr::Decodings, state::Tuple{Int,M,T,T,M}) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
    (t, query, weights, Σweights, frame) = state
    (t == itr.time_out) && (return nothing)
    t += 1
    context, weights = itr.m.attention(itr.values, itr.keys, query, weights, Σweights)
    Σweights += weights
    prenetoutput = itr.m.prenet(frame)
    query = itr.m.lstms([prenetoutput; context])
    query_context = [query; context]
    frame = itr.m.frameproj(query_context)
    σ⁻¹pstopᵀ = itr.m.stopproj(query_context)
    σ⁻¹pstop = reshape(σ⁻¹pstopᵀ, Val(1))
    i = (frame, σ⁻¹pstop)
    state = (t, query, weights, Σweights, frame)
    return i, state
end

function (m::Tacotron2)(textindices::DenseMatrix{<:Integer}, time_out::Integer)
   # dimensions
   time_in, batchsize = size(textindices)
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # encodings
   values = m.encoder(textindices)
   @ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
   # #=check=# Vh ≈ reduce(hcat, [reshape(m.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
   decodings′ = Decodings(m, values, keys, batchsize, querydim, nmelfeatures, time_in, time_out)
   decodings = [decoding for decoding ∈ decodings′]
   frames = first.(decodings)
   σ⁻¹pstops = last.(decodings)
   σ⁻¹stoprobs = reduce(hcat, σ⁻¹pstops)
   prediction = reshape(reduce(hcat, frames), nmelfeatures, batchsize, time_out)
   # #=check=# prediction == cat(frames...; dims=3)
   melprediction = permutedims(prediction, (3,1,2))
   melprediction⁺residual = melprediction + m.postnet(melprediction)
   return melprediction, melprediction⁺residual, σ⁻¹stoprobs
end
