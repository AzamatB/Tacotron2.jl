# recursion approach
function decode(m::Tacotron₂, values::T, keys::T, stepsleft::Integer) where T <: DenseArray{<:Real,3}
   #initialize dimensions
   _, time_in, batchsize = size(values)
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # initialize parameters
   query   = gpu(zeros(Float32, querydim, batchsize))
   weights = gpu(zeros(Float32, time_in, 1, batchsize))
   frame   = gpu(zeros(Float32, nmelfeatures, batchsize))

   context, weights = m.attention(values, keys, query, weights, weights)
   prenetoutput = m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   query = m.lstms(prenetoutput_context)
   query_context = [query; context]
   frame = m.frameproj(query_context)
   decoding₁ = (frame, query_context)
   decode(m, values, keys, stepsleft-1, query, weights, weights, frame, decoding₁)
end

function decode(m::Tacotron₂, values::T, keys::T, stepsleft::Integer, query::M, weights::T, Σweights::T, frame::M, decodings) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
   (stepsleft == 0) && (return decodings)
   context, weights = m.attention(values, keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   query = m.lstms(prenetoutput_context)
   query_context = [query; context]
   frame = m.frameproj(query_context)
   decoding = (frame, query_context)
   decodings = [decodings; decoding]
   decode(m, values, keys, stepsleft-1, query, weights, Σweights, frame, decodings)
end

###
function (m::Tacotron₂)(textindices::DenseMatrix{<:Integer}, time_out::Integer)
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
   decodings = decode(m, values, keys, time_out)
   frames = first.(decodings)
   query_contexts = last.(decodings)
   σ⁻¹stoprobs = reshape(m.stopproj(reduce(hcat, query_contexts)), (batchsize, time_out))
   # #=check=# σ⁻¹stoprobs == reduce(hcat, reshape.(m.stopproj.(query_contexts), Val(1)))
   prediction = reshape(reduce(hcat, frames), (nmelfeatures, batchsize, time_out))
   # #=check=# prediction == cat(frames...; dims=3)
   melprediction = permutedims(prediction, (3,1,2))
   melprediction⁺residual = melprediction + m.postnet(melprediction)
   return melprediction, melprediction⁺residual, σ⁻¹stoprobs
end
