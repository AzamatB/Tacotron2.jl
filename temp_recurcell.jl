mutable struct DecoderCell{A <: LocationAwareAttention, P <: PreNet, L <: NTuple{2,LSTMCell}, D <: Dense, S <: Tuple{T,T,M,T₃,T₃,M} where {T <: NTuple{2,DenseVector}, M <: DenseMatrix, T₃ <: DenseArray{<:Real,3}}}
   attention :: A
   prenet    :: P
   lstmcells :: L
   frameproj :: D
   state₀    :: S
end

trainable(m::DecoderCell) = (m.attention, m.prenet, m.lstmcells, m.frameproj, m.state[1:2])

@functor DecoderCell

function DecoderCell(dims, filtersizes, pdrops)
   attention = LocationAwareAttention(dims.encoding, dims.location_feature, dims.attention, dims.query, filtersizes.attention)
   prenet = PreNet(dims.melfeatures, dims.prenet, pdrops.prenet)
   lstmcells = (LSTMCell(dims.prenet + dims.encoding, dims.query), LSTMCell(dims.query, dims.query))
   frameproj = Dense(dims.query + dims.encoding, dims.melfeatures)
   # initialize parameters
   h₁, h₂  = Flux.hidden(lstmcells[1]), Flux.hidden(lstmcells[2])
   query   = gpu(zeros(Float32, dims.query, 0))
   weights = gpu(zeros(Float32, 0, 1, 0))
   frame   = gpu(zeros(Float32, dims.melfeatures, 0))
   state₀ = (h₁, h₂, query, weights, weights, frame)
   DecoderCell(attention, prenet, lstmcells, frameproj, state₀)
end

Flux.hidden(m::DecoderCell) = m.state₀

function Decoder(args...)
   m = DecoderCell(args...)
   state = Flux.hidden(m)
   init = state[1:2]
   Recur(m, init, state)
end

function Flux.reset!(decoder::Recur{<:DecoderCell})
   typeofdecoder = typeof(decoder)
   error("$typeofdecoder cannot be reset without information about input dimensions (batch size and its character length). Consider using reset!(decoder::Recur{<:DecoderCell}, (time_in, batchsize)::NTuple{2,Integer}) or reset!(decoder::Recur{<:DecoderCell}, input::DenseMatrix) instead.")
end

Flux.reset!(decoder::Recur{<:DecoderCell}, input::DenseMatrix) = Flux.reset!(decoder, size(input))
function Flux.reset!(decoder::Recur{<:DecoderCell}, (time_in, batchsize)::NTuple{2,Integer})
   #initialize dimensions
   querydim = size(decoder.cell.attention.dense.W, 2)
   nmelfeatures = length(decoder.cell.frameproj.b)
   # initialize parameters
   h₁, h₂  = decoder.init
   query   = gpu(zeros(Float32, querydim, batchsize))
   weights = gpu(zeros(Float32, time_in, 1, batchsize))
   frame   = gpu(zeros(Float32, nmelfeatures, batchsize))
   decoder.state = (h₁, h₂, query, weights, weights, frame)
end

function Base.show(io::IO, m::DecoderCell)
   print(io, """DecoderCell(
                   $(m.attention),
                   $(m.prenet),
                   $(m.lstmcells),
                   $(m.frameproj)
                )""")
end

function (m::DecoderCell)((h₁, h₂, query, weights, Σweights, frame), values, keys)
   context, weights = m.attention(values, keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   h₁, y₁ = m.lstmcells[1](h₁, prenetoutput_context)
   h₂, query = m.lstmcells[2](h₂, y₁)
   query_context = [query; context]
   frame = m.frameproj(query_context)
   output = (frame, query_context)
   return (h₁, h₂, query, weights, Σweights, frame), output
end

decoder = Decoder(dims, filtersizes, pdrops)
Flux.hidden(decoder.cell)
Flux.reset!(decoder, textindices)

###
struct Tacotron₂{M <: DenseMatrix, V <: DenseVector, C₃ <: Chain, DC <: DecoderCell, D <: Dense, C₅ <: Chain}
   che        :: CharEmbedding{M}
   convblock₃ :: C₃
   blstm      :: BLSTM{M,V}
   decoder    :: Recur{DC}
   stopproj   :: D
   postnet    :: C₅
end

@functor Tacotron₂

function Tacotron₂(dims::NamedTuple{(:alphabet,:encoding,:attention,:location_feature,:prenet,:query,:melfeatures,:postnet), <:NTuple{8,Integer}},
   filtersizes::NamedTuple{(:encoding,:attention,:postnet),<:NTuple{3,Integer}},
   pdrops::NamedTuple{(:encoding,:prenet,:postnet),<:NTuple{3,AbstractFloat}})
   @assert iseven(dims.encoding)

   che = CharEmbedding(dims.alphabet, dims.encoding)
   convblock₃ = Chain(reduce(vcat, map(_ -> convblock(dims.encoding=>dims.encoding, leakyrelu, filtersizes.encoding, pdrops.encoding), 1:3))...) |> gpu
   blstm = BLSTM(dims.encoding, dims.encoding÷2)

   decoder = Decoder(dims, filtersizes, pdrops)
   # will apply sigmoid implicitly during loss calculation with logitbinarycrossentropy for numerical stability
   stopproj = Dense(dims.query + dims.encoding, 1#=, σ=#)

   nchmel, nch, fs, pdrop = dims.melfeatures, dims.postnet, filtersizes.postnet, pdrops.postnet
   postnet = Chain([convblock(nchmel=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nchmel, identity, fs, pdrop)]...)
   Tacotron₂(che, convblock₃, blstm, decoder, stopproj, postnet)
end

function Base.show(io::IO, m::Tacotron₂)
   print(io, """Tacotron₂(
                   $(m.che),
                   $(m.convblock₃),
                   $(m.blstm),
                   $(m.decoder),
                   $(m.stopproj),
                   $(m.postnet)
                )""")
end

function Flux.reset!(m::Tacotron₂)
   error("Tacotron₂ cannot be reset without information about input dimensions (batch size and its character length). Consider using reset!(m::Tacotron₂, time_in_batchsize::NTuple{2,Integer}) or reset!(m::Tacotron₂, input::DenseMatrix) instead.")
end

Flux.reset!(m::Tacotron₂, input::DenseMatrix) = Flux.reset!(m::Tacotron₂, size(input))
function Flux.reset!(m::Tacotron₂, time_in_batchsize::NTuple{2,Integer})
   Flux.reset!(m.blstm)
   Flux.reset!(m.decoder, time_in_batchsize)
end

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
   decodings = (_ -> m.decoder(values, keys)).(1:time_out)
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
