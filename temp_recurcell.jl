mutable struct State{M <: DenseMatrix, T₃ <: DenseArray{<:Real,3}}
   h::Tuple{NTuple{2,M}, NTuple{2,M}, M, T₃, T₃, M}
end

trainable(m::State) = ()

@functor State

struct Decoder{V <: DenseVector, M <: DenseMatrix, T₃ <: DenseArray{<:Real,3}, D <: Dense}
   attention :: LocationAwareAttention{T₃, M, V}
   prenet    :: PreNet{D}
   lstmcells :: NTuple{2,LSTMCell{M,V}}
   frameproj :: Dense{typeof(identity), M, V}
   state₀    :: Tuple{NTuple{2,M}, NTuple{2,M}, M, T₃, T₃, M}
   state     :: State{M, T₃}
end

trainable(m::Decoder) = (m.attention, m.prenet, m.lstmcells, m.frameproj, m.state₀[1:2])

@functor Decoder

function Decoder(dims, filtersizes, pdrops)
   attention = LocationAwareAttention(dims.encoding, dims.location_feature, dims.attention, dims.query, filtersizes.attention) |> gpu
   prenet = PreNet(dims.melfeatures, dims.prenet, pdrops.prenet)
   lstmcells = (LSTMCell(dims.prenet + dims.encoding, dims.query), LSTMCell(dims.query, dims.query)) |> gpu
   frameproj = Dense(dims.query + dims.encoding, dims.melfeatures)
   # initialize parameters
   h₁ = repeat.(Flux.hidden(lstmcells[1]), 1, 1)
   h₂ = repeat.(Flux.hidden(lstmcells[2]), 1, 1)
   query   = gpu(zeros(Float32, dims.query, 0))
   weights = gpu(zeros(Float32, 0, 1, 0))
   frame   = gpu(zeros(Float32, dims.melfeatures, 0))
   state₀ = (h₁, h₂, query, weights, weights, frame)
   Decoder(attention, prenet, lstmcells, frameproj, state₀, State(state₀))
end

function Flux.reset!(m::Decoder)
   typeofdecoder = typeof(m)
   error("$typeofdecoder cannot be reset without information about input dimensions (batch size and its character length). Consider using reset!(m::Decoder, (time_in, batchsize)::NTuple{2,Integer}) or reset!(m::Decoder, input::DenseMatrix) instead.")
end

Flux.reset!(m::Decoder, input::DenseMatrix) = Flux.reset!(m, size(input))
function Flux.reset!(m::Decoder, (time_in, batchsize)::NTuple{2,Integer})
   #initialize dimensions
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # initialize parameters
   h₁      = repeat.(m.state₀[1], 1, batchsize)
   h₂      = repeat.(m.state₀[2], 1, batchsize)
   query   = gpu(zeros(Float32, querydim, batchsize))
   weights = gpu(zeros(Float32, time_in, 1, batchsize))
   frame   = gpu(zeros(Float32, nmelfeatures, batchsize))
   m.state.h = (h₁, h₂, query, weights, weights, frame)
end

function Base.show(io::IO, m::Decoder)
   print(io, """Decoder(
                   $(m.attention),
                   $(m.prenet),
                   $(m.lstmcells),
                   $(m.frameproj)
                )""")
end

function (m::Decoder)(values::T, keys::T) where {T <: Array{<:Real,3}}
   (h₁, h₂, query, weights, Σweights, frame) = m.state.h
   context, weights = m.attention(values, keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]
   h₁, y₁ = m.lstmcells[1](h₁, prenetoutput_context)
   h₂, query = m.lstmcells[2](h₂, y₁)
   query_context = [query; context]
   frame = m.frameproj(query_context)
   m.state.h = (h₁, h₂, query, weights, Σweights, frame)
   return frame, query_context
end

###
struct Tacotron₂{T₃ <: Array{<:Real,3}, M <: DenseMatrix, V <: DenseVector, D <: Dense, C₃ <: Chain, C₅ <: Chain}
   che        :: CharEmbedding{M}
   convblock₃ :: C₃
   blstm      :: BLSTM{M, V}
   decoder    :: Decoder{V, M, T₃, D}
   stopproj   :: Dense{typeof(identity), M, V}
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

function Tacotron₂(alphabet,
   otherdims=(encoding=512, attention=128, location_feature=32, prenet=256, query=1024, melfeatures=80, postnet=512),
   filtersizes = (encoding=5, attention=31, postnet=5),
   pdrop=0.5f0)
   dims = merge((alphabet = length(alphabet),), otherdims)
   pdrops = (encoding=pdrop, prenet=pdrop, postnet=pdrop)
   Tacotron₂(dims, filtersizes, pdrops)
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

function (m::Tacotron₂{T₃})(textindices::DenseMatrix{<:Integer}, time_out::Integer) where T₃ <: DenseArray{<:Real,3}
   # dimensions
   batchsize = size(textindices, 2)
   nmelfeatures = length(m.decoder.frameproj.b)
   # encoding stage
   chex = m.che(textindices)
   convblock₃x = m.convblock₃(chex)::T₃
   values = m.blstm(convblock₃x)
   # @ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
   keys = einsum(EinCode{((1,2), (2,3,4)), (1,3,4)}(), (m.decoder.attention.V, values))::T₃
   # #=check=# Vh ≈ reduce(hcat, [reshape(m.decoder.attention.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
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

m = Tacotron₂(alphabet)

Flux.reset!(m, textindices)

@time m(textindices, time_out)

@time loss(m, textindices, meltarget, stoptarget)

@time gs = gradient(θ) do
   loss(m, textindices, meltarget, stoptarget)
end
