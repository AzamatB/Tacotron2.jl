using Flux
using Flux: @functor, Recur, LSTMCell, dropout
using Zygote
using Zygote: Buffer, @adjoint
using OMEinsum
using NamedTupleTools

include("dataprepLJSpeech/dataprep.jl")

struct CharEmbedding{M <: DenseMatrix}
   embedding :: M
end

@functor CharEmbedding

CharEmbedding(alphabet_size::Integer, embedding_dim=512) = CharEmbedding(gpu(Dense(alphabet_size, embedding_dim).W))
function CharEmbedding(alphabet, embedding_dim=512)
   alphabet_size = length(alphabet)
   CharEmbedding(alphabet_size, embedding_dim)
end

function Base.show(io::IO, m::CharEmbedding)
   embedding_dim, alphabet_size = size(m.embedding)
   print(io, "CharEmbedding($alphabet_size, $embedding_dim)")
end

(m::CharEmbedding)(charidxs::Integer) = m.embedding[:,charidxs]
function (m::CharEmbedding)(indices::DenseMatrix{<:Integer})
   time, batch_size = size(indices)
   embedding_dim = size(m.embedding, 1)
   output = permutedims(reshape(m.embedding[:,reshape(indices, Val(1))], embedding_dim, time, batch_size), (2,1,3))
   # #=check=# output == permutedims(cat(map(eachcol(indices)) do indicesᵢ
   #    m.embedding[:,indicesᵢ]
   # end...; dims=3), (2,1,3))
   return output # [t×c×b]
end

function convblock(nchannels::Pair{<:Integer,<:Integer} = (512=>512),
                   σ = identity,
                   filtersize::Integer = 5,
                   pdrop = 0.5f0;
                   pad = (filtersize-1,0), kwargs...)
   # "The convolutional layers in the network are regularized using dropout with probability 0.5"
   [Conv((filtersize,), nchannels; pad=pad, kwargs...),
    BatchNorm(last(nchannels), σ),
    Dropout(pdrop)] |> gpu
end

"""
    BLSTM(in::Integer, out::Integer)

Constructs a bidirectional LSTM layer.
"""
struct BLSTM{M <: DenseMatrix, V <: DenseVector}
   forward  :: Recur{LSTMCell{M,V}}
   backward :: Recur{LSTMCell{M,V}}
   outdim   :: Int
end

@functor BLSTM (forward, backward)

function BLSTM(in::Integer, out::Integer)
   forward  = LSTM(in, out)
   backward = LSTM(in, out)
   return BLSTM(gpu(forward), gpu(backward), out)
end

function Base.show(io::IO, m::BLSTM)
   in = size(m.forward.cell.Wi, 2)
   out = m.outdim
   print(io, "BLSTM($in, $out)")
end

"""
    (m::BLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the bidirectional LSTM layer for a 3D tensor input.
Input tensor must be arranged in T×D×B (time duration × input dimension × # batches) order.
"""
function (m::BLSTM)(Xs::DenseArray{<:Real,3})
   Xs = permutedims(Xs, (2, 3, 1)) # [t×d×b] -> [d×b×t]
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.outdim, size(Xs,3), size(Xs,2))
   axisYs₁ = axes(Ys, 1)
   time_f = axes(Ys, 2)
   time_b = reverse(time_f)
   @inbounds begin
      # get forward and backward slice indices
      slice_f = axisYs₁[1:m.outdim]
      slice_b = axisYs₁[(m.outdim+1):end]
      # bidirectional run step
      setindex!.((Ys,),  m.forward.(view.((Xs,), :, :, time_f)), (slice_f,), time_f, :)
      setindex!.((Ys,), m.backward.(view.((Xs,), :, :, time_b)), (slice_b,), time_b, :)
      # the same as
      # @views for (t_f, t_b) ∈ zip(time_f, time_b)
      #    Ys[slice_f, t_f, :] =  m.forward(Xs[:, :, t_f])
      #    Ys[slice_b, t_b, :] = m.backward(Xs[:, :, t_b])
      # end
      # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
   end
   return copy(Ys) # [d×t×b]
end

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @functor

struct LocationAwareAttention{A <: DenseArray, M <: DenseMatrix, V <: DenseVector, D <: Dense}
   dense :: D # W & b
   pad   :: NTuple{2,Int}
   F :: A
   U :: M
   V :: M
   w :: V
end

@functor LocationAwareAttention (dense, F, U, V, w)

function LocationAwareAttention(encoding_dim=512, location_feature_dim=32, attention_dim=128, decoding_dim=1024, filtersize=31)
   @assert isodd(filtersize)
   dense = Dense(decoding_dim, attention_dim)
   convF = Conv((filtersize,), 2=>location_feature_dim, pad = (filtersize-1)÷2)
   denseU = Dense(location_feature_dim, attention_dim)
   denseV = Dense(encoding_dim, attention_dim)
   densew = Dense(attention_dim, 1)
   LocationAwareAttention(gpu(dense), convF.pad, gpu(convF.weight), gpu(denseU.W), gpu(denseV.W), gpu(vec(densew.W)))
end

function Base.show(io::IO, m::LocationAwareAttention)
   encoding_dim = size(m.V, 2)
   location_feature_dim = size(m.F, 3)
   attention_dim, decoding_dim = size(m.dense.W)
   print(io, "LocationAwareAttention($encoding_dim, $location_feature_dim, $attention_dim, $decoding_dim)")
end

function (m::LocationAwareAttention)(values::T, keys::T, query::M, Σweights::M, weightsᵢ₋₁::M) where {T <: DenseArray{<:Real,3}, M <: DenseMatrix}
   time, batch_size = size(Σweights)
   attention_dim = length(m.w)
   rdims = (time, 1, batch_size)
   weights_cat = [reshape(Σweights, rdims) reshape(weightsᵢ₋₁, rdims)]
   cdims = DenseConvDims(weights_cat, m.F; padding=m.pad)
   # location features
   fs = conv(weights_cat, m.F, cdims) # F✶weights_cat
   @ein Uf[a,t,b] := m.U[a,c] * fs[t,c,b] # dispatches to batched_contract
   # check: Uf ≈ reduce(hcat, [reshape(m.U * fs[t,:,:], size(m.U,1), 1, :) for t ∈ axes(fs,1)])
   Ws⁺b = reshape(m.dense(query), attention_dim, 1, batch_size) # (a -> b) & (t -> c -> b)
   tanhWs⁺Vh⁺Uf⁺b = tanh.(Ws⁺b .+ keys .+ Uf)
   @ein energies[t,b] := m.w[a] * tanhWs⁺Vh⁺Uf⁺b[a,t,b] # dispatches to batched_contract
   # check: energies == reduce(vcat, [m.w'tanhWs⁺Vh⁺Uf⁺b[:,t,:] for t ∈ axes(tanhWs⁺Vh⁺Uf⁺b,2)])
   weights = softmax(energies) # α
   @ein context[d,b] := values[d,t,b] * weights[t,b] # dispatches to batched_contract
   # check: context ≈ reduce(hcat, [sum(weights[t,b] * values[:,t,b] for t ∈ axes(values,2)) for b ∈ axes(values,3)])
   return context, weights
end

struct PreNet{D<:Dense}
   dense₁ :: D
   dense₂ :: D
   pdrop  :: Float32
end

@functor PreNet (dense₁, dense₂)

function PreNet(in::Integer, out::Integer=256, pdrop=0.5f0, σ=leakyrelu)
   @assert 0 < pdrop < 1
   PreNet(Dense(in, out, σ) |> gpu, Dense(out, out, σ) |> gpu, Float32(pdrop))
end

function Base.show(io::IO, m::PreNet)
   out, in = size(m.dense₁.W)
   pdrop = m.pdrop
   σ = m.dense₁.σ
   print(io, "PreNet($in, $out, $pdrop, $σ)")
end

# "In order to introduce output variation at inference time, dropout with probability 0.5 is applied only to layers in the pre-net of the autoregressive decoder"
(m::PreNet)(x::DenseVecOrMat) = dropout(m.dense₂(dropout(m.dense₁(x), m.pdrop)), m.pdrop)

struct Tacotron2{E <: Chain, A <: LocationAwareAttention, P <: PreNet, L <: Chain, D₁ <: Dense, D₂ <: Dense, C <: Chain}
   encoder   :: E
   attention :: A
   prenet    :: P
   lstms     :: L
   frameproj :: D₁
   stopproj  :: D₂
   postnet   :: C
end

@functor Tacotron2

function Tacotron2(dims::NamedTuple{(:alphabet,:encoding,:attention,:location_feature,:prenet,:decoding,:melfeatures,:postnet), <:NTuple{8,Integer}},
   filtersizes::NamedTuple{(:encoding,:attention,:postnet),<:NTuple{3,Integer}},
   pdrops::NamedTuple{(:encoding,:prenet,:postnet),<:NTuple{3,AbstractFloat}})
   @assert iseven(dims.encoding)

   che = CharEmbedding(dims.alphabet, dims.encoding)
   convblock₃ = Chain(reduce(vcat, map(_ -> convblock(dims.encoding=>dims.encoding, leakyrelu, filtersizes.encoding, pdrops.encoding), 1:3))...) |> gpu
   blstm = BLSTM(dims.encoding, dims.encoding÷2)
   encoder = Chain(che, convblock₃.layers..., blstm)

   attention = LocationAwareAttention(dims.encoding, dims.location_feature, dims.attention, dims.decoding, filtersizes.attention)
   prenet = PreNet(dims.melfeatures, dims.prenet, pdrops.prenet)
   lstms = Chain(LSTM(dims.prenet + dims.encoding, dims.decoding), LSTM(dims.decoding, dims.decoding))

   frameproj = Dense(dims.decoding + dims.encoding, dims.melfeatures)
   stopproj = Dense(dims.decoding + dims.encoding, 1, σ)

   nchmel, nch, fs, pdrop = dims.melfeatures, dims.postnet, filtersizes.postnet, pdrops.postnet
   postnet = Chain([convblock(nchmel=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nchmel, identity, fs, pdrop)]...)
   Tacotron2(encoder, attention, prenet, lstms, frameproj, stopproj, postnet)
end

function Tacotron2(alphabet,
   otherdims=(encoding=512, attention=128, location_feature=32, prenet=256, decoding=1024, melfeatures=80, postnet=512),
   filtersizes = (encoding=5, attention=31, postnet=5),
   pdrop=0.5f0)
   dims = merge((alphabet = length(alphabet),), otherdims)
   pdrops = (encoding=pdrop, prenet=pdrop, postnet=pdrop)
   Tacotron2(dims, filtersizes, pdrops)
end

function Base.show(io::IO, m::Tacotron2)
   print(io, """Tacotron2(
                   $(m.encoder),
                   $(m.attention),
                   $(m.prenet),
                   $(m.lstms),
                   $(m.frameproj),
                   $(m.stopproj),
                   $(m.postnet)
                )""")
end



# need to be initialized
query = zeros(Float32, decoding_dim, batch_size)
Σweights = zeros(Float32, size(values, 2), 1, batch_size)
lastframe = zeros(Float32, 80, batch_size)


values = m.encoder(textindices)
@ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract
# check: Vh ≈ reduce(hcat, [reshape(m.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
context, weights = m.attention(values, keys, query, Σweights)





mt = deepcopy(m)
m = m.attention
m = mt



decoding_dim = 1024
batch_size









m = Tacotron2(alphabet)

lstms([prenet(lastframe); context])
values = encoder(textindices)
context, weights = attention(values, query, Σweights)

y: [c×b×t]
prenet: [c×bt]
lstms: [c×b×t]
proj: [c×bt]


1
postnet: [t,c,b]
1
y: [c×b×t]

y: [t,c,b]
1
prenet: [c×bt]
lstms: [c×b×t]
proj: [c×bt]
1

postnet: [t,c,b]
y: [t,c,b]

c×b
context


nmels✶batch_size = nmelfeatures*batch_size


ŷ_last = permutedims(ŷ_last, (2,3,1))

resize!(reshape(ŷ_last, Val(1)), nmels✶batch_size)




datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batch_size = 77
batches, alphabet = build_batches(metadatapath, melspectrogramspath, batch_size)
textindices, targets = rand(batches)


x = che(textindices)
x = cb3(x)
x = blstm(x)
decoding_dim = 1024
attention_dim = 512
out_dim = 80
x_prenet = prenet(ŷ_last)


Σweights += weights
targets

x = lstms([x_prenet; context])

x_cat_context = [x; context]
frame = pred_projection(x_cat_context)
pstop = stop_projection(x_cat_context)
