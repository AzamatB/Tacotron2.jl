using Flux
using Flux: @functor, Recur, LSTMCell
using Zygote
using Zygote: Buffer
using OMEinsum

include("dataprepLJSpeech/dataprep.jl")

struct CharEmbedding{M <: DenseMatrix, D <: AbstractDict}
   alphabet  :: D
   embedding :: M
end

@functor CharEmbedding (embedding,)

function CharEmbedding(alphabet::AbstractVector{Char}, embedding_dim=512)
   sort!(alphabet)
   alphabet′ = Dict(v => k for (k, v) ∈ enumerate(alphabet))
   n = length(alphabet′)
   n == length(alphabet) || throw("alphabet contains duplicate characters")
   d = Dense(n, embedding_dim)
   CharEmbedding(alphabet′, gpu(permutedims(d.W)))
end

function Base.show(io::IO, l::CharEmbedding)
   alphabet = keys(sort(l.alphabet; by = last))
   dim = size(l.embedding, 2)
   print(io, "CharEmbedding(", alphabet, ", ", dim, ")")
end

getindices(dict::AbstractDict, chars) = getindex.((dict,), chars)

(m::CharEmbedding)(c::Char) = m.embedding[:, m.alphabet[c]]
(m::CharEmbedding)(chars) = m.embedding[:, getindices(m.alphabet, chars)]
function (m::CharEmbedding)(textsbatch::AbstractVector{<:DenseVector})
   indices = getindices.((m.alphabet,), textsbatch)
   embeddings = Buffer(m.embedding, length(first(textsbatch)), size(m.embedding,2), length(textsbatch))
   setindex!.((embeddings,), getindex.((m.embedding,), indices, :), :, :, axes(embeddings,3))
   # the same as
   # for (k, idcs) ∈ enumerate(indices)
   #    embeddings[:,:,k] = m.embedding[idcs,:]
   # end
   # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
   return copy(embeddings)
end

function ConvBlock(nlayers::Integer, σ=leakyrelu; filter_size=5, nchannels=512, pdrop=0.5)
   # "The convolutional layers in the network are regularized using dropout with probability 0.5"
   padding = (filter_size-1, 0)
   layers = reduce(vcat, map(1:nlayers) do _
      [Conv((filter_size,), nchannels=>nchannels; pad=padding),
       BatchNorm(nchannels, σ),
       Dropout(pdrop)]
   end)
   return Chain(layers...)
end

"""
    BLSTM(in::Integer, out::Integer)

Constructs a bidirectional LSTM layer.
"""
struct BLSTM{M <: DenseMatrix, V <: DenseVector}
   forward  :: Recur{LSTMCell{M,V}}
   backward :: Recur{LSTMCell{M,V}}
   outdim  :: Int
end

@functor BLSTM (forward, backward)

function BLSTM(in::Integer, out::Integer)
   forward  = LSTM(in, out)
   backward = LSTM(in, out)
   return BLSTM(forward, backward, out)
end

function BLSTM(forward::Recur{LSTMCell{M,V}}, backward::Recur{LSTMCell{M,V}}) where {M <: DenseMatrix, V <: DenseVector}
    size(forward.cell.Wi, 2) == size(backward.cell.Wi, 2) || throw(DimensionMismatch("input dimension, $(size(forward.cell.Wi, 2)), of the forward-time LSTM layer does not match the input dimension, $(size(backward.cell.Wi, 2)), of the backward-time LSTM layer"))

    outdim = length(forward.cell.h)
    outdim == length(backward.cell.h) || throw(DimensionMismatch("output dimension, $outdim, of the forward-time LSTM layer does not match the output dimension, $(length(backward.cell.h)), of the backward-time LSTM layer"))
    return BLSTM(forward, backward, outdim)
end

Base.show(io::IO, l::BLSTM) = print(io,  "BLSTM(", size(l.forward.cell.Wi, 2), ", ", l.outdim, ")")

"""
    (m::BLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the bidirectional LSTM layer for a 3D tensor input.
Input tensor must be arranged in T×D×B (time duration × input dimension × # batches) order.
"""
function (m::BLSTM)(Xs::DenseArray{<:Real,3})
   Xs = permutedims(Xs, (2, 3, 1)) # [t×d×b] -> [d×b×t]
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.outdim, size(Xs,2), size(Xs,3))
   axisYs₁ = axes(Ys, 1)
   time_f = axes(Ys, 3)
   time_b = reverse(time_f)
   @inbounds begin
      # get forward and backward slice indices
      slice_f = axisYs₁[1:m.outdim]
      slice_b = axisYs₁[(m.outdim+1):end]
      # bidirectional run step
      setindex!.((Ys,),  m.forward.(view.((Xs,), :, :, time_f)), (slice_f,), :, time_f)
      setindex!.((Ys,), m.backward.(view.((Xs,), :, :, time_b)), (slice_b,), :, time_b)
      # the same as
      # @views for (t_f, t_b) ∈ zip(time_f, time_b)
      #    Ys[slice_f, :, t_f] =  m.forward(Xs[:, :, t_f])
      #    Ys[slice_b, :, t_b] = m.backward(Xs[:, :, t_b])
      # end
      # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
   end
   return permutedims(copy(Ys), (1,3,2)) # [d×b×t] -> [d×t×b]
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

@functor LocationAwareAttention

function LocationAwareAttention(attention_dim=128, encoding_dim=512, decoding_dim=1024, location_feature_dim=32)
   dense = Dense(decoding_dim, attention_dim)
   convF = Conv((31,), 1=>location_feature_dim, pad = (31-1)÷2)
   denseU = Dense(location_feature_dim, attention_dim)
   denseV = Dense(encoding_dim, attention_dim)
   densew = Dense(attention_dim, 1)
   LocationAwareAttention(dense, convF.pad, convF.weight, denseU.W, denseV.W, vec(densew.W))
end

function Base.show(io::IO, l::LocationAwareAttention)
   attention_dim, decoding_dim = size(l.dense.W)
   encoding_dim = size(l.V, 2)
   location_feature_dim = size(l.F, 3)
   print(io, "LocationAwareAttention($attention_dim, $encoding_dim, $decoding_dim, $location_feature_dim)")
end

function (laa::LocationAwareAttention)(query::DenseMatrix, values::T, Σweights::T) where T <: DenseArray{<:Real,3}
   dense, F, U, V, w = laa.dense, laa.F, laa.U, laa.V, laa.w
   cdims = DenseConvDims(Σweights, F; padding=laa.pad)
   fs = conv(Σweights, F, cdims) # F✶Σα
   # location keys
   @ein Uf[a,t,b] := U[a,c] * fs[t,c,b] # dispatches to batched_contract
   # check: Uf ≈ reduce(hcat, [reshape(U * fs[t,:,:], size(U,1), 1, :) for t ∈ axes(fs,1)])
   # memory keys
   @ein Vh[a,t,b] := V[a,d] * values[d,t,b] # dispatches to batched_contract
   # check: Vh ≈ reduce(hcat, [reshape(V * values[:,t,:], size(V,1), 1, :) for t ∈ axes(values,2)])
   Ws⁺b = reshape(dense(query), size(V,1), 1, :) # (a -> b) & (t -> c -> b)
   tanhWs⁺Vh⁺Uf⁺b = tanh.(Ws⁺b .+ Vh + Uf)
   @ein energies[t,b] := w[a] * tanhWs⁺Vh⁺Uf⁺b[a,t,b] # dispatches to batched_contract
   # check: energies == reduce(vcat, [w'tanhWs⁺Vh⁺Uf⁺b[:,t,:] for t ∈ axes(tanhWs⁺Vh⁺Uf⁺b,2)])
   α = softmax(energies)
   @ein context[d,b] := α[t,b] * values[d,t,b] # dispatches to batched_contract
   # check: context ≈ reduce(hcat, [sum(α[t,b] * values[:,t,b] for t ∈ axes(values,2)) for b ∈ axes(values,3)])
   return context, reshape(α, size(α,1), 1, :)
end

# "In order to introduce output variation at inference time, dropout with probability 0.5 is applied only to layers in the pre-net of the autoregressive decoder"
PreNet(in=80, out=256, σ=leakyrelu, pdrop=0.5) =
   Chain(Dense(in, out, σ), Dropout(pdrop),
        Dense(out, out, σ), Dropout(pdrop))

datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batch_size = 77
batches, alphabet = batch_dataset(metadatapath, melspectrogramspath, batch_size)
texts, targets = first(batches)
che = CharEmbedding(alphabet)
cb3 = ConvBlock(3)
blstm = BLSTM(512, 256)
laa = LocationAwareAttention()
prenet = PreNet()


function Base.permutedims(B::StridedArray, perm)
    @warn "permutedims is called"
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P, B, perm)
end



x = che(texts)
x = cb3(x)
x = blstm(x)
context, weights = laa(quer, x, Σweights)
x = prenet(ŷ_prev)

([x; context])





ŷ_prev = rand(Float32, 80, batch_size)
targets

Σweights += weights
decoding_dim=1024
quer = rand(Float32, decoding_dim, batch_size)
Σweights = rand(Float32, size(x, 2), 1, batch_size)


struct Tacotron2
   embedchar::CharEmbedding
   convblock3
   blstm::BLSTM
   attend::LocationAwareAttention
   convblock5
end
