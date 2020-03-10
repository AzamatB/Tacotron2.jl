using Flux
using Flux: @functor, Recur, LSTMCell
using Zygote
using Zygote: Buffer

"""
    BLSTM(in::Integer, out::Integer)

Constructs a bidirectional LSTM layer.
"""
struct BLSTM{M <: DenseMatrix, V <: DenseVector}
   forward  :: Recur{LSTMCell{M,V}}
   backward :: Recur{LSTMCell{M,V}}
   dim_out  :: Int
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

Base.show(io::IO, l::BLSTM) = print(io,  "BLSTM(", size(l.forward.cell.Wi, 2), ", ", l.dim_out, ")")

"""
    (m::BLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the bidirectional LSTM layer for a 3D tensor input.
Input tensor must be arranged in D×T×B (input dimension × time duration × # batches) order.
"""
function (m::BLSTM)(Xs::DenseArray{<:Real,3})
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.dim_out, size(Xs,2), size(Xs,3))
   axisYs₁ = axes(Ys, 1)
   time    = axes(Ys, 2)
   rev_time = reverse(time)
   @inbounds begin
      # get forward and backward slice indices
      slice_f = axisYs₁[1:m.dim_out]
      slice_b = axisYs₁[(m.dim_out+1):end]
      # bidirectional run step
      setindex!.(Ref(Ys),  m.forward.(view.(Ref(Xs), :, time, :)), Ref(slice_f), time, :)
      setindex!.(Ref(Ys), m.backward.(view.(Ref(Xs), :, rev_time, :)), Ref(slice_b), rev_time, :)
      # the same as
      # @views for (t_f, t_b) ∈ zip(time, rev_time)
      #    Ys[slice_f, t_f, :] =  m.forward(Xs[:, t_f, :])
      #    Ys[slice_b, t_b, :] = m.backward(Xs[:, t_b, :])
      # end
      # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
   end
   return copy(Ys)
end

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @functor

include("dataprepLJSpeech/dataprep.jl")

struct CharEmbedding{M <: DenseMatrix, D <: AbstractDict}
   alphabet  :: D
   embedding :: M
end

@functor CharEmbedding (embedding,)

function CharEmbedding(alphabet::AbstractVector{Char}, embedding_dim::Integer = 512)
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
function (m::CharEmbedding)(textsbatch::AbstractVector)
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


datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batches, alphabet = batch_dataset(metadatapath, melspectrogramspath, 32)
texts, targets = first(batches)
m = CharEmbedding(alphabet)
convblock3 = ConvBlock(3)

x = m(texts)
x = convblock3(x)

permutedims(x, (2, 3, 1))






blstm = BLSTM(512, 256)
println(conv)
println(convblock3)

labels = sort(unique(text))

xs = Flux.onehotbatch((char for char ∈ text), labels)

xs1 = char_embedding(xs)'
xs1 = reshape(xs1, size(xs1,1), :, 1)
xs2 = conv(xs1)
xs2 = permutedims(xs2, (2, 1, 3))
xs3 = blstm(xs2)

values = xs3
hs = values
L = size(hs,2)

# attention weights
α = attend(s, hs)
α = reshape(softmax(rand(L)), :, 1, 1)
# glimpse
g = dropdims(sum(α .* hs; dims=2); dims=2)



conv = Conv((31,), 1=>32, pad = (31-1)÷2)
conv(α)

struct LocationAwareAttention{M <: DenseMatrix, V <: DenseVector, A <: DenseArray, D <: Dense}
   dense :: D # W & b
   F :: A
   U :: M
   V :: M
   w :: V
end

@functor LocationAwareAttention

function LocationAwareAttention(attention_dim::Integer = 128,
                                encoding_dim::Integer = 512,
                                decoding_dim::Integer = 1024,
                                location_feature_dim::Integer = 32)
   dense = Dense(decoding_dim, attention_dim)
   c = Conv((31,), 1=>location_feature_dim, pad = (31-1)÷2)
   denseU = Dense(location_feature_dim, attention_dim)
   denseV = Dense(encoding_dim, attention_dim)
   densew = Dense(attention_dim,1)
   LocationAwareAttention(dense, c.weight, denseU.W, denseV.W, vec(densew.W))
end

laa = LocationAwareAttention()

function (laa::LocationAwareAttention)(α::DenseArray{<:Real,3}, s::DenseVector, hs)
   dense, F, U, V, w = laa.dense, laa.F, laa.U, laa.V, laa.w
   cdims = DenseConvDims(α, F; padding=(15, 15))
   f = permutedims(conv(α, F, cdims), (2,1,3)) # F✶α
   energiesᵢ = w'tanh.(dense(s) .+ V * reshape(hs, size(hs, 1), :) + U * reshape(f, size(f, 1), :))
   αᵢ = softmax(energiesᵢ; dims=2)
end
