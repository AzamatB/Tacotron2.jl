using CuArrays
using Flux
using Flux: @functor, Recur, LSTMCell, dropout, mse, logitbinarycrossentropy
using Zygote
using Zygote: Buffer
using OMEinsum
using NamedTupleTools
using Statistics

CuArrays.allowscalar(false)

include("utils.jl")
include("dataprepLJSpeech/dataprep.jl")

struct CharEmbedding{M <: DenseMatrix}
   embedding :: M
end

@functor CharEmbedding

CharEmbedding(alphabetsize::Integer, embeddingdim=512) = CharEmbedding(gpu(Dense(alphabetsize, embeddingdim).W))
function CharEmbedding(alphabet, embeddingdim=512)
   alphabetsize = length(alphabet)
   CharEmbedding(alphabetsize, embeddingdim)
end

function Base.show(io::IO, m::CharEmbedding)
   embeddingdim, alphabetsize = size(m.embedding)
   print(io, "CharEmbedding($alphabetsize, $embeddingdim)")
end

(m::CharEmbedding)(charidxs::Integer) = m.embedding[:,charidxs]
function (m::CharEmbedding)(indices::DenseMatrix{<:Integer})
   time, batchsize = size(indices)
   embeddingdim = size(m.embedding, 1)
   output = permutedims(reshape(m.embedding[:,reshape(indices, Val(1))], embeddingdim, time, batchsize), (2,1,3))
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

Flux.trainable(m::BLSTM) = (m.forward, m.backward)
@functor BLSTM

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

struct LocationAwareAttention{T <: DenseArray{<:Real,3}, M <: DenseMatrix, V <: DenseVector, D <: Dense}
   dense :: D # W & b
   pad   :: NTuple{2,Int}
   F :: T
   U :: M
   V :: M
   w :: V
end

Flux.trainable(m::LocationAwareAttention) = (m.dense, m.F, m.U, m.V, m.w)
@functor LocationAwareAttention

function LocationAwareAttention(encodingdim=512, location_featuredim=32, attentiondim=128, querydim=1024, filtersize=31)
   @assert isodd(filtersize)
   dense = Dense(querydim, attentiondim)
   convF = Conv((filtersize,), 2=>location_featuredim, pad = (filtersize-1)÷2)
   denseU = Dense(location_featuredim, attentiondim)
   denseV = Dense(encodingdim, attentiondim)
   densew = Dense(attentiondim, 1)
   LocationAwareAttention(gpu(dense), convF.pad, gpu(convF.weight), gpu(denseU.W), gpu(denseV.W), gpu(vec(densew.W)))
end

function Base.show(io::IO, m::LocationAwareAttention)
   encodingdim = size(m.V, 2)
   location_featuredim = size(m.F, 3)
   attentiondim, querydim = size(m.dense.W)
   print(io, "LocationAwareAttention($encodingdim, $location_featuredim, $attentiondim, $querydim)")
end

function (m::LocationAwareAttention)(values::T, keys::T, query::DenseMatrix, lastweights::T, Σweights::T) where T <: DenseArray{<:Real,3}
   # weights_cat = cat(lastweights, Σweights; dims=2)::T
   weights_cat = [lastweights Σweights]::T
   cdims = DenseConvDims(weights_cat, m.F; padding=m.pad)
   return m(values, keys, query, weights_cat, cdims)
end
function (m::LocationAwareAttention)(values::T, keys::T, query::M, weights_cat::T, cdims::DenseConvDims) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
   # location features
   fs = conv(weights_cat, m.F, cdims) # F✶weights_cat
   # @ein Uf[a,t,b] := m.U[a,c] * fs[t,c,b] # dispatches to batched_contract (vetted)
   Uf = einsum(EinCode{((1,2), (3,2,4)), (1,3,4)}(), (m.U, fs))::T
   # #=check=# Uf ≈ reduce(hcat, [reshape(m.U * fs[t,:,:], size(m.U,1), 1, :) for t ∈ axes(fs,1)])
   time, _, batchsize = size(weights_cat)
   attentiondim = length(m.w)
   Ws⁺b = reshape(m.dense(query), attentiondim, 1, batchsize) # (a -> b) & (t -> c -> b)
   tanhWs⁺Vh⁺Uf⁺b = tanh.(Ws⁺b .+ keys .+ Uf)
   # @ein energies[t,b] := m.w[a] * tanhWs⁺Vh⁺Uf⁺b[a,t,b] # dispatches to batched_contract (vetted)
   energies = einsum(EinCode{((1,), (1,2,3)), (2,3)}(), (m.w, tanhWs⁺Vh⁺Uf⁺b))::M
   # #=check=# energies == reduce(vcat, [m.w'tanhWs⁺Vh⁺Uf⁺b[:,t,:] for t ∈ axes(tanhWs⁺Vh⁺Uf⁺b,2)])
   weights = softmax(energies) # α
   # @ein context[d,b] := values[d,t,b] * weights[t,b] # dispatches to batched_contract (vetted)
   context = einsum(EinCode{((1,2,3), (2,3)), (1,3)}(), (values, weights))::M
   # #=check=# context ≈ reduce(hcat, [sum(weights[t,b] * values[:,t,b] for t ∈ axes(values,2)) for b ∈ axes(values,3)])
   return context, reshape(weights, (time, 1, batchsize))
end

struct PreNet{D<:Dense}
   dense₁ :: D
   dense₂ :: D
   pdrop  :: Float32
end

Flux.trainable(m::PreNet) = (m.dense₁, m.dense₂)
@functor PreNet

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

struct Tacotron₂{E <: CharEmbedding, C₃ <: Chain, B <: BLSTM, A <: LocationAwareAttention, P <: PreNet, L <: Chain, D₁ <: Dense, D₂ <: Dense, C₅ <: Chain}
   che        :: E
   convblock₃ :: C₃
   blstm      :: B
   attention  :: A
   prenet     :: P
   lstms      :: L
   frameproj  :: D₁
   stopproj   :: D₂
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

   attention = LocationAwareAttention(dims.encoding, dims.location_feature, dims.attention, dims.query, filtersizes.attention)
   prenet = PreNet(dims.melfeatures, dims.prenet, pdrops.prenet)
   lstms = Chain(LSTM(dims.prenet + dims.encoding, dims.query), LSTM(dims.query, dims.query))

   frameproj = Dense(dims.query + dims.encoding, dims.melfeatures)
   # will apply sigmoid implicitly during loss calculation with logitbinarycrossentropy for numerical stability
   stopproj = Dense(dims.query + dims.encoding, 1 #=, σ =#)

   nchmel, nch, fs, pdrop = dims.melfeatures, dims.postnet, filtersizes.postnet, pdrops.postnet
   postnet = Chain([convblock(nchmel=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nchmel, identity, fs, pdrop)]...)
   Tacotron₂(che, convblock₃, blstm, attention, prenet, lstms, frameproj, stopproj, postnet)
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
                   $(m.attention),
                   $(m.prenet),
                   $(m.lstms),
                   $(m.frameproj),
                   $(m.stopproj),
                   $(m.postnet)
                )""")
end

function Flux.reset!(m::Tacotron₂)
   Flux.reset!(m.blstm)
   Flux.reset!(m.lstms)
end

function (m::Tacotron₂)(textindices::DenseMatrix{<:Integer}, time_out::Integer)
   # dimensions
   time_in, batchsize = size(textindices)
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # initialize parameters
   query₀    = gpu(zeros(Float32, querydim, batchsize))
   weights₀  = gpu(zeros(Float32, time_in, 1, batchsize))
   Σweights₀ = weights₀
   frame₀    = gpu(zeros(Float32, nmelfeatures, batchsize))
   return m(textindices, time_out, query₀, weights₀, Σweights₀, frame₀)
end

function (m::Tacotron₂)(textindices::DenseMatrix{<:Integer}, time_out::Integer, query::M, weights::T₃, Σweights::T₃, frame::M) where {M <: DenseMatrix, T₃ <: DenseArray{<:Real,3}}
   # encoding stage
   chex = m.che(textindices)
   convblock₃x = m.convblock₃(chex)::T₃
   values = m.blstm(convblock₃x)
   # @ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
   keys = einsum(EinCode{((1,2), (2,3,4)), (1,3,4)}(), (m.attention.V, values))::T₃
   # #=check=# Vh ≈ reduce(hcat, [reshape(m.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
   # preallocate output buffer
   nmelfeatures, batchsize = size(frame)
   prediction = Buffer(keys, nmelfeatures, batchsize, time_out)
   σ⁻¹stoprobs′ = Buffer(frame, batchsize, time_out)
   # main autoregressive loop
   for t ∈ 1:time_out
      context, weights = m.attention(values, keys, query, weights, Σweights)::Tuple{M,T₃}
      Σweights += weights
      prenetoutput = m.prenet(frame)
      prenetoutput_context = [prenetoutput; context]
      query = m.lstms(prenetoutput_context)::M
      query_context = [query; context]
      frame = m.frameproj(query_context)
      σ⁻¹pstopᵀ = m.stopproj(query_context)
      σ⁻¹stoprobs′[:,t] = reshape(σ⁻¹pstopᵀ, Val(1))
      prediction[:,:,t] = frame
   end
   σ⁻¹stoprobs = copy(σ⁻¹stoprobs′)
   melprediction = permutedims(copy(prediction), (3,1,2))
   melprediction⁺residual = melprediction + m.postnet(melprediction)::T₃
   return melprediction, melprediction⁺residual, σ⁻¹stoprobs
end

function loss(model::Tacotron₂, textindices::DenseMatrix{<:Integer}, meltarget::DenseArray{<:Real,3}, stoptarget::DenseMatrix{<:Real})
   melprediction, melprediction⁺residual, σ⁻¹stoprobs = model(textindices, size(meltarget, 1))
   l = mse(melprediction, meltarget) +
       mse(melprediction⁺residual, meltarget) +
       mean(logitbinarycrossentropy.(σ⁻¹stoprobs, stoptarget))
   return l
end

loss(model::Tacotron₂, (textindices, meltarget, stoptarget)::Tuple{DenseMatrix{<:Integer}, DenseArray{<:Real,3}, DenseMatrix{<:Real}}) = loss(model, textindices, meltarget, stoptarget)

# TODO 3. add implement iterators to get rid of the Buffer code in the forward pass
# TODO 4. add adjoint for hcat of 2 3D tensors and replace cat with hcat in the attentions forward pass

datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batchsize = 11
batches, alphabet = build_batches(metadatapath, melspectrogramspath, batchsize)
batch = batches[argmin(map(x -> size(last(x), 2), batches))]

textindices, meltarget, stoptarget = batch
time_out = size(stoptarget, 2)


m = Tacotron₂(alphabet)

@time loss(m, batch)

θ = Flux.params(m)

@time gs = gradient(θ) do
   loss(m, batch)
end

Juno.@profiler gradient(θ) do
   loss(m, batch)
end
