using Flux
using Flux: @functor, Recur, LSTMCell, dropout
using Zygote
using Zygote: Buffer, @adjoint
using OMEinsum
using NamedTupleTools

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

function LocationAwareAttention(encodingdim=512, location_featuredim=32, attentiondim=128, decodingdim=1024, filtersize=31)
   @assert isodd(filtersize)
   dense = Dense(decodingdim, attentiondim)
   convF = Conv((filtersize,), 2=>location_featuredim, pad = (filtersize-1)÷2)
   denseU = Dense(location_featuredim, attentiondim)
   denseV = Dense(encodingdim, attentiondim)
   densew = Dense(attentiondim, 1)
   LocationAwareAttention(gpu(dense), convF.pad, gpu(convF.weight), gpu(denseU.W), gpu(denseV.W), gpu(vec(densew.W)))
end

function Base.show(io::IO, m::LocationAwareAttention)
   encodingdim = size(m.V, 2)
   location_featuredim = size(m.F, 3)
   attentiondim, decodingdim = size(m.dense.W)
   print(io, "LocationAwareAttention($encodingdim, $location_featuredim, $attentiondim, $decodingdim)")
end

function (m::LocationAwareAttention)(values::T, keys::T, query::M, lastweights::M, Σweights::M) where {T <: DenseArray{<:Real,3}, M <: DenseMatrix}
   time, batchsize = size(Σweights)
   attentiondim = length(m.w)
   rdims = (time, 1, batchsize)
   weights_cat = [reshape(lastweights, rdims) reshape(Σweights, rdims)]
   cdims = DenseConvDims(weights_cat, m.F; padding=m.pad)
   # location features
   fs = conv(weights_cat, m.F, cdims) # F✶weights_cat
   @ein Uf[a,t,b] := m.U[a,c] * fs[t,c,b] # dispatches to batched_contract (vetted)
   # #=check=# Uf ≈ reduce(hcat, [reshape(m.U * fs[t,:,:], size(m.U,1), 1, :) for t ∈ axes(fs,1)])
   Ws⁺b = reshape(m.dense(query), attentiondim, 1, batchsize) # (a -> b) & (t -> c -> b)
   tanhWs⁺Vh⁺Uf⁺b = tanh.(Ws⁺b .+ keys .+ Uf)
   @ein energies[t,b] := m.w[a] * tanhWs⁺Vh⁺Uf⁺b[a,t,b] # dispatches to batched_contract (vetted)
   # #=check=# energies == reduce(vcat, [m.w'tanhWs⁺Vh⁺Uf⁺b[:,t,:] for t ∈ axes(tanhWs⁺Vh⁺Uf⁺b,2)])
   weights = softmax(energies) # α
   @ein context[d,b] := values[d,t,b] * weights[t,b] # dispatches to batched_contract (vetted)
   # #=check=# context ≈ reduce(hcat, [sum(weights[t,b] * values[:,t,b] for t ∈ axes(values,2)) for b ∈ axes(values,3)])
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

function (m::Tacotron2)(textindices::DenseMatrix{<:Integer})
# dimensions
time, batchsize = size(textindices)
decodingdim = size(m.attention.dense.W, 2)
nmelfeatures = length(m.frameproj.b)
# encoding
values = m.encoder(textindices)
@ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
# #=check=# Vh ≈ reduce(hcat, [reshape(m.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
# initialize parameters
query    = gpu(zeros(Float32, decodingdim, batchsize))
weights  = gpu(zeros(Float32, time, batchsize))
Σweights = gpu(zeros(Float32, time, batchsize))
frame    = gpu(zeros(Float32, nmelfeatures, batchsize))
for _ ∈ 1:time
   context, weights = m.attention(values, keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = m.prenet(frame)
   decoding = m.lstms([prenetoutput; context])
   frame = m.frameproj([decoding; context])
   pstop = m.stopproj([decoding; context])
   (pstop > 0.5f0) && break
   @ein frame′[m,b] := frame[m,b] * pstop[u,b] # (vetted)
   # #=check=# frame′ == reduce(hcat, [pstop[1,b] * frame[:,b] for b ∈ axes(frame,2)])
end


@benchmark f0($pstop, $frame)

@edit OMEinsum.einsum(OMEinsum.EinCode{((1, 2), (3, 2)), (3, 2)}(), (pstop, frame))

function Base.permutedims(B::StridedArray, perm)
   # @warn "permutedims is called"
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P, B, perm)
end


function OMEinsum.batched_contract(iAs, A::AbstractArray, iBs, B::AbstractArray, iOuts::NTuple{NO,T}) where {NO,T}
   # @warn "batched_contract is called"
    pA, sAb, sAs, sAp, pB, sBs, sBb, sBp, sAB, pOut = OMEinsum.analyse_batched(iAs, size(A), iBs, size(B), iOuts)

    A, B = OMEinsum.align_eltypes(A, B)
    Apr = reshape(OMEinsum.conditioned_permutedims(A, pA), sAb, sAs, sAp)
    Bpr = reshape(OMEinsum.conditioned_permutedims(B, pB), sBs, sBb, sBp)
    AB = OMEinsum._batched_gemm('N','N', Apr, Bpr)
    AB = OMEinsum.conditioned_permutedims(reshape(AB, sAB...), [pOut...])
end



m = Tacotron2(alphabet)


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


nmels✶batchsize = nmelfeatures*batchsize


resize!(reshape(ŷ_last, Val(1)), nmels✶batchsize)


datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batchsize = 77
batches, alphabet = build_batches(metadatapath, melspectrogramspath, batchsize)
textindices, targets = rand(batches)
