
struct CharEmbedding{M <: DenseMatrix}
   embedding :: M
end

@functor CharEmbedding

CharEmbedding(alphabetsize::Integer, embeddingdim=512) = CharEmbedding(Dense(alphabetsize, embeddingdim).W) |> gpu
function CharEmbedding(alphabet, embeddingdim=512)
   alphabetsize = length(alphabet)
   CharEmbedding(alphabetsize, embeddingdim) |> gpu
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
   return reshape(output, (time, 1, embeddingdim, batchsize)) # [t×c×b]
end

function convblock(nchannels::Pair{<:Integer,<:Integer} = (512=>512),
                   σ = identity,
                   filtersize::Integer = 5,
                   pdrop = 0.5f0;
                   pad = (filtersize-1)÷2, kwargs...)
   # "The convolutional layers in the network are regularized using dropout with probability 0.5"
   [Conv((filtersize,1), nchannels; pad=(pad,0), kwargs...),
    BatchNorm(last(nchannels), σ),
    Dropout(pdrop)]
end

###
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
   return BLSTM(forward, backward, Int(out)) |> gpu
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
function (m::BLSTM)(Xs₄::DenseArray{<:Real,4})
   Xs = permutedims(dropdims(Xs₄; dims=2), (2, 3, 1)) # [t×d×b] -> [d×b×t]
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.outdim, size(Xs,3), size(Xs,2))
   axisYs₁ = axes(Ys, 1)
   time_f = axes(Ys, 2)
   time_b = reverse(time_f)
   @inbounds begin
      # get forward and backward slice indices
      slice_f = axisYs₁[begin:m.outdim]
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

###
struct LocationAwareAttention{M₃ <: DenseMatrix, T <: DenseArray{<:Real,4}, M <: DenseMatrix, V′ <: DenseVector}
   dense :: Dense{typeof(identity), M, V′} # W & b
   F :: T
   U :: M
   V :: M
   w :: V′
end

@functor LocationAwareAttention

function LocationAwareAttention(dense::Dense{typeof(identity)}, F::T, U::M, V::M, w::V′) where {T <: DenseArray{<:Real,4}, M <: DenseMatrix, V′ <: DenseVector}
   M₃ = addparent(M, tensor₃of(M))
   LocationAwareAttention{M₃, T, M, V′}(dense, F, U, V, w)
end

function LocationAwareAttention(encodingdim=512, location_featuredim=32, attentiondim=128, querydim=1024, filtersize=31)
   @assert isodd(filtersize)
   dense = Dense(querydim, attentiondim)
   convF = Conv((filtersize,1), 2=>location_featuredim, pad = ((filtersize-1)÷2,0))
   denseU = Dense(location_featuredim, attentiondim)
   denseV = Dense(encodingdim, attentiondim)
   densew = Dense(attentiondim, 1)
   LocationAwareAttention(dense, convF.weight, denseU.W, denseV.W, vec(densew.W)) |> gpu
end

function Base.show(io::IO, m::LocationAwareAttention)
   encodingdim = size(m.V, 2)
   location_featuredim = size(m.F, 3)
   attentiondim, querydim = size(m.dense.W)
   print(io, "LocationAwareAttention($encodingdim, $location_featuredim, $attentiondim, $querydim)")
end

function (m::LocationAwareAttention)(values::T, keys::DenseArray{<:Real,3}, query::M, lastweights::M, Σweights::M) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
   time, batchsize = size(lastweights)
   rdims = (time, 1, batchsize)
   lastweights³ = reshape(lastweights, rdims)
   Σweights³ = reshape(Σweights, rdims)
   weights_cat³ = [lastweights³ Σweights³]::T
   weights_cat⁴ = reshape(weights_cat³, (time, 1, 2, batchsize))
   pad = ((size(m.F, 1) - 1) ÷ 2, 0)
   cdims = DenseConvDims(weights_cat⁴, m.F; padding=pad)
   return m(values, keys, query, weights_cat⁴, cdims)
end

function (m::LocationAwareAttention{M₃})(values::DenseArray{<:Real,3}, keys::T₃₃, query::DenseMatrix, weights_cat⁴::DenseArray{<:Real,4}, cdims::DenseConvDims) where {M₃ <: DenseMatrix, T₃₃ <: DenseArray{<:Real,3}}
   attentiondim = length(m.w)
   _, time, batchsize = size(values)
   # location features
   fs = dropdims(conv(weights_cat⁴, m.F, cdims); dims=2) # F✶weights_cat
   Uf = reshape(m.U * reshape(permutedims(fs, (2,1,3)), Val(2)), (attentiondim, time, batchsize))
   # @ein Uf[a,t,b] := m.U[a,c] * fs[t,c,b] # dispatches to batched_contract (vetted)
   # Uf = einsum(EinCode{((1,2), (3,2,4)), (1,3,4)}(), (m.U, fs))::T₃₃
   # #=check=# Uf ≈ reduce(hcat, [reshape(m.U * fs[t,:,:], size(m.U,1), 1, :) for t ∈ axes(fs,1)])
   Ws⁺b = reshape(m.dense(query), (attentiondim, 1, batchsize)) # (a -> b) & (t -> c -> b)
   tanhWs⁺Vh⁺Uf⁺b = tanh.(Ws⁺b .+ keys .+ Uf)
   # @ein energies[t,b] := m.w[a] * tanhWs⁺Vh⁺Uf⁺b[a,t,b] # dispatches to batched_contract (vetted)
   energies = einsum(EinCode{((1,), (1,2,3)), (2,3)}(), (m.w, tanhWs⁺Vh⁺Uf⁺b))::M₃
   # #=check=# energies == reduce(vcat, [m.w'tanhWs⁺Vh⁺Uf⁺b[:,t,:] for t ∈ axes(tanhWs⁺Vh⁺Uf⁺b,2)])
   weights = softmax(energies) # α
   # @ein context[d,b] := values[d,t,b] * weights[t,b] # dispatches to batched_contract (vetted)
   context = einsum(EinCode{((1,2,3), (2,3)), (1,3)}(), (values, weights))::M₃
   # #=check=# context ≈ reduce(hcat, [sum(weights[t,b] * values[:,t,b] for t ∈ axes(values,2)) for b ∈ axes(values,3)])
   return context, weights
end

###
struct PreNet{D <: Dense}
   dense₁ :: D
   dense₂ :: D
   pdrop  :: Float32
end

Flux.trainable(m::PreNet) = (m.dense₁, m.dense₂)
@functor PreNet

function PreNet(in::Integer, out::Integer=256, pdrop=0.5f0, σ=leakyrelu)
   @assert 0 < pdrop < 1
   PreNet(Dense(in, out, σ), Dense(out, out, σ), Float32(pdrop)) |> gpu
end

function Base.show(io::IO, m::PreNet)
   out, in = size(m.dense₁.W)
   pdrop = m.pdrop
   σ = m.dense₁.σ
   print(io, "PreNet($in, $out, $pdrop, $σ)")
end

# "In order to introduce output variation at inference time, dropout with probability 0.5 is applied only to layers in the pre-net of the autoregressive decoder"
(m::PreNet)(x::DenseVecOrMat) = dropout(m.dense₂(dropout(m.dense₁(x), m.pdrop)), m.pdrop)

###
mutable struct State{M <: DenseMatrix}
   hs::NTuple{4, M}
end

Flux.trainable(m::State) = ()

@functor State

###
struct Decoder{M <: DenseMatrix, M₃ <: DenseMatrix, V <: DenseVector, T₄ <: DenseArray{<:Real,4}, D <: Dense}
   attention :: LocationAwareAttention{M₃, T₄, M, V}
   prenet    :: PreNet{D}
   lstms     :: Chain{NTuple{2, Recur{LSTMCell{M, V}}}}
   frameproj :: Dense{typeof(identity), M, V}
   state     :: State{M}
end

Flux.trainable(m::Decoder) = (m.attention, m.prenet, m.lstms, m.frameproj)
@functor Decoder

function Decoder(dims, filtersizes, pdrops)
   attention = LocationAwareAttention(dims.encoding, dims.location_feature, dims.attention, dims.query, filtersizes.attention)
   prenet = PreNet(dims.melfeatures, dims.prenet, pdrops.prenet)
   lstms = Chain(LSTM(dims.prenet + dims.encoding, dims.query), LSTM(dims.query, dims.query)) |> gpu
   frameproj = Dense(dims.query + dims.encoding, dims.melfeatures) |> gpu
   # initialize parameters
   query   = zeros(Float32, dims.query, 0)
   weights = zeros(Float32, 0, 0)
   frame   = zeros(Float32, dims.melfeatures, 0)
   state = State((query, weights, weights, frame)) |> gpu
   Decoder(attention, prenet, lstms, frameproj, state)
end

function Flux.reset!(m::Decoder)
   typeofdecoder = typeof(m)
   error("$typeofdecoder cannot be reset without information about input dimensions (batch size and its character length). Consider using reset!(m::Decoder, (time_in, batchsize)::NTuple{2,Integer}) or reset!(m::Decoder, input::DenseMatrix) instead.")
end

Flux.reset!(m::Decoder, input::DenseMatrix) = Flux.reset!(m, size(input))
function Flux.reset!(m::Decoder, (time_in, batchsize)::NTuple{2,Integer})
   Flux.reset!(m.lstms)
   #initialize dimensions
   querydim = size(m.attention.dense.W, 2)
   nmelfeatures = length(m.frameproj.b)
   # initialize parameters
   query   = gpu(zeros(Float32, querydim, batchsize))
   weights = gpu(zeros(Float32, time_in, batchsize))
   frame   = gpu(zeros(Float32, nmelfeatures, batchsize))
   m.state.hs = (query, weights, weights, frame)
   return nothing
end

function Base.show(io::IO, m::Decoder)
   print(io, """Decoder(
                   $(m.attention),
                   $(m.prenet),
                   $(m.lstms),
                   $(m.frameproj)
                )""")
end

function (m::Decoder{M})(values::DenseArray{<:Real,3}, keys::DenseArray{<:Real,3}) where M <: DenseMatrix
   (query, weights, Σweights, frame) = m.state.hs
   context, weights = m.attention(values, keys, query, weights, Σweights)
   Σweights += weights
   prenetoutput = m.prenet(frame)
   prenetoutput_context = [prenetoutput; context]::M
   query = m.lstms(prenetoutput_context)::M
   query_context = [query; context]::M
   frame = m.frameproj(query_context)
   m.state.hs = (query, weights, Σweights, frame)
   return frame, query_context
end

###
struct Tacotron₂{T₄ <: DenseArray{<:Real,4}, T₃₃ <: DenseArray{<:Real,3}, M <: DenseMatrix, M₃ <: DenseMatrix, V <: DenseVector, D <: Dense, C₃ <: Chain, C₅ <: Chain}
   che        :: CharEmbedding{M}
   convblock₃ :: C₃
   blstm      :: BLSTM{M, V}
   decoder    :: Decoder{M, M₃, V, T₄, D}
   stopproj   :: Dense{typeof(identity), M, V}
   postnet    :: C₅
end

@functor Tacotron₂

function Tacotron₂(che::CharEmbedding{M}, convblock₃::C₃, blstm::BLSTM{M,V}, decoder::Decoder{M,M₃,V,T₄,D}, stopproj::Dense{typeof(identity),M,V}, postnet::C₅) where {T₄ <: DenseArray{<:Real,4}, M <: DenseMatrix, M₃ <: DenseMatrix, V <: DenseVector, D <: Dense, C₃ <: Chain, C₅ <: Chain}
   T₃₃ = tensor₃of(M₃)
   Tacotron₂{T₄, T₃₃, M, M₃, V, D, C₃, C₅}(che, convblock₃, blstm, decoder, stopproj, postnet)
end

function Tacotron₂(dims::NamedTuple{(:alphabet,:encoding,:attention,:location_feature,:prenet,:query,:melfeatures,:postnet), <:NTuple{8,Integer}},
   filtersizes::NamedTuple{(:encoding,:attention,:postnet),<:NTuple{3,Integer}},
   pdrops::NamedTuple{(:encoding,:prenet,:postnet),<:NTuple{3,AbstractFloat}})
   @assert iseven(dims.encoding)

   che = CharEmbedding(dims.alphabet, dims.encoding)
   convblock₃ = Chain(reduce(vcat, map(_ -> convblock(dims.encoding=>dims.encoding, leakyrelu, filtersizes.encoding, pdrops.encoding), 1:3))...) |> gpu
   blstm = BLSTM(dims.encoding, dims.encoding÷2)

   decoder = Decoder(dims, filtersizes, pdrops)
   # will apply sigmoid implicitly during loss calculation with logitbinarycrossentropy for numerical stability
   stopproj = Dense(dims.query + dims.encoding, 1#=, σ=#) |> gpu

   nchmel, nch, fs, pdrop = dims.melfeatures, dims.postnet, filtersizes.postnet, pdrops.postnet
   postnet = Chain([convblock(nchmel=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nch, tanh, fs, pdrop);
                    convblock(nch=>nchmel, identity, fs, pdrop)]...) |> gpu
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

function Flux.reset!(m::Tacotron₂)
   error("Tacotron₂ cannot be reset without information about input dimensions (batch size and its character length). Consider using reset!(m::Tacotron₂, time_in′batchsize::NTuple{2,Integer}) or reset!(m::Tacotron₂, input::DenseMatrix) instead.")
end

Flux.reset!(m::Tacotron₂, input::DenseMatrix) = Flux.reset!(m::Tacotron₂, size(input))
function Flux.reset!(m::Tacotron₂, time_in′batchsize::NTuple{2,Integer})
   Flux.reset!(m.blstm)
   Flux.reset!(m.decoder, time_in′batchsize)
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

function (m::Tacotron₂{T₄,T₃₃})(textindices::DenseMatrix{<:Integer}, time_out::Integer) where {T₄ <: DenseArray{<:Real,4}, T₃₃ <: DenseArray{<:Real,3}}
   # dimensions
   batchsize = size(textindices, 2)
   nmelfeatures = length(m.decoder.frameproj.b)
   # encoding stage
   chex = m.che(textindices)
   convblock₃x = m.convblock₃(chex)::T₄
   values = m.blstm(convblock₃x)
   # @ein keys[a,t,b] := m.attention.V[a,d] * values[d,t,b] # dispatches to batched_contract (vetted)
   keys = einsum(EinCode{((1,2), (2,3,4)), (1,3,4)}(), (m.decoder.attention.V, values))::T₃₃
   # #=check=# Vh ≈ reduce(hcat, [reshape(m.decoder.attention.V * values[:,t,:], size(m.V,1), 1, :) for t ∈ axes(values,2)])
   decodings = (_ -> m.decoder(values, keys)).(1:time_out)
   frames = first.(decodings)
   query_contexts = last.(decodings)
   σ⁻¹stoprobs = reshape(m.stopproj(reduce(hcat, query_contexts)), (batchsize, time_out))
   # #=check=# σ⁻¹stoprobs == reduce(hcat, reshape.(m.stopproj.(query_contexts), Val(1)))
   prediction = reshape(reduce(hcat, frames), (nmelfeatures, batchsize, time_out))
   # #=check=# prediction == cat(frames...; dims=3)
   melprediction = reshape(permutedims(prediction, (3,1,2)), (time_out, 1, nmelfeatures, batchsize))
   melprediction⁺residual = melprediction + m.postnet(melprediction)::T₄
   return melprediction, melprediction⁺residual, σ⁻¹stoprobs
end

###
function loss(model::Tacotron₂, textindices::DenseMatrix{<:Integer}, meltarget::DenseArray{<:Real,4}, stoptarget::DenseMatrix{<:Real})
   melprediction, melprediction⁺residual, σ⁻¹stoprobs = model(textindices, size(meltarget, 1))
   l = mse(melprediction, meltarget) +
       mse(melprediction⁺residual, meltarget) +
       mean(logitbinarycrossentropy.(σ⁻¹stoprobs, stoptarget))
   return l
end

loss(model::Tacotron₂, (textindices, meltarget, stoptarget)::Tuple{DenseMatrix{<:Integer}, DenseArray{<:Real,4}, DenseMatrix{<:Real}}) = loss(model, textindices, meltarget, stoptarget)
