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
include("model.jl")

###
# datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
datadir = "/home/azamat/Projects/TTS/LJSpeech-1.1"
metadatapath = joinpath(datadir, "metadata.csv")
melspectrogramspath = joinpath(datadir, "melspectrograms.jld2")

batchsize = 11
batches_trn, alphabet = build_batches(metadatapath, melspectrogramspath, batchsize)

batch = batches_trn[argmin(map(x -> size(last(x), 2), batches_trn))]
textindices, meltarget, stoptarget = batch
meltarget, stoptarget = meltarget |> gpu, stoptarget |> gpu
time_out = size(stoptarget, 2)

m = Tacotron₂(alphabet)

Flux.reset!(m, textindices)

##
m(textindices, time_out)

##
@time loss(m, textindices, meltarget, stoptarget)

θ = Flux.params(m)
# #=check=# length(θ) == 71 == 1 + 3(2 + 2 + 0) + 10 + ((2 + 4) + 4 + 10 + 2 + 4 + 0) + 2 + 5(2 + 2 + 0)

@time gs = gradient(θ) do
   loss(m, textindices, meltarget, stoptarget)
end

Juno.@profiler gradient(θ) do
   loss(m, batch)
end
