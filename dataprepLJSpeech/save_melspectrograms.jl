using PyCall
using JLD2

load = pyimport("numpy").load

datadir = "/Users/aza/Projects/TTS/data/LJSpeech-1.1"
savepath = joinpath(datadir, "melspectrograms.jld2")
npy_mels_pectrograms_dir = joinpath(datadir, "melspectrograms")

melspectrograms = Dict{String,Matrix{Float32}}()
for (root, _, files) ∈ walkdir(npy_mels_pectrograms_dir)
   for file ∈ files
      key = first(splitext(file))
      npyfilepath = joinpath(root, file)
      value = load(npyfilepath)
      melspectrograms[key] = value
   end
end

@save savepath melspectrograms
