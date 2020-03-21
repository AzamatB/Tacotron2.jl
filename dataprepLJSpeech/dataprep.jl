using CSV
using DataFrames
using FileIO
using Base.Iterators
using Random
using Flux: onehotbatch

function build_batches(metadatapath::AbstractString, melspectrogramspath::AbstractString, batchsize::Integer; eos='~', pad='_', kwargs...)
   dataset, alphabet = build_dataset(metadatapath, melspectrogramspath; eos=eos, pad=pad)
   batches = build_batches!(dataset, alphabet, batchsize, alphabet[eos], alphabet[pad], kwargs...)
   return batches, alphabet
end

build_batches(dataset::AbstractVector{<:Tuple{AbstractString,DenseMatrix{Float32}}}, alphabet::AbstractDict{Char,<:Integer}, batchsize::Integer, eosindex::Integer, padindex::Integer; kwargs...) =
   build_batches!(copy(dataset), alphabet, batchsize, eosindex, padindex; kwargs...)

function build_batches!(dataset::AbstractVector{<:Tuple{AbstractString,DenseMatrix{Float32}}},
                        alphabet::AbstractDict{Char,<:Integer},
                        batchsize::Integer,
                        eosindex::Integer,
                        padindex::Integer;
                        sorted=false, reorder=true)
   sorted || sort!(dataset; by=length∘first)
   batches = map(partition(dataset, batchsize)) do batch
      pad_batch(batch, alphabet, eosindex, padindex; sorted=true, reorder=reorder)
   end
   reorder && shuffle!(batches)
   return batches
end

function pad_batch(batch::AbstractVector{<:Tuple{String,DenseMatrix{Float32}}}, alphabet::AbstractDict{Char,<:Integer}, eosindex::Integer, padindex::Integer; sorted=false, reorder=true)
   batchsize = length(batch)
   maxlength_x = 1 + (sorted ? (length∘first∘last)(batch) : maximum(length∘first, batch))
   reorder && shuffle!(batch)

   mellengths = map(xy -> size(last(xy), 1), batch)
   maxlength_y = maximum(mellengths)
   nchannels_y = size(last(first(batch)), 2)

   stoptarget = Float32.(permutedims(onehotbatch(mellengths, 1:maxlength_y)))
   textindices = fill(padindex, maxlength_x, batchsize)
   melspectrograms = zeros(Float32, maxlength_y, nchannels_y, batchsize)

   for (n, (text, melspectrogram)) ∈ enumerate(batch)
      melspectrograms[axes(melspectrogram,1),:,n] = melspectrogram
      for (i, char) ∈ enumerate(text)
         textindices[i,n] = alphabet[char]
      end
      textindices[length(text)+1,n] = eosindex
   end
   (time, nmelfeatures, batchsize) = size(melspectrograms)
   melspectrograms⁴ = reshape(melspectrograms, (time, 1, nmelfeatures, batchsize))
   return textindices, gpu(melspectrograms⁴), gpu(stoptarget)
end

function build_dataset(metadatapath::AbstractString, melspectrogramspath::AbstractString; eos='~', pad='_')
   df = DataFrame(CSV.File(metadatapath; delim='|', header=[:file_name, :text, :text_normalized], quotechar='\\', escapechar='\\'))
   ## sanity check that dataframe was parsed correctly
   # lines = readlines(metadatapath)
   # all(i -> collect(df[i,:]) == split(lines[i], '|'), eachindex(lines))
   select!(df, Not(:text))
   melspectrograms = load(melspectrogramspath, "melspectrograms")
   dataset = map(eachrow(df)) do row
      row.text_normalized, melspectrograms[row.file_name]
   end
   alphabet = build_alphabet(df.text_normalized; eos='~', pad='_')
   return dataset, alphabet
end

function build_alphabet(texts::AbstractVector{<:AbstractString}; eos::Char, pad::Char)
   alphabet = sort(unique(reduce(*, texts)))
   # padding character will be used a lot, so make its index to be 2 and make eos char index to be 1
   errormessage(char) = char -> "Cannot add the $char character to the alphabet because it already contains it"
   (eos ∈ alphabet) && error(errormessage("eos"))
   (pad ∈ alphabet) && error(errormessage("pad"))
   pushfirst!(alphabet, eos, pad)
   return Dict(v => k for (k, v) ∈ enumerate(alphabet))
end
