using CSV
using DataFrames
using FileIO
using Base.Iterators
using Random

function batch_dataset(metadatapath::AbstractString, melspectrogramspath::AbstractString, batch_size::Integer; eos='~', pad='_', kwargs...)
   dataset, alphabet = build_dataset(metadatapath, melspectrogramspath; eos=eos, pad=pad)
   batches = batch_dataset!(dataset, batch_size; eos=eos, pad=pad, kwargs...)
   return batches, alphabet
end

batch_dataset(dataset::Vector{<:Tuple{String,DenseMatrix{Float32}}}; kwargs...) = batch_dataset!(copy(dataset); kwargs...)

function batch_dataset!(dataset::Vector{<:Tuple{String,DenseMatrix{Float32}}}, batch_size::Integer; eos='~', pad='_', sorted=false, reorder=true)
   sorted || sort!(dataset; by=length∘first)
   batches = map(partition(dataset, batch_size)) do batch
      pad_batch(batch, eos, pad; sorted=true, reorder=reorder)
   end
   reorder && shuffle!(batches)
   return batches
end

function pad_batch(batch::AbstractVector{<:Tuple{String,DenseMatrix{Float32}}}, eos::Char, pad::Char; sorted=false, reorder=true)
   batch_size = length(batch)
   len_x = sorted ? (length∘first∘last)(batch) : maximum(length∘first, batch)
   len_y = maximum(xy -> size(last(xy), 2), batch)
   nchannels_y = size(last(first(batch)), 1)

   reorder && (batch = shuffle(batch))

   texts = map(batch) do (text, _)
      collect(text * eos * pad^(len_x - length(text)))
   end

   melspectrograms = zeros(Float32, len_y, nchannels_y, batch_size)
   for (k, (_, melspectrogram)) ∈ enumerate(batch)
      len = size(melspectrogram, 2)
      melspectrograms[1:len,:,k] = melspectrogram'
   end
   return texts, melspectrograms
end

function build_dataset(metadatapath::AbstractString, melspectrogramspath::AbstractString; eos='~', pad='_')
   df = DataFrame(CSV.File(metadatapath; delim='|', header=[:file_name, :text, :text_normalized], quotechar='\\', escapechar='\\'))
   ## sanity check that dataframe was parsed correctly
   # lines = open(metadatapath) do file
   #     readlines(file)
   # end
   # all(i -> collect(df[i,:]) == split(lines[i], '|'), eachindex(lines))
   select!(df, Not(:text))
   alphabet = build_alphabet(df.text_normalized; eos='~', pad='_')
   melspectrograms = load(melspectrogramspath, "melspectrograms")

   dataset = map(eachrow(df)) do row
      row.text_normalized, melspectrograms[row.file_name]
   end
   return dataset, alphabet
end

function build_alphabet(texts::AbstractVector{<:AbstractString}; eos::Char, pad::Char)
   alphabet = sort(unique(reduce(*, texts)))
   push!(alphabet, eos, pad)
   return alphabet
end
