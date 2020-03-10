text = "the input x must be a vector of length in, or a batch of vectors"
alphabet = (c for c ∈ unique(text))
m = CharEmbedding(alphabet)
m(text)
m(['c','e'])
