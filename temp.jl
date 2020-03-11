@ein Eᵢs[t,b] := ϕsᵢ[d,b] * ψHs[d,t,b]

@ein context[d,b] := αᵢs[t,b] * Hs′[t,b,d]

@ein context1[d,b] := Hs[d,t,b] * αᵢs[t,b]


function Base.permutedims(B::StridedArray, perm)
    @warn "permutedims is called"
    @show perm
    @show size(B)
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P, B, perm)
end
