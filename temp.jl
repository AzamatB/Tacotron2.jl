
einsum(rule, code, xs, size_dict)

using LinearAlgebra: BlasFloat
function OMEinsum.einsum(::OMEinsum.BatchedContract, ::EinCode{ixs,iy}, xs::NTuple{<:Any, AbstractArray{<:BlasFloat}}, size_dict) where {ixs, iy}
    ixs2, xs2 = OMEinsum._preprocess_dupindices(ixs[2], xs[2])
    ixs1, xs1 = OMEinsum._preprocess_dupindices(ixs[1], xs[1])
    OMEinsum.batched_contract(ixs1, xs1, ixs2, xs2, iy)
end

permutedims(x, (2,3,1))

function Base.permutedims(B::StridedArray, perm)
    @warn "permutedims is called"
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P, B, perm)
end


@code_lowered(@ein Uf[a,t,b] := U[a,c] * fs[t,c,b])
##

@macroexpand(@ein Uf[a,t,b] := U[a,c] * fs[t,c,b])

ixs = ((1,2), (3,2,4))
iy = (1,3,4)
code = EinCode{ixs, iy}()
xs = (U, fs)

rule = OMEinsum.match_rule(ixs, iy)
rule = OMEinsum.match_rule(((3,2,4),), (3,2,4))

@edit einsum(code, xs)
@edit einsum(code, xs, nothing)

size_dict = OMEinsum.get_size_dict(ixs, xs)

@edit einsum(code, xs, size_dict)


@edit einsum(rule, code, xs, size_dict)

OMEinsum._preprocess_dupindices
