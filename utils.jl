function Flux.dropout(x, p)
   q = 1 - p
   y = rand!(similar(x))
   y .= Flux._dropout_kernel.(y, p, q)
   x .* y
end

Flux.@adjoint function Flux.dropout(x, p)
   q = 1 - p
   y = rand!(similar(x))
   y .= Flux._dropout_kernel.(y, p, q)
   return x .* y, Δ -> (Δ .* y, nothing)
end
