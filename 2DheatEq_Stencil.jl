using CUDA
using CairoMakie

α  = 0.01                                                  # Diffusivity
L  = 0.1                                                   # Length
W  = 0.1                                                   # Width
M  = 66                                                    # No.of steps
Δx = L/(M-1)                                               # x-grid spacing
Δy = W/(M-1)                                               # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))               # Largest stable time step

temp_left   = 100.0                                        # Boundary condition
temp_right  = 0
temp_bottom = 100
temp_top    = 0

function diffuse!(data, a, Δt, Δx, Δy)
    dij  = view(data, 2:M-1, 2:M-1)
    di1j = view(data, 1:M-2, 2:M-1)
    dij1 = view(data, 2:M-1, 1:M-2)
    di2j = view(data, 3:M  , 2:M-1)
    dij2 = view(data, 2:M-1, 3:M  )                        # Stencil Computations
  
    @. dij = dij + α * Δt * (
        (di1j - 2 * dij + di2j)/Δx^2 +
        (dij1 - 2 * dij + dij2)/Δy^2)                      # Apply diffusion
   
    data[1, :]    .= temp_left 
    data[M, :]    .= temp_right 
    data[:, 1]    .= temp_bottom 
    data[:, M]    .= temp_top                              # update boundary condition (Dirichlet BCs) 
    data
end

domain = zeros(M,M)
domain_GPU = CuArray(convert(Array{Float32}, domain)) 
domain_GPU[16:32, 16:32] .= 5


heatmap(domain_GPU)
for i in 1:1000
       diffuse!(domain_GPU, α, Δt, Δx, Δy)
    if i % 20 == 0
        display(heatmap(domain_GPU))
    end
end



    

