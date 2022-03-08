using CairoMakie

α  = 0.01                                                      # Diffusivity
L  = 0.1                                                       # Length
W  = 0.1                                                       # Width
Nx = 66                                                        # No.of steps
Ny = 66                                                        # No.of steps
Δx = L/(Nx-1)                                                   # x-grid spacing
Δy = W/(Ny-1)                                                   # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))                   # Largest stable time step

temp_left   =100.0                                              # Boundary condition
temp_right  = 0
temp_bottom = 100
temp_top    = 0

function diffusion2d_x!(dx,x,Nx, Ny, Δx) # in-place
    
    for iy in 1 : Ny
        for ix in 2 : Nx-1
            i = (iy-1)*Nx + ix
            dx[i] =  (x[i-1] - 2*x[i] + x[i+1])/Δx^2
        end
        i1 = (iy-1)*Nx + 1                                       # West direction
        i2 = (iy-1)*Nx + Nx                                      # East direction

        dx[i1] = temp_left 
        dx[i2] = temp_right                                      # update boundary condition (Dirichlet BCs) 
    end 
end

function diffusion2d_y!(dx,x,Nx, Ny, Δy) # in-place
    
    for ix in 1 : Nx
        for  iy in 2 : Ny-1
            i = (iy-1)*Nx + ix
            dx[i] = dx[i] + (x[i-Nx] - 2*x[i] + x[i+Nx])/Δy^2
        end
        i1 = ix                                                 # South direction
        i2 = Nx*(Ny-1) + ix                                     # North direction
 
        dx[i1] = temp_bottom 
        dx[i2] = temp_top                                       # update boundary condition (Dirichlet BCs) 
    end 
end

function heat_eq_2d!(dΘ,Θ,p,t)

    diffusion2d_x!(dΘ,Θ,Nx, Ny, Δx)
    diffusion2d_y!(dΘ,Θ,Nx, Ny, Δy)
    
    dΘ .= α * dΘ
end

temp2d = zeros(Nx, Ny)
temp2d[16:32,16:32] .= 5

temp_vec = reshape(temp2d,Nx*Ny,1)
display(sum(temp_vec))
d_temp = similar(temp_vec)

for i = 1 : 2000 

    heat_eq_2d!(d_temp, temp_vec, 0, 0)

    temp_vec = temp_vec + Δt*d_temp
    temp2d = reshape(temp_vec, Nx, Ny)

    if i % 20 == 0
        display(heatmap(temp2d))
        display(sum(temp_vec))
    end
end