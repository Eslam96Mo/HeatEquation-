using CUDA ,LinearAlgebra
using CairoMakie
using DifferentialEquations 
using DiffEqGPU 

#= using the DifferentialEquations.jl packages
   to calculate the heat transfer 
   on 2-Dimesional plate, By appling 
   Dirichlet Boundary Condition
=#

α  = 0.01                                                  # Diffusivity
L  = 0.1                                                   # Length
W  = 0.1                                                   # Width
Nx = 66                                                    # No.of steps in x-axis
Ny = 66                                                    # No.of steps in y-axis
Δx = L/(Nx-1)                                               # x-grid spacing
Δy = W/(Ny-1)                                               # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))               # Largest stable time step
p = (α,Δx,Δy,100.0,0.0,100.0,0.0, Nx,Ny)                      # Parameters      
tspan = (0.0, 0.5) # (0.0:1.0)
xspan = 0 : Δx : L

temp_left   = 10.0                                        # Boundary condition
temp_right  = 0.0
temp_bottom = 10.0
temp_top    = 0.0

function diffuse!(du,u,p,t)

#α,Δx,Δy,temp_left,temp_right,temp_bottom,temp_top,Nx, Ny= p
 
 dij  = view(u, 2:Nx-1, 2:Ny-1)
 di1j = view(u, 1:Nx-2, 2:Ny-1)
 dij1 = view(u, 2:Nx-1, 1:Ny-2)
 di2j = view(u, 3:Nx  , 2:Ny-1)
 dij2 = view(u, 2:Ny-1, 3:Nx  )  

 @inbounds for j in 1:Ny, i in 1:Nx
 @. du[i,j] = α*(di1j - 2*dij + di2j)/Δx^2
              α*(dij1 - 2*dij + dij2)/Δy^2
 end

#=
    @inbounds for j in 2:Ny-1, i in 2:Nx-1
      du[i,j] += α*(u[i-1,j] - 2*u[i,j] + u[i+1,j])/Δx^2
              +  α*(u[i,j-1] - 2*u[i,j] + u[i,j+1])/Δy^2
    end

    @inbounds for j in 3:Ny-2, i in 2:Nx-1
      du[i,j]  += α*(u[i-1,j] - 2*u[i,j] + u[i+1,j])/Δx^2
               +  α*(u[i,j-1] - 2*u[i,j] + u[i,j+1])/Δy^2
    end

    @inbounds for j in 2:Ny-1, i in 1:Nx-2
      du[i,j]  += α*(u[i-1,j] - 2*u[i,j] + u[i+1,j])/Δx^2
               +  α*(u[i,j-1] - 2*u[i,j] + u[i,j+1])/Δy^2
   end

   @inbounds for j in 3:Ny-1, i in 2:Nx-1
      du[i,j] += α*(u[i-1,j] - 2*u[i,j] - u[i+1,j])/Δx^2
              +  α*(u[i,j-1] - 2*u[i,j] + u[i,j+1])/Δy^2
   end

   @inbounds for j in 2:Ny-1, i in 3:Nx
      du[i,j] +=  α*(u[i-1,j] - 2*u[i,j] + u[i+1,j])/Δx^2
               +  α*(u[i,j-1] - 2*u[i,j] + u[i,j+1])/Δy^2                                                                              # Stencil Computations
   end
=#
#=
   # @. dij_x = (di1j - 2 * dij + di2j)/Δx^2
   # @. dij_y = (dij1 - 2 * dij + dij2)/Δy^2)
   # @. dij = dij + α * Δt * ( dij_x + dij_y)
=#

   @inbounds begin   
      du[1, :]    .= 0 
      du[Nx, :]   .= 0 
      du[:, 1]    .= 0 
      du[:, Ny]   .= 0                              # update boundary condition (Dirichlet BCs) 
     end
end

# α,Δx,Δy=p

#domain =Array(Tridiagonal([1.0 for i in 1:Nx-1],[-2.0 for i in 1:Ny],[1.0 for i in 1:Nx-1]))
#domain[1,2] = 2.0
#domain[Nx, Ny-1] = 2.0
#domain_GPU = CuAArray(convert(Array{Float32}, domain)) 
domain_GPU = CUDA.fill(0.0f0, Nx,Ny)
domain_GPU[16:32, 16:32] .= 5

@inbounds begin   
   domain_GPU[1, :]    .= temp_left 
   domain_GPU[Nx, :]    .= temp_right 
   domain_GPU[:, 1]    .= temp_bottom 
   domain_GPU[:, Ny]    .= temp_top                              # update boundary condition (Dirichlet BCs) 
  end

#prob = ODEProblem(diffuse!,domain_GPU,tspan)
#prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
#monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)

#sol = solve(prob,Tsit5(),save_everystep=false, saveat=0.1)                              # using Runge-Kutta method 
#sol = solve(prob,Eural(),save_everystep=false)                             # using Eural method  
#sol = solve(prob,CVODE_BDF(linear_solver= :GMRES),save_everystep=false)    # using the default dense Jacobian  CVODE_BDF       
                                                                            # CVODE_BDF allows us to use a sparse Newton-Krylov solver by setting linear_solver = :GMRES

du1 = similar(domain_GPU)                                                                            
u1 = copy(domain_GPU)
t = 0.0
#1. Euler
#=
diffuse!(du1,domain_GPU,0,0)
du1_cpu = Array(convert(Array{Float32}, du1)) 
u1 = u1 + Δt*du1
u1_cpu = Array(convert(Array{Float32}, u1))

using Plots
heatmap(u1_cpu)

diffuse!(du1,u1,0,0)
u1 = u1 + Δt*du1
u1_cpu = Array(convert(Array{Float32}, u1))
heatmap(u1_cpu)
=#

# u1 = u1 + Δt*du

for i = 1 : 66
   diffuse!(du1,u1,0,t)
   u1 = u1 + Δt*du1
   t = t + Δt 
end

u1_cpu = Array(convert(Array{Float32}, u1)) 
heatmap(u1_cpu)


using Plots
gr()
plot(sol) # Plots the solution
                                                                            
#display(heatmap(domain_GPU))
plot(xspan, sol.u[end], xlabel = "Position x", ylabel="Temperature", title="Temperature at the final time", legend=false)