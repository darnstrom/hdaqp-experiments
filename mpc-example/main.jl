## Init
using LinearMPC
using LinearAlgebra
using Plots
using Combinatorics
using DelimitedFiles
include("obstacles.jl")
include("simulate.jl")
include("model.jl")

## Setup MPC 
Ts = 0.01;
mpc = LinearMPC.MPC(Ac,Bc,Ts;C,Np=30,Nc=30);

LinearMPC.set_objective!(mpc;Q=[1e3],R=[0],Rr=[1e-1])
LinearMPC.set_bounds!(mpc;umin=[-pi/6;],umax=[pi/6;])

Cy_hard =  [0 0 -1 -lf/Vx;  # alpha_f,lim
            0 0 -1 lr/Vx];  # alpha_r,lim

Cylim_hard =  [pi/22.5;  # delta
               pi/22.5]; # alpha_f_lim

Cy_soft =  [1.0 0 0 0;]   # e_y
Cylim_soft = [W_line/2;] # e_y

LinearMPC.add_constraint!(mpc,Ax=Cy_hard, Au = [1.0;0;;], lb = -Cylim_hard,ub = Cylim_hard,prio=1)
LinearMPC.add_constraint!(mpc,Ax=Cy_soft,lb = -Cylim_soft,ub = Cylim_soft, prio=2)

#mpc.settings.reference_tracking=false;

## Simulate
isdir("result") || mkdir("result")
prios = collect(permutations([3,4,5]))
for prio in prios
    println("prio: $prio")

    ob1 = Obstacle(-W_line,0.5,5,15,0,prio[1]) # Blue 
    ob2 = Obstacle(-0.1,W_line,7,13,0,prio[2]) # Red 
    ob3 = Obstacle(-W_line,0.4,9,11,0,prio[3]) # Green
    obstacles = [ob1,ob2,ob3]

    Xs,Us,ts,tdaqp = simulate(mpc,obstacles,Ac,Bc)
    plot_scenario(ts,Xs,obstacles,W_line,tdaqp)

    αs = [Cy_hard*Xs[i]+[1;0;;]*Us[i] for i in 1:length(ts)]
    down_sample = 5
    open("result/scenario"*string(prio[1])*string(prio[2])*string(prio[3])*".dat"; write=true) do f
        write(f, "t delta s psi beta omega alphaf alphar tdaqp \n")
        writedlm(f,[collect(ts) [u[1] for u in Us] mapreduce(permutedims,vcat,Xs) mapreduce(permutedims,vcat,αs) tdaqp][1:down_sample:end,:])
    end
end
