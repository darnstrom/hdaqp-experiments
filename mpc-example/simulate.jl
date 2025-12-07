function simulate(mpc,obstacles,Ac,Bc;nominal_constraints = mpc.constraints, 
        T = 20, simfactor=4, x0 = zeros(mpc.model.nx))
    dt = mpc.model.Ts/simfactor
    x,u = x0,zeros(mpc.model.nu) 
    Xs,Us = [],[]
    ts = 0:dt:T
    tdaqp = Float64[]
    for (k,t) in enumerate(ts) 
        # Ego
        #println("k:$k | t:$t")
        if((k-1)%simfactor == 0)

            # Update constraints
            mpc.constraints =  deepcopy(nominal_constraints)
            # Add obstacles
            for obs in obstacles
                create_obstacle_constraints!(mpc,obs,t,x;plt=nothing)
            end

            # Create mpQP 
            mpQP = LinearMPC.mpc2mpqp(mpc);

            # Solve
            setup!(mpc)
            tsolve = @elapsed u = compute_control(mpc,x;uprev=u)
            push!(tdaqp,tsolve)
        else
            push!(tdaqp,NaN)
        end
        push!(Xs,x)
        push!(Us,u)
        x += dt*(Ac*x+Bc*u);
    end
    return Xs,Us,ts,tdaqp
end

function plot_scenario(ts,Xs,obstacles,W_line,tdaqp)
    ## Visualize
    s = [x[1] for x in Xs];
    phi = [x[2] for x in Xs];
    beta= [x[3] for x in Xs];
    omega = [x[4] for x in Xs];
    plt = plot()
    hline!(W_line/2*[1;-1],label="Road")
    for obs in obstacles
        obs_plot!(obs)
    end
    plot!(ts,s,label="Lateral distance")

    plt_time = scatter(ts,tdaqp,yaxis=:log,ylabel="Execution time [s]",xlabel="Time [s]")
    hline!([0.01],label="Sample time")

    display(plot(plt,plt_time,layout=(2,1)))

    println("Press any key to continue")
    readline()
    closeall()
end
