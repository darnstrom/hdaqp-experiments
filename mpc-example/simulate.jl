using HDF5 # To save problems to run in C++ (since lexls & NIPM are not available in Julia) 

function simulate(mpc,obstacles,Ac,Bc;nominal_constraints = mpc.constraints, 
        T = 20, simfactor=4, x0 = zeros(mpc.model.nx), fname=nothing)
    dt = mpc.model.Ts/simfactor
    x,u = x0,zeros(mpc.model.nu) 
    Xs,Us = [],[]
    ts = 0:dt:T
    tdaqp = Float64[]

    fid = isnothing(fname) ? nothing : h5open(fname, "w")

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

            # Setup MPC 
            setup!(mpc)

            # Save problems for test in C++ 
            isnothing(fid) || save_problem(fid,t,Int((k-1)/simfactor),mpc,x,u)

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
    isnothing(fid) || close(fid)
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

function save_problem(file,t,k,mpc,x,uprev)
    mpQP = LinearMPC.mpc2mpqp(mpc);

    θ = LinearMPC.form_parameter(mpc,x,nothing,nothing,uprev,nothing)
    Wth = mpc.mpQP.W*θ
    Fth = mpc.mpQP.f_theta*θ

    ms = length(mpQP.bu) - size(mpQP.A,1)

    # Handle simple bounds
    A = ms > 0 ? [I(length(mpQP.f))[1:ms,:]; mpQP.A]  : mpQP.A 

    C = cholesky(mpQP.H)
    M = A / C  
    Mnorms = [norm(a) for a in eachrow(M)]
    Mn = M ./ Mnorms 
    dTH =A*(mpc.mpQP.H\(mpc.mpQP.f+Fth))
    du = mpQP.bu+Wth+dTH
    dl = mpQP.bl+Wth+dTH
    # Write to file
    g = create_group(file, "problem"*string(k))
    g["matrix"] = copy(Mn') # col -> row major
    g["upper"] = du./Mnorms
    g["lower"] = dl./Mnorms
    g["break_points"] = mpQP.break_points
    g["t"] = t
end
