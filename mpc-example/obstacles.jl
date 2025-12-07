struct Obstacle
    emin::Cdouble
    emax::Cdouble
    tmin::Cdouble
    tmax::Cdouble
    v::Cdouble
    priority::Cdouble
end

Obstacle(emin,emax) = Obstacle(emin,emax,-Inf,Inf,0,1)


function obs_plot(o)
    Δe = o.v*(o.tmax-o.tmin)
    ts = [o.tmin, o.tmax]
    Emin= o.emin.+[0;Δe]
    Emax = o.emax.+[0;Δe]
    plot(ts,Emin,fillrange = Emax);
end

function obs_plot!(o)
    Δe = o.v*(o.tmax-o.tmin)
    ts = [o.tmin, o.tmax]
    Emin= o.emin.+[0;Δe]
    Emax = o.emax.+[0;Δe]
    plot!(ts,Emin,fillrange = Emax,fillalpha = 0.5,linewidth=0,label="Obstacle");
end

function create_obstacle_constraints!(mpc,obs,t,x;plt = nothing)
    Ts = mpc.model.Ts
    kmin = Int(floor((obs.tmin-t)/Ts))+1
    kmax = Int(floor((obs.tmax-t)/Ts))+1

    if(kmin > mpc.Np && kmax < 0) # No intersection
        return
    end

    kstart = max(kmin,2)
    kstop = min(kmax,mpc.Np)
    ks = kstart:kstop
    #println("ks:$ks")

    emin = obs.emin
    emax = obs.emax
    Δt = t-obs.tmin
    if(Δt >= 0)
        emin += Δt*obs.v
        emax += Δt*obs.v
    end
    es = []
    for k in ks 
        #push!(mpc.constraints.Cy,[1.0 0 0 0]);
        if((emax+emin)/2-x[1] > 0) # Pass under
            lby,uby = -1e30,emin
            push!(es,emin)
        else
            lby,uby = emax,1e30
            push!(es,emax)
        end
        LinearMPC.add_constraint!(mpc, Ax = [1.0 0 0 0], lb = [lby], ub = [uby], ks = k:k, prio = obs.priority)

        emin += Ts*obs.v
        emax += Ts*obs.v
    end
    if(!isnothing(plt))
        plot!(plt,ks,es,xlims=(1,mpc.Np),ylims=(-1,1))
    end
    # create lower
end

