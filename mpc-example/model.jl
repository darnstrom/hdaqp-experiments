# Constants
g = 9.81; # m/s^2
m = 2041; # kg
Izz = 4964; # kg-m^2
lf, lr = 1.56, 1.64;
Ca = 246994
W_line = 3; # m
Vx = 30 
## State space model \dot{x} = Ac x + Bc u, y=Cx
Ac = [0 Vx Vx 0; 
      0 0 0 1;
      0 0 -2*Ca/(m*Vx) Ca*(lr-lf)/(m*Vx^2)-1;
      0 0 Ca*(lr-lf)/Izz -Ca*(lr^2+lf^2)/(Izz*Vx)] 

Bc = [0; 0; Ca/(m*Vx); Ca*lf/Izz;;];

C = [1.0 0 0 0;];
