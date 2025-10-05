# %%
from casadi import *
x = MX.sym("x")
print(jacobian(sin(x),x))

# %%
x = SX.sym('x'); y = SX.sym('y')
qp = {'x':vertcat(x,y), 'f':x**2+y**2, 'g':x+y-10}
S = qpsol('S', 'qpoases', qp)
print(S)