import numpy as np
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

dx_dt = get_dxdt()

dt = 0.1
t_start = 0.0
t_end = 4
x = [2, 0, 0, 0]  # Anfangswerte für phi, dphi, s, ds
#voltage = 2  # Spannungswert
t = 0

t_eval = np.arange(t_start, t_end+dt, dt)
res = solve_ivp(lambda t, x: dx_dt(t, x),[t_start,t_end],x, t_eval = t_eval) #voltage in dynamics definiert

ttrain = res.t
#noise1 = np.random.normal(0,0.1,len(ttrain))
#noise2 = np.random.normal(0,0.1,len(ttrain))

ytrain1 = res.y[0] #+ noise1
ytrain2 = res.y[2] #+ noise2
ytrain = np.concatenate((ytrain1, ytrain2))

import matplotlib.pyplot as plt

# Zustandsvariablen auslesen
times = res.t
phi = res.y[0]
dphi = res.y[1]
s = res.y[2]
ds = res.y[3]

#Plotten

plt.subplot(2,2,1)
plt.plot(times, phi, label='phi')
plt.plot(times, ytrain1, label='phi mit Rauschen')
plt.xlabel('Zeit in s')
plt.ylabel('phi1')

plt.subplot(2,2,2)
plt.plot(times, dphi, label='dphi')
plt.xlabel('Zeit in s')
plt.ylabel('dphi1')

plt.subplot(2,2,3)
plt.plot(times, s, label='s')
plt.plot(times, ytrain2, label='s mit Rauschen')
plt.xlabel('Zeit in s')
plt.ylabel('phi2')

plt.subplot(2,2,4)
plt.plot(times, ds, label='ds')
plt.xlabel('Zeit in s')
plt.ylabel('dphi2')
#plt.show()


#Least Square Parameter Estimation
paramsecht = [0.7, 0.221, 0.5, 0.1]
print('Die echten Werte sind: ', paramsecht)
def system_gleichungen(params, t, x):
    parameter = {
        'g': 9.81,
        'M': params[0],
        'm': params[1],
        'l': params[2],
        'b': params[3],
    }
    resest = solve_ivp(lambda t, x: dx_dt(t, x, parameter), [t_start, t_end], x, t_eval = t_eval)
    return [resest.y[0], resest.y[2]]

def residuals(params, t, y):
    exp = system_gleichungen(params, t, x)
    res1 = ytrain1- exp[0]
    res2 = ytrain2 - exp[1]
    res1 = np.squeeze(res1)
    res2 = np.squeeze(res2)
    return np.concatenate((res1, res2))

paramsinitial = [1, 1, 1, 1]

result = least_squares(residuals, paramsinitial, args=(ttrain, ytrain))
params_est = result.x
print('Die geschätzten Werte sind: ', params_est)
f = abs(params_est-paramsecht)
print('Der betragsmäßige Fehler ist: ', f)