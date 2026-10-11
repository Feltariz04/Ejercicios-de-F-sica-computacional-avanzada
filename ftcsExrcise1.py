import numpy as np
import matplotlib.pyplot as plt

#parametros
L = 0.5
nx = 11
x = np.linspace(0, L, nx)
dx = x[1] - x[0]
c2=1e-4

def ftcs(dt, tau):
  
    Nt = int(tau/dt) + 1

    #Temperature array: time vs space
    T = np.zeros((Nt, nx))
    T[0, :]=100.0 # Initial condition
    T[0,0]= 0   # left boundary
    T[0, -1] = T[0, -2] # rigth boundary

    r = c2*dt/(dx**2)
    print(f'r = {r:.3f}')

    #Time steppeing

    for n in range(Nt - 1):
           T[n+1, 1:-1] = (
            T[n, 1:-1]
            + r * (
                T[n, 2:]
                + T[n, :-2]
                - 2*T[n, 1:-1]))

            #Frontera de la izquierda
           T[n+1, 0] = 0

            #Frontera de la derecha
           T[n+1, -1] = T[n+1,-2]
    return x, T, r


# Stable Run and Unstable Run
x, Tstable, r_stable = ftcs(dt = 12, tau = 120)
x, Tunstable, r_unstable = ftcs(dt=20, tau=100)


print(f"dx = {dx:.3f} m")
print(f"r estable = {r_stable:.2f}")
print(f"r inestable = {r_unstable:.2f}")

#Graficas
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

time = [0, 24, 48, 72, 120]
time2 = [0, 20 , 40, 60, 100]

for t in time:
      #Stable
      n = round(t / 12)
      ax[0].plot(x, Tstable[n,:], label = f't = {t:.0f}')

for t in time2:
      #Unstable
      n_new = round(t / 20)
      ax[1].plot(x, Tunstable[n_new,:], label = f't = {t:.0f}')


ax[0].set_xlabel('Position x (m)')
ax[0].set_ylabel('Temperature (°C)')
ax[0].set_title(f'FTCS stable run (r = {r_stable:.2f})')
ax[0].legend()
ax[0].grid(True)

ax[1].set_xlabel('Position x (m)')
ax[1].set_ylabel('Temperature (°C)')
ax[1].set_title(f'FTCS unstable demonstration (r = {r_unstable:.2f})')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
