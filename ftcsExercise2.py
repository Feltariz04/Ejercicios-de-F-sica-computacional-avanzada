import numpy as np
import matplotlib.pyplot as plt

# Parametros
L = 0.5 # m
K= 0.12 #cal/(cm * s * °C)
ce = 0.113 # cal/(g*°C)
rho = 7.8 # g/cm**3
C2 = round((K/(ce*rho))*(3600/10000), 3) # m**2 / h


nx = 11
x = np.linspace(0, L, nx)
dx = x[1]-x[0]


def ftcs(dt, tau):
# Para una hora tau = 3600 s y dt = 72
    nt = int(tau/dt) 

    # condiciones de temperatura
    T = np.zeros((nt, nx))

    T[0, :]=100.0 # Initial condition
    T[0,0]= 0   # left boundary
    T[0, -1] = 0 # rigth boundary
# Cumple con la condicion
    r = C2*dt/(dx**2)
    

    for n in range(nt - 1):
        T[n+1, 1:-1] = (T[n, 1:-1] +
                        r * (T[n, 2:]
                             + T[n, :-2]
                             - 2*T[n, 1:-1]
                             )
                        )
        
            #Frontera de la izquierda
        T[n+1, 0] = 0

            #Frontera de la derecha
        T[n+1, -1] = 0

        time = np.linspace(0, tau, nt)
    return time, x, T, r


#Implimentacion del metodo Crank_Nicolson
def crank_nicolson(dt, tau):
    nt = int(round(tau / dt)) + 1
    time = np.arange(nt) * dt

    T = np.zeros((nt, nx))
    T[0, :] = 100.0
    T[0, 0] = 0.0
    T[0, -1] = 0.0

    r = C2 * dt / dx**2
    m = nx - 2  # numero internos de puntos

    # Matriz de temperatura desconocida a un tiempo n+1
    A = np.diag(np.full(m, 1 + r))
    A += np.diag(np.full(m - 1, -r / 2), k=1)
    A += np.diag(np.full(m - 1, -r / 2), k=-1)

    # Matriz de temperaturas a un tiempo  n
    B = np.diag(np.full(m, 1 - r))
    B += np.diag(np.full(m - 1, r / 2), k=1)
    B += np.diag(np.full(m - 1, r / 2), k=-1)

    # Ecuación matricial Au(n+1) = Bu(n) + b(frontera)
    for n in range(nt - 1):
        # Right-hand side
        b = B @ T[n, 1:-1] # @ es un operador que multiplica las matrices

        # Boundary contributions (both boundaries are zero here)
        b[0] += (r / 2) * (T[n, 0] + T[n+1, 0])
        b[-1] += (r / 2) * (T[n, -1] + T[n+1, -1])

        # Solve for interior temperatures
        T[n+1, 1:-1] = np.linalg.solve(A, b) # resuelve el sistema lineal

    return time, x, T, r

time_f, x_f, T_f, r_f = ftcs(dt = 0.02, tau= 1)
time_cn, x_cn, T_cn, r_cn = crank_nicolson(dt = 0.02, tau= 1)

print(f"r de ftsc = {r_f:.3f} m")
print(f"r de crank-Nicolson = {r_cn:.3f} m")
print(f"Temperatura final máxima de ftsc = {T_f[-1].max():.2f} °C")
print(f"Temperatura final máxima de crank_Nicolson = {T_cn[-1].max():.2f} °C")


# =====================================================
# GRAFICAS FTCS Y CRANK-NICOLSON: 1 FILA, 2 COLUMNAS
# =====================================================

instantes = [0, 0.25, 0.50, 0.75, 1.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafica 1: FTCS
for hora in instantes:
    idx = np.argmin(np.abs(time_f - hora))
    axes[0].plot(
        x, T_f[idx], marker="o",
        label=f"t = {time_f[idx]:.2f} h"
    )

axes[0].set_xlabel("Posicion x (m)")
axes[0].set_ylabel("Temperatura (°C)")
axes[0].set_title(f"FTCS (r = {r_f:.3f})")
axes[0].legend()
axes[0].grid(True)

# Grafica 2: Crank-Nicolson
for hora in instantes:
    idx = np.argmin(np.abs(time_cn - hora))
    axes[1].plot(
        x, T_cn[idx], marker="s",
        label=f"t = {time_cn[idx]:.2f} h"
    )

axes[1].set_xlabel("Posicion x (m)")
axes[1].set_ylabel("Temperatura (°C)")
axes[1].set_title(f"Crank-Nicolson (r = {r_cn:.3f})")
axes[1].legend()
axes[1].grid(True)

fig.suptitle("Evolucion de la temperatura en la barra")
plt.tight_layout()
plt.show()