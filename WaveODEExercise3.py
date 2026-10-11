
import numpy as np
import matplotlib.pyplot as plt


def run_wave(L, dx, dt, c, t_final, f_func, g_func):
    """
    Metodo explicito de diferencias finitas para la ecuacion
    de onda 1D: y_tt = c^2 * y_xx.

    Devuelve la posicion x, los tiempos t y la matriz Y.
    """

    # Parametro de estabilidad
    r = c * dt / dx

    # Malla espacial y temporal
    Nx = int(round(L / dx))
    nodes = Nx + 1
    x = np.linspace(0, L, nodes)

    nt = int(round(t_final / dt))
    t = np.linspace(0, nt * dt, nt + 1)

    # Matriz: filas = tiempo, columnas = posicion
    Y = np.zeros((nt + 1, nodes))

    # Condiciones iniciales
    Y[0, :] = f_func(x)
    g = g_func(x)

    # Condiciones de frontera en t = 0
    Y[0, 0] = 0.0
    Y[0, -1] = 0.0

    # Primera fila temporal: segundo orden
    for i in range(1, nodes - 1):
        Y[1, i] = (
            Y[0, i]
            + dt * g[i]
            + 0.5 * r**2 * (
                Y[0, i+1]
                + Y[0, i-1]
                - 2 * Y[0, i]
            )
        )

    # Condiciones de frontera
    Y[1, 0] = 0.0
    Y[1, -1] = 0.0

    # Evolucion temporal
    for j in range(1, nt):
        for i in range(1, nodes - 1):
            Y[j+1, i] = (
                2 * Y[j, i]
                - Y[j-1, i]
                + r**2 * (
                    Y[j, i+1]
                    + Y[j, i-1]
                    - 2 * Y[j, i]
                )
            )

        # Fronteras fijas
        Y[j+1, 0] = 0.0
        Y[j+1, -1] = 0.0

    print(f"dx = {dx:.4f}")
    print(f"dt = {dt:.4f}")
    print(f"r = {r:.4f}")

    if r <= 1:
        print("El parametro cumple el criterio de estabilidad r <= 1.")
    else:
        print("Advertencia: r > 1; el esquema explicito es inestable.")

    return x, t, Y


#condiciones iniciales
def f_func(x):
    return np.sin(np.pi * x) + np.sin(2 * np.pi * x)

def g_func(x):
    return np.zeros_like(x)


# Parametros
L = 1.0
dx = 0.04
dt = 0.02
c = 2.0
t_final = 0.5

# Resolver la ecuacion de onda
x, t, Y = run_wave(L, dx, dt, c, t_final, f_func, g_func)

# Graficar 
plt.figure(figsize=(9, 5))

for tiempo in [0.0, 0.125, 0.25, 0.375, 0.5]:
    j = int(round(tiempo / dt))
    plt.plot(x, Y[j, :], label=f"t = {t[j]:.3f} s")

plt.xlabel("Posicion x [m]")
plt.ylabel("Desplazamiento y [cm]")
plt.title("Ecuacion de onda 1D - Metodo explicito")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
