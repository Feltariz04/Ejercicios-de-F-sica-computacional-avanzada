
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
n = 40
x = np.linspace(0, 1, n)
y = np.linspace(1, 0, n)  # Fila 0 = parte superior
X, Y = np.meshgrid(x, y)

# Crear las condiciones de frontera
def make_bc_constant(n, top=100.0):
    V = np.zeros((n, n))
    bc_mask = np.zeros((n, n), dtype=bool)
    bc_vals = np.zeros((n, n))

    # Frontera superior
    V[0, :] = top
    bc_vals[0, :] = top

    # Los lados laterales y el inferior están a tierra
    V[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0

    # Marcar todas las fronteras
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    # Esquinas superiores
    bc_vals[0, 0] = 100.0
    bc_vals[0, -1] = 100.0

    V[bc_mask] = bc_vals[bc_mask]

    return V, bc_mask, bc_vals


# Método de Jacobi
def jacobi(V, bc_mask, bc_vals,
           max_iter=100000, tol=1e-8):

    for iteration in range(max_iter):
        V_old = V.copy()

        V_new = V_old.copy()
        V_new[1:-1, 1:-1] = 0.25 * (
            V_old[:-2, 1:-1]
            + V_old[2:, 1:-1]
            + V_old[1:-1, :-2]
            + V_old[1:-1, 2:]
        )

        V = V_new
        V[bc_mask] = bc_vals[bc_mask]

        cambio = np.max(np.abs(V - V_old))

        if cambio < tol:
            return V, iteration + 1, cambio

    return V, max_iter, cambio


# Método de Gauss-Seidel
def gauss_seidel(V, bc_mask, bc_vals,
                 max_iter=100000, tol=1e-8):

    nx, ny = V.shape

    for iteration in range(max_iter):
        V_old = V.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if not bc_mask[i, j]:
                    V[i, j] = 0.25 * (
                        V[i-1, j] + V[i+1, j]
                        + V[i, j-1] + V[i, j+1]
                    )

        V[bc_mask] = bc_vals[bc_mask]

        cambio = np.max(np.abs(V - V_old))

        if cambio < tol:
            return V, iteration + 1, cambio

    return V, max_iter, cambio


# Resolver con ambos métodos
V0, bc_mask, bc_vals = make_bc_constant(n, top=100.0)

V_jacobi, iter_j, cambio_j = jacobi(
    V0.copy(), bc_mask, bc_vals
)

V_gs, iter_gs, cambio_gs = gauss_seidel(
    V0.copy(), bc_mask, bc_vals
)

print("Jacobi:")
print("Iteraciones:", iter_j)
print("Cambio final:", cambio_j)

print("\nGauss-Seidel:")
print("Iteraciones:", iter_gs)
print("Cambio final:", cambio_gs)

# Diferencia entre las dos soluciones
diferencia = np.max(np.abs(V_jacobi - V_gs))
print("\nDiferencia máxima entre soluciones:", diferencia)


# Gráficas de curvas equipotenciales
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

niveles = np.linspace(0, 100, 11)

cont1 = axes[0].contour(
    X, Y, V_jacobi, levels=niveles
)
axes[0].clabel(cont1, inline=True, fontsize=8)
axes[0].set_title("Curvas equipotenciales: Jacobi")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

cont2 = axes[1].contour(
    X, Y, V_gs, levels=niveles
)
axes[1].clabel(cont2, inline=True, fontsize=8)
axes[1].set_title("Curvas equipotenciales: Gauss-Seidel")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.show()


# Mapa de potencial con Gauss-Seidel
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, V_gs, levels=20)
plt.colorbar(label="Potencial (V)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Distribución del potencial electrostático")
plt.tight_layout()
plt.show()
