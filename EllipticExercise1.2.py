
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Funcion que realiza una iteracion de Jacobi
def jacobi(V):
    Vn = V.copy()

    Vn[1:-1, 1:-1] = 0.25 * (
        V[1:-1, 2:] + V[1:-1, :-2]
        + V[2:, 1:-1] + V[:-2, 1:-1]
    )

    return Vn


# Funcion para crear la matriz inicial
def crear_matriz():
    V = np.zeros((5, 5))
    V[0, :] = 100.0
    return V


# ------------------------------------------
# PARTE A: exactamente 50 iteraciones
# ------------------------------------------
V = crear_matriz()
cambio_50 = 0.0

for iteracion in range(50):
    Vn = jacobi(V)

    # Cambio maximo entre iteraciones
    cambio_50 = np.max(np.abs(Vn - V))

    V = Vn

print("Matriz despues de 50 iteraciones:")
print(np.round(V, 4))
print(f"Cambio maximo en la iteracion 50: {cambio_50:.8f}")


# ------------------------------------------
# PARTE B: iterar hasta alcanzar tolerancia
# ------------------------------------------
tolerance = 0.01
max_iteraciones = 10000

V2 = crear_matriz()
cambio = float("inf")
iteraciones = 0

while cambio > tolerance and iteraciones < max_iteraciones:
    Vn = jacobi(V2)

    cambio = np.max(np.abs(Vn - V2))

    V2 = Vn
    iteraciones += 1

print("\nMatriz con criterio de tolerancia:")
print(np.round(V2, 4))
print(f"Cambio maximo final: {cambio:.8f}")
print(f"Numero de iteraciones: {iteraciones}")
print(f"Tolerancia: {tolerance}")


# ------------------------------------------
# GRAFICAR LOS RESULTADOS
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(V, cmap="hot", vmin=0, vmax=100)
axes[0].set_title("Jacobi: 50 iteraciones")

im1 = axes[1].imshow(V2, cmap="hot", vmin=0, vmax=100)
axes[1].set_title(f"Jacobi: tolerancia = {tolerance}")

for ax, matriz in zip(axes, [V, V2]):
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")

    for fila in range(5):
        for columna in range(5):
            ax.text(
                columna, fila,
                f"{matriz[fila, columna]:.1f}",
                ha="center", va="center", color="cyan"
            )

cbar_ax = fig.add_axes([0.46, 0.20, 0.025, 0.60])
fig.colorbar(im1, cax=cbar_ax, label="Valor de V")

plt.tight_layout()
plt.show()
