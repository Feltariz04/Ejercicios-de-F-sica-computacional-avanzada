
import numpy as np
import matplotlib.pyplot as plt




# Funcion para crear la matriz inicial
def make_bc(n, x):
    V = np.zeros((n,n))
    
    top = np.sin(np.pi * x)

    bc_mask = np.zeros_like(V, dtype=bool)
    bc_vals = np.zeros_like(V)

    #Frontera superior
    V[0, :] = top
    bc_vals[0, :] = top

    # Resto de las fronteras: potencial cero
    V[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0

    # Marcar las fronteras
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    V[bc_mask] = bc_vals[bc_mask]
    
    return V, bc_mask, bc_vals



# funcion para realizar una iteracion de Gauss-Seidel
def Gauss_Seidel(V, bc_mask, bc_vals, max_iter=10000, tol=1e-10):
  
    nx, ny = V.shape

    for iteration in range(max_iter):
        V_old = V.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                if not bc_mask[i, j]:
                    V[i, j] = 0.25 * (
                        V[i-1, j] +
                        V[i+1, j] +
                        V[i, j-1] +
                        V[i, j+1]
                    )

        # Restaurar las condiciones de frontera
        V[bc_mask] = bc_vals[bc_mask]

        # Comprobar convergencia
        cambio = np.max(np.abs(V - V_old))


    return V, cambio

# Funcion de la solucion analitica
def analytical_solution(x, y):
    return (np.sin(np.pi * x) * np.sinh(np.pi * y)) / (np.sinh(np.pi)) 


tamanos = [10, 20, 40, 80, 120]

errores_maximos = []
pasos_malla = []

print("EJERCICIO 2")
print(" n       Error máximo")
print("---------------------")

for n in tamanos:
    x = np.linspace(0, 1, n)
    y = np.linspace(1, 0, n)  # Fila 0 = parte superior

    # meshgrid mantiene la orientación de la matriz
    X, Y = np.meshgrid(x, y)

    V, bc_mask, bc_vals = make_bc(n, x)

    V_num, cambio = Gauss_Seidel(
        V, bc_mask, bc_vals
    )

    V_exacta = analytical_solution(X, Y)

    error = np.abs(V_num - V_exacta)
    error_max = np.max(error)

    errores_maximos.append(error_max)
    pasos_malla.append(1 / (n - 1))

    print(f"{n:3d}    {error_max:.8e}")


# Gráfica del error máximo frente al tamaño de la malla
plt.figure(figsize=(8, 5))
plt.loglog(pasos_malla, errores_maximos, "o-")
plt.xlabel("Paso de malla h = 1/(n-1)")
plt.ylabel("Error máximo")
plt.title("Gauss-Seidel: convergencia con el refinamiento")
plt.grid(True, which="both")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

