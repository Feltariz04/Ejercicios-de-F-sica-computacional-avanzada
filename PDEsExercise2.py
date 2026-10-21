
import numpy as np
import matplotlib.pyplot as plt

n = 100
esp0 = 8.855e-12


x = np.linspace(1, 5, n)
y = np.linspace(1, 5, n)
h = x[1] - x[0]

# Funcion para crear la matriz inicial
def make_bc(n):
    V = np.zeros((n,n))
    bc_mask = np.zeros_like(V, dtype=bool)
    bc_vals = np.zeros_like(V)

    # bordes a potencial cero
    V[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0
    V[0, :] = 0.0

    V[bc_mask] = bc_vals[bc_mask]
    
    return V, bc_mask, bc_vals


# Densidad de carga superficial
rho = np.zeros((n,n))
rho0 = 1 # C/m**2

rho[60:79, 20:39] = -rho0
rho[20:39, 60:79] = rho0


# funcion para realizar una iteracion de Gauss-Seidel
def Gauss_Seidel(V, bc_mask, bc_vals, rho, h, esp0):
  
    nx, ny = V.shape


    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            if not bc_mask[i, j]:
                    V[i, j] = 0.25 * (
                        V[i-1, j] +
                        V[i+1, j] +
                        V[i, j-1] +
                        V[i, j+1]
                    ) + 0.25*(rho[i,j]*(h**2)/esp0)

        # Restaurar las condiciones de frontera
        V[bc_mask] = bc_vals[bc_mask]



    return V

 #Solucion del problema

V, bc_mask , bc_vals = make_bc(n)

# Inicio del metodo
for _ in range(10000):
    V_old = V.copy()

    Vg = Gauss_Seidel(V, bc_mask, bc_vals, rho, h, esp0)

    error = np.max(np.abs(V- V_old))

    if error < 1e-7:
        break

print("Iteraciones:", _ + 1)
print("Error máximo:", error)
print("Potencial máximo:", V.max(), "V")
print("Potencial mínimo:", V.min(), "V")

# Representación del potencial

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(7, 6))
plt.imshow(
    V,
    origin="upper",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="coolwarm",
    aspect="equal"
)

plt.colorbar(label="Potencial eléctrico (V)")
plt.title("Distribución del potencial eléctrico")
plt.tight_layout()
plt.show()