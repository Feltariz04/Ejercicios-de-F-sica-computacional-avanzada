import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Plata de alumnio con flujo de calor
#Constantes del ejercicio
n = 10
k = 0.49 # cal/(s*cm*°C)

#parametros
x = np.linspace(0, 40, n)
y = np.linspace(0, 40, n)

dx = x[1] - x[0]
dy = y[1] - y[0]

def make_bc(n):
    T = np.zeros((n,n))
    bc_mask = np.zeros_like(T, dtype = bool )
    bc_Vals = np.zeros_like(T)

    # Condiciones de las fronteras
   
              
    # Inferior
    bc_mask[0, :] = True
    bc_Vals[0, :] = 0.0 # °C

    # Izquierda
    bc_mask[:, 0] = True
    bc_Vals[:, 0] = 75.0 # °C
    
    # Derecha
    bc_mask[:, -1] = True
    bc_Vals[:, -1] = 50.0 # °C

    # Superior
    bc_mask[-1, :] = True 
    bc_Vals[-1, :] = 100.0  
    

    T[bc_mask] = bc_Vals[bc_mask]
    
    return T, bc_mask, bc_Vals

def Lap(T, bc_mask, bc_Vals):
    nx, ny = T.shape

    for i in range(1, nx-1):
        for j in range(1, ny -1):
            if not bc_mask[i,j]:
                T[i,j] = 0.25 * (
                    T[i+1,j] + T[i-1,j]
                    + T[i,j+1] + T[i,j-1]
                )
# restauro las condiciones iniciales
        T[bc_mask] = bc_Vals[bc_mask]        

    return T



def Fourier(T, k, dx, dy):
    # Calculo del calor con la lez de fourier
   
    # np.gradient devuelve primero la derivada por filas (y)
    # y después la derivada por columnas (x)
    dTdy, dTdx = np.gradient(T, dy, dx)

    qx = -k * dTdx
    qy = -k * dTdy

    return qx, qy

# Solucion del ejercicio
T, bc_mask, bc_Vals = make_bc(int(n))

tol = 1e-7
ite = 1000

for _ in range(int(ite)):
    T_old = T.copy()

    Vg = Lap(T, bc_mask, bc_Vals)

    error = np.max(np.abs(T- T_old))
    
    if error < tol:
        break


print("Iteraciones:", _ + 1)
print("Error máximo:", error)
print("Potencial máximo:", T.max(), "°C")
print("Potencial mínimo:", T.min(), "°C")

# Calcular el flujo térmico
qx, qy = Fourier(T, k, dx, dy)

print("Flujo qx máximo:", qx.max())
print("Flujo qy máximo:", qy.max())


# Representacion de la Temperatura
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

im0 = axes[0].imshow(
    T, origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="coolwarm", aspect="equal"
)
axes[0].set_title("Temperatura")
axes[0].set_xlabel("x (cm)")
axes[0].set_ylabel("y (cm)")
fig.colorbar(im0, ax=axes[0], label="T (°C)")

im1 = axes[1].imshow(
    qx, origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="coolwarm", aspect="equal"
)
axes[1].set_title("Flujo térmico $q_x$")
axes[1].set_xlabel("x (cm)")
axes[1].set_ylabel("y (cm)")
fig.colorbar(im1, ax=axes[1], label="cal/(s cm²)")

im2 = axes[2].imshow(
    qy, origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="coolwarm", aspect="equal"
)
axes[2].set_title("Flujo térmico $q_y$")
axes[2].set_xlabel("x (cm)")
axes[2].set_ylabel("y (cm)")
fig.colorbar(im2, ax=axes[2], label="cal/(s cm²)")

fig.suptitle("Placa de aluminio: estado estacionario")
plt.tight_layout()
plt.show()
