import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Function boundary 

def make_bc(n, top, bottom):
      
    V = np.zeros((n,n))

    bc_mask = np.zeros_like(V , dtype = bool)
    bc_vals = np.zeros_like(V)

    V[0,:] = top
    V[-1,:] = bottom
    V[:,0] = 0
    V[:,-1] = 0
    
    bc_mask[0,:] = True
    bc_mask[-1,:] = True
    bc_mask[:,0] = True
    bc_mask[:,-1] = True
    
    bc_vals[0,:] = top
    bc_vals[-1,:] = bottom
   

    return V, bc_mask, bc_vals

## Gauss Seidel 
def g_s(V, bc_mask, bc_vals):
    ## note the V.copy() step is gone
    ny, nx = V.shape
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if not bc_mask[i,j]: V[i,j] = 0.25*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])
    V[bc_mask] = bc_vals[bc_mask]
    return V

# Solucion del problema
V, bc_mask, bc_vals = make_bc(40, 100, -100)

# Usamos el metodo numerico
for _ in range(1000):
    V = g_s(V, bc_mask, bc_vals)

# Coordenadas espaciales
x = np.linspace(-1, 1, 40)
y = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(x, y)

# Representación del potencial
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


