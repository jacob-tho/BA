from ngsolve import * 
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
# MeshGrid
# Linear und Bilinearformen
"""
mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
mesh.nv, mesh.ne
#Draw(mesh)


fes = H1(mesh, order = 2, dirichlet="bottom|right")
fes.ndof
u,v = fes.TnT()
gfu = GridFunction(fes)

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

f = LinearForm(fes)
f += x*v*dx
f.Assemble()
#print(f.vec)
#print(a.mat)



gfu.vec.data = a.mat.Inverse(freedofs = fes.FreeDofs()) * f.vec
Draw(gfu);

"""

# Funktionen

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

fes = H1(mesh, order=2)
gfu = GridFunction(fes)

gfu.Set(sin(pi*x)*sin(pi*y))

Draw(gfu)

Redraw(blocking=True)
