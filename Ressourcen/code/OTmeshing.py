from ngsolve.meshes import MakeStructured2DMesh, MakeHexMesh
import netgen.meshing as NetgenMeshing

#from netgen.meshing import *
from netgen.meshing import *
from netgen.csg import *
from netgen.geom2d import unit_square, SplineGeometry
import netgen.geom2d as geom2d
from netgen.geom2d import CSG2d, Circle, Rectangle

from ngsolve import *

#from ngsolve.krylovspace import CG,CGSolver
from ngsolve.fem import MinimizationCF
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import matplotlib.pyplot as plt

from ngsolve.webgui import Draw

from scipy.optimize import root, minimize
import scipy.sparse as sp

from ngsolve.krylovspace import GMRes

class OTMesh:
    def __init__(self, h=.1, nz = 10, holes = False, curve = None):
        self.h = h
        self.curve = curve
        self.nz = nz
        if holes:
            self.ngsmesh, self.mesh2d = self.Create2dMeshHole()
        else:
            self.ngsmesh, self.mesh2d = self.Create2dMesh()
        print(self.ngsmesh)
        print(self.mesh2d)
        if holes:
            self.mesh3d = self.CreateMesh3dHole()
        else:
            self.mesh3d = self.Create3dMesh()
        
    def Create2dMeshHole(self):#, tocurve_start, tocurve_end):
        geo = geom2d.SplineGeometry()
        p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [(0,0), (1,0), (1,1), (0,1)] ]
        
        obs_pts = [(0.4,0.3), (0.4,0.5), (0.5,0.5), (0.5,0.3)]       
        obstacle_points = [ geo.AppendPoint(x,y) for x,y in obs_pts ] #g1, g2, g3, g4 = [ geo.AppendPoint(x,y) for x,y in curve_points ]
        
        # Define the outer rectangle
        geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="Neumann")
        geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="Neumann")
        geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="Neumann")
        geo.Append(["line", p4, p1], leftdomain=1, rightdomain=0, bc="inflow")

        # Define the inner rectangle (hole) using obstacle_points
        geo.Append(["line", obstacle_points[0], obstacle_points[1]], leftdomain=1, rightdomain=0, bc="curve")
        geo.Append(["line", obstacle_points[1], obstacle_points[2]], leftdomain=1, rightdomain=0, bc="curve")
        geo.Append(["line", obstacle_points[2], obstacle_points[3]], leftdomain=1, rightdomain=0, bc="curve")
        geo.Append(["line", obstacle_points[3], obstacle_points[0]], leftdomain=1, rightdomain=0, bc="curve")
        
        

        # # curve_points = [(0.2,0.2), (0.25,0.2), (0.4,0.35), (0.5,0.55)]
        # pts = np.arange(.1,.9,.1)
        # vals = pts**2+.2
        # # curve_points = [(0.25,0.5), (0.75,0.5)]#, (0.4,0.35), (0.5,0.55)]
        # curve_points = list(zip(pts,vals))
        # pts = [ geo.AppendPoint(x,y) for x,y in curve_points ] #g1, g2, g3, g4 = [ geo.AppendPoint(x,y) for x,y in curve_points ]
        # npts = len(curve_points)
        # # dgamma = np.diff(curve_points, axis=0) # difference between consecutive points
        # # np.linalg.norm(dgamma, axis=1) # length of each segment

        # # print(p4,pts[0])
        # tocurve_pt_start = p1
        # tocurve_pt_end = p3
        # geo.Append (["line", tocurve_pt_start, pts[0]], leftdomain=1, rightdomain=2, bc="tocurve") ## If you change the vertices here, you need to change the declaration of leftdomain below
        # for i in range(npts-1):
        #     geo.Append (["line", pts[i], pts[i+1]], leftdomain=1, rightdomain=2, bc="curve")
        # #geo.Append (["line", g2, g3], leftdomain=1, rightdomain=2, bc="curve")
        # #geo.Append (["line", g3, g4], leftdomain=1, rightdomain=2, bc="curve")

        # geo.Append (["line",pts[npts-1], tocurve_pt_end], leftdomain=1, rightdomain=2, bc="tocurve") # REMOVE LATER
        
        # #geo.Append (["line", g4, p3], leftdomain=1, rightdomain=2, bc="tocurve")
        # geo.Append (["line", p3, p4], leftdomain=1, rightdomain=0)
        # geo.Append (["line", p4, p1], leftdomain=1, rightdomain=0)

        # geo.Append (["line", p1, p2], leftdomain=2, rightdomain=0)
        # geo.Append (["line", p2, p3], leftdomain=2, rightdomain=0)
        # # geo.Append (["line", p5, p6], leftdomain=2, rightdomain=0, bc="wall")
        # # geo.Append (["line", p6, p3], leftdomain=2, rightdomain=0)

        # hmesh = .02
        #ngsmesh = geo.GenerateMesh (maxh=hmesh)
        #ngsmesh = unit_square.GenerateMesh(maxh=self.h, quad_dominated=False)        
        # geo = CSG2d()

        # # define some primitives
        # rect = Rectangle( pmin=(0,0), pmax=(1,1), mat="mat2", bc="bc_rect" )
        # rect2 = Rectangle( pmin=(0.4,0.3), pmax=(0.5,0.75), mat="mat1", bc="curve" )

        # # use operators +, - and * for union, difference and intersection operations
        # domain1 = rect - rect2
        # #domain2 = circle * rect
        # #domain2.Mat("mat3").Maxh(0.1) # change domain name and maxh
        # #domain3 = rect-circle

        # # add top level objects to geometry
        # geo.Add(domain1)
        # #geo.Add(domain2)
        # #geo.Add(domain3)

        # # generate mesh
        # #m = geo.GenerateMesh(maxh=0.3)

        # # geo = geom2d.SplineGeometry()
        # # p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [(0,0), (1,0), (1,1), (0,1)] ]

    

        ngsmesh = geo.GenerateMesh (maxh=self.h)

        # print("curve vertices are ", pts)

        mesh2d = Mesh(ngsmesh)
        # Draw(mesh2d)

        # Create another 2d mesh used to solve 1d-OT on the curve
        #mesh1d = Mesh(unit_square.GenerateMesh(maxh=hmesh,quad_dominated=True))

        return ngsmesh, mesh2d        
        
    def Create2dMesh(self):
        if self.curve is None:
            geo = geom2d.SplineGeometry()
            p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [(0,0), (1,0), (1,1), (0,1)] ]
            
            obs_pts = [(0.4,0.3), (0.4,0.5), (0.5,0.5), (0.5,0.3)]       
            obstacle_points = [ geo.AppendPoint(x,y) for x,y in obs_pts ] #g1, g2, g3, g4 = [ geo.AppendPoint(x,y) for x,y in curve_points ]
            
            # Define the outer rectangle
            geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="Neumann")
            geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="Neumann")
            geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="Neumann")
            geo.Append(["line", p4, p1], leftdomain=1, rightdomain=0, bc="inflow")

            # # curve_points = [(0.2,0.2), (0.25,0.2), (0.4,0.35), (0.5,0.55)]
            # pts = np.arange(.1,.9,.1)
            # vals = pts**2+.2
            # # curve_points = [(0.25,0.5), (0.75,0.5)]#, (0.4,0.35), (0.5,0.55)]
            # curve_points = list(zip(pts,vals))
            # pts = [ geo.AppendPoint(x,y) for x,y in curve_points ] #g1, g2, g3, g4 = [ geo.AppendPoint(x,y) for x,y in curve_points ]
            # npts = len(curve_points)
            # # dgamma = np.diff(curve_points, axis=0) # difference between consecutive points
            # # np.linalg.norm(dgamma, axis=1) # length of each segment

            # # print(p4,pts[0])
            # tocurve_pt_start = p1
            # tocurve_pt_end = p3
            # geo.Append (["line", tocurve_pt_start, pts[0]], leftdomain=1, rightdomain=2, bc="tocurve") ## If you change the vertices here, you need to change the declaration of leftdomain below
            # for i in range(npts-1):
            #     geo.Append (["line", pts[i], pts[i+1]], leftdomain=1, rightdomain=2, bc="curve")
            # #geo.Append (["line", g2, g3], leftdomain=1, rightdomain=2, bc="curve")
            # #geo.Append (["line", g3, g4], leftdomain=1, rightdomain=2, bc="curve")

            # geo.Append (["line",pts[npts-1], tocurve_pt_end], leftdomain=1, rightdomain=2, bc="tocurve") # REMOVE LATER
            
            # #geo.Append (["line", g4, p3], leftdomain=1, rightdomain=2, bc="tocurve")
            # geo.Append (["line", p3, p4], leftdomain=1, rightdomain=0)
            # geo.Append (["line", p4, p1], leftdomain=1, rightdomain=0)

            # geo.Append (["line", p1, p2], leftdomain=2, rightdomain=0)
            # geo.Append (["line", p2, p3], leftdomain=2, rightdomain=0)
            # # geo.Append (["line", p5, p6], leftdomain=2, rightdomain=0, bc="wall")
            # # geo.Append (["line", p6, p3], leftdomain=2, rightdomain=0)

            # hmesh = .02
            ngsmesh = geo.GenerateMesh (maxh=self.h)
            #ngsmesh = unit_square.GenerateMesh(maxh=self.h, quad_dominated=False)

            #print("curve vertices are ", pts)

            mesh2d = Mesh(ngsmesh)
            #Draw(mesh2d)


            # Create another 2d mesh used to solve 1d-OT on the curve
            #mesh1d = Mesh(unit_square.GenerateMesh(maxh=hmesh,quad_dominated=True))#(maxh=0.02,quad_dominated=True))        else:
            #pass
        return ngsmesh, mesh2d
    
    def CreateMesh3dHole(self):
        # Next step: Take 2d mesh and extend it to 3d using prism elements
        nz = self.nz
        
        ngsmesh3 = NetgenMeshing.Mesh()
        ngsmesh3.dim = 3
        pids = []

        ind_curve_top = []
        ind_curve_bottom = []

        ind_neumann = []
        ind_top = []
        ind_bottom = []

        # Copy points to new mesh
        for p in self.ngsmesh.Points():
            pids.append(ngsmesh3.Add(MeshPoint(Pnt(p[0],p[1],p[2]))))

        n2dpoints = len(pids)
        # print("Initial number of points = ", n2dpoints)


        # Add additional points in z-direction
        for k in range(nz):
            for p in self.ngsmesh.Points():
                pids.append(ngsmesh3.Add(MeshPoint(Pnt( p[0],p[1],p[2]+(k+1)/nz ))))

        fd_bottom = ngsmesh3.Add(FaceDescriptor(surfnr=0, domin=1, bc=1))
        fd_top = ngsmesh3.Add(FaceDescriptor(surfnr=1, domin=1, bc=2))
        fd_neumann = ngsmesh3.Add(FaceDescriptor(surfnr=2, domin=1, bc=3))
        fd_obstacles = ngsmesh3.Add(FaceDescriptor(surfnr=3, domin=1, bc=4))
        fd_inflow = ngsmesh3.Add(FaceDescriptor(surfnr=4, domin=1, bc=5))

        # Loop through all elements and extend points in z-direction and add prism elements
        for el in self.ngsmesh.Elements2D():
            
            # Add surface elements for "lower" part of cube
            #facenr = 1
            ind_bottom.append(el.vertices[0].nr)
            ind_bottom.append(el.vertices[1].nr)

            ngsmesh3.Add(Element1D([el.vertices[0].nr, el.vertices[1].nr], index=fd_bottom))
            ngsmesh3.Add(Element1D([el.vertices[1].nr, el.vertices[2].nr], index=fd_bottom))
            ngsmesh3.Add(Element1D([el.vertices[2].nr, el.vertices[0].nr], index=fd_bottom))
            ngsmesh3.Add(Element2D(fd_bottom, [el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr]))
            
            # Add surface elements for "upper" part of cube
            #facenr = 2
            ind_top.append(el.vertices[0].nr+nz*n2dpoints)
            ind_top.append(el.vertices[1].nr+nz*n2dpoints)

            ngsmesh3.Add(Element1D([el.vertices[0].nr+nz*n2dpoints,el.vertices[1].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element1D([el.vertices[1].nr+nz*n2dpoints,el.vertices[2].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element1D([el.vertices[2].nr+nz*n2dpoints,el.vertices[0].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element2D(fd_top, [el.vertices[0].nr+nz*n2dpoints, el.vertices[1].nr+nz*n2dpoints, el.vertices[2].nr+nz*n2dpoints]))


            # Add the prism element
            for k in range(nz):
                ngsmesh3.Add(Element3D(1, [el.vertices[0].nr + k*n2dpoints, el.vertices[1].nr + k*n2dpoints, el.vertices[2].nr + k*n2dpoints, 
                                        el.vertices[0].nr + (k+1)*n2dpoints, el.vertices[1].nr + (k+1)*n2dpoints, el.vertices[2].nr + (k+1)*n2dpoints]))



        # Loop trough boundaries
        for k in range(nz):
            for el in self.ngsmesh.Elements1D():

                if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],1.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],1.0)):
                    
                    if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0)):
                        fd = fd_inflow
                    else:
                        fd = fd_neumann
                        
                    ngsmesh3.Add(Element2D(fd, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # 

                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints], index=fd))
                    ngsmesh3.Add(Element1D([el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints], index=fd)) # Hopefully this is the correct order


                if( self.ngsmesh.GetBCName(el.index-1) == "curve"):
                    
                    ngsmesh3.Add(Element2D(fd_obstacles, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # 
                    
                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints], index=fd_obstacles))
                    ngsmesh3.Add(Element1D([el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints], index=fd_obstacles)) # Hopefully this is the correct order

                    # Add boundary conditions on the curve
                    if k==0:
                        ind_curve_top.append(el.vertices[0].nr+nz*n2dpoints)
                        ind_curve_top.append(el.vertices[1].nr+nz*n2dpoints)
                        ind_curve_bottom.append(el.vertices[0].nr)
                        ind_curve_bottom.append(el.vertices[1].nr)
                        ngsmesh3.Add(Element1D([el.vertices[0].nr, el.vertices[1].nr], index=fd_obstacles)) # add initial condition once
                        ngsmesh3.Add(Element1D([el.vertices[0].nr+(nz)*n2dpoints, el.vertices[1].nr+(nz)*n2dpoints], index=fd_obstacles)) # add terminal condition once ## FIXME

        for k in range(nz+1):
            for el in self.ngsmesh.Elements1D():

                if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],1.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],1.0)):
                    
                    if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0)):
                        fd = fd_inflow
                    else:
                        fd = fd_neumann

                    #ind_neumann.append(el.vertices[0].nr+k*n2dpoints)
                    #ind_neumann.append(el.vertices[1].nr+k*n2dpoints)

                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd)) # Add 1d element at "upper" part of cube

                if( self.ngsmesh.GetBCName(el.index-1) == "curve"):
                    
                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_obstacles))
                    

        ngsmesh3.SetBCName(0,"bottom")
        ngsmesh3.SetBCName(1,"top")
        ngsmesh3.SetBCName(2,"Neumann")
        ngsmesh3.SetBCName(3,"obstacles")     
        ngsmesh3.SetBCName(4,"inflow")     

        mesh = Mesh(ngsmesh3)


        mesh.ngmesh.SetCD2Name(1,"one") 
        mesh.ngmesh.SetCD2Name(2,"two") 
        mesh.ngmesh.SetCD2Name(3,"three") 
        mesh.ngmesh.SetCD2Name(4,"cedge") 
        mesh.ngmesh.SetCD2Name(5,"curvetop") 
        mesh.ngmesh.SetCD2Name(6,"curvebottom") 


        # Draw(mesh)

        mesh.GetBBoundaries()

        return mesh    
    
    def Create3dMesh(self):
    # Next stop: Take 2d mesh and extend it to 3d using prism elements
        nz = self.nz

        ngsmesh3 = NetgenMeshing.Mesh()
        ngsmesh3.dim = 3
        pids = []

        ind_curve_top = []
        ind_curve_bottom = []

        ind_neumann = []
        ind_top = []
        ind_bottom = []

        # Copy points to new mesh
        for p in self.ngsmesh.Points():
            pids.append(ngsmesh3.Add(MeshPoint(Pnt(p[0],p[1],p[2]))))

        n2dpoints = len(pids)
        print("Initial number of points = ", n2dpoints)


        # Add additional points in z-direction
        for k in range(nz):
            for p in self.ngsmesh.Points():
                pids.append(ngsmesh3.Add(MeshPoint(Pnt( p[0],p[1],p[2]+(k+1)/nz ))))

        fd_bottom = ngsmesh3.Add(FaceDescriptor(surfnr=0, domin=1, bc=1))
        fd_top = ngsmesh3.Add(FaceDescriptor(surfnr=1, domin=1, bc=2))
        fd_neumann = ngsmesh3.Add(FaceDescriptor(surfnr=2, domin=1, bc=3))
        #fd_curve = ngsmesh3.Add(FaceDescriptor(surfnr=3, domin=1, bc=4))
        #fd_ctop = ngsmesh3.Add(FaceDescriptor(surfnr=4, domin=1, bc=5)) # FIXME: ?
        #fd_cbottom = ngsmesh3.Add(FaceDescriptor(surfnr=5, domin=1, bc=6)) # FIXME: ?

        # Loop through all elements and extend points in z-direction and add prism elements
        for el in self.ngsmesh.Elements2D():

            # Add surface elements for "lower" part of cube
            #facenr = 1
            ind_bottom.append(el.vertices[0].nr)
            ind_bottom.append(el.vertices[1].nr)
            #ind_bottom.append(el.vertices[3].nr)

            ngsmesh3.Add(Element1D([el.vertices[0].nr, el.vertices[1].nr], index=fd_bottom))
            ngsmesh3.Add(Element1D([el.vertices[1].nr, el.vertices[2].nr], index=fd_bottom))
            ngsmesh3.Add(Element1D([el.vertices[2].nr, el.vertices[0].nr], index=fd_bottom))
            ngsmesh3.Add(Element2D(fd_bottom, [el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr]))
                
            # Add surface elements for "upper" part of cube
            #facenr = 2
            ind_top.append(el.vertices[0].nr+nz*n2dpoints)
            ind_top.append(el.vertices[1].nr+nz*n2dpoints)
            #ind_top.append(el.vertices[3].nr+nz*n2dpoints)

            ngsmesh3.Add(Element1D([el.vertices[0].nr+nz*n2dpoints,el.vertices[1].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element1D([el.vertices[1].nr+nz*n2dpoints,el.vertices[2].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element1D([el.vertices[2].nr+nz*n2dpoints,el.vertices[0].nr+nz*n2dpoints], index=fd_top))
            ngsmesh3.Add(Element2D(fd_top, [el.vertices[0].nr+nz*n2dpoints, el.vertices[1].nr+nz*n2dpoints, el.vertices[2].nr+nz*n2dpoints]))


            # Add the prism element
            for k in range(nz):
                ngsmesh3.Add(Element3D(1, [el.vertices[0].nr + k*n2dpoints, el.vertices[1].nr + k*n2dpoints, el.vertices[2].nr + k*n2dpoints, 
                                        el.vertices[0].nr + (k+1)*n2dpoints, el.vertices[1].nr + (k+1)*n2dpoints, el.vertices[2].nr + (k+1)*n2dpoints]))



        # Loop trough boundaries
        for k in range(nz):
            for el in self.ngsmesh.Elements1D():
                
                #print(self.ngsmesh.Points()[el.vertices[0].nr], self.ngsmesh.Points()[el.vertices[1].nr])     

                if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],1.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],1.0)):
                    
                    
                    

                    #ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_neumann)) # Add 1d element at "upper" part of cube
                    
                    # JAN: Auskommentiert, weil ich glaube, dass es falsch ist
                    ngsmesh3.Add(Element2D(fd_neumann, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # 

                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints], index=fd_neumann))
                    ngsmesh3.Add(Element1D([el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints], index=fd_neumann)) # Hopefully this is the correct order
                    


                # if( self.ngsmesh.GetBCName(el.index-1) == "curve"):
                
                #     #ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_curve))
                    
                #     # ind_curve_cur.append(el.vertices[0].nr+k*n2dpoints)
                #     # ind_curve_cur.append(el.vertices[1].nr+k*n2dpoints)

                #     ngsmesh3.Add(Element2D(fd_curve, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # 
                    
                #     ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints], index=fd_curve))
                #     ngsmesh3.Add(Element1D([el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints], index=fd_curve)) # Hopefully this is the correct order

                #     # Add boundary conditions on the curve
                #     if k==0:
                #         ind_curve_top.append(el.vertices[0].nr+nz*n2dpoints)
                #         ind_curve_top.append(el.vertices[1].nr+nz*n2dpoints)
                #         ind_curve_bottom.append(el.vertices[0].nr)
                #         ind_curve_bottom.append(el.vertices[1].nr)
                #         ngsmesh3.Add(Element1D([el.vertices[0].nr, el.vertices[1].nr], index=fd_cbottom)) # add initial condition once
                #         ngsmesh3.Add(Element1D([el.vertices[0].nr+(nz)*n2dpoints, el.vertices[1].nr+(nz)*n2dpoints], index=fd_ctop)) # add terminal condition once ## FIXME

                    # if( ngsmesh.GetBCName(el.index-1) == "tocurve"):
                    #   ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_curve))
                    #   ngsmesh3.Add(Element2D(fd_curve, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # Hopefully this is the correct order      

        for k in range(nz+1):
            for el in self.ngsmesh.Elements1D():

                if( np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][0],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][0],1.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],0.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],0.0) or 
                    np.isclose(self.ngsmesh.Points()[el.vertices[0].nr][1],1.0) and np.isclose(self.ngsmesh.Points()[el.vertices[1].nr][1],1.0)):
                    

                    ind_neumann.append(el.vertices[0].nr+k*n2dpoints)
                    ind_neumann.append(el.vertices[1].nr+k*n2dpoints)

                    ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_neumann)) # Add 1d element at "upper" part of cube
                    #ngsmesh3.Add(Element2D(fd_neumann, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # Hopefully this is the correct order

                    # if( self.ngsmesh.GetBCName(el.index-1) == "curve"):
                    
                    #     ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_curve))
                        
                    #     #ngsmesh3.Add(Element2D(fd_curve, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # Hopefully this is the correct order

                    #     # if( ngsmesh.GetBCName(el.index-1) == "tocurve"):
                    #     #   ngsmesh3.Add(Element1D([el.vertices[0].nr+k*n2dpoints,el.vertices[1].nr+k*n2dpoints], index=fd_curve))
                    #     #   ngsmesh3.Add(Element2D(fd_curve, [el.vertices[0].nr+k*n2dpoints, el.vertices[0].nr+(k+1)*n2dpoints, el.vertices[1].nr+(k+1)*n2dpoints, el.vertices[1].nr+k*n2dpoints])) # Hopefully this is the correct order      


        ngsmesh3.SetBCName(0,"bottom")
        ngsmesh3.SetBCName(1,"top")
        ngsmesh3.SetBCName(2,"Neumann")
        # ngsmesh3.SetBCName(3,"curve") 
        # ngsmesh3.SetBCName(4,"curvetop") 
        # ngsmesh3.SetBCName(5,"curvebottom") 

        # ngsmesh3.SetCD2Name(1,"one") 
        # ngsmesh3.SetCD2Name(2,"two") 
        # ngsmesh3.SetCD2Name(3,"three") 
        # ngsmesh3.SetCD2Name(4,"four") 
        # ngsmesh3.SetCD2Name(5, "five") #"curvetop") 
        # ngsmesh3.SetCD2Name(6,"six") #"curvebottom") 
            

        mesh3d = Mesh(ngsmesh3)


        # mesh3d.ngmesh.SetCD2Name(1,"one") 
        # mesh3d.ngmesh.SetCD2Name(2,"two") 
        # mesh3d.ngmesh.SetCD2Name(3,"three") 
        # mesh3d.ngmesh.SetCD2Name(4,"cedge") 
        # mesh3d.ngmesh.SetCD2Name(5,"curvetop") 
        # mesh3d.ngmesh.SetCD2Name(6,"curvebottom") 

        #mesh = MakeHexMesh()

        #Draw(mesh)

        #mesh.GetBBoundaries()
        return mesh3d
