from safescope import safescope, Imports
import pdb
"""
Alles TODOS bearbeiten, Fallunterscheidung Diffusion, InitialFinal vollständig bearbeiten
funcderivInv -> UpdatePhi if not source & if bInverse
"""

with Imports(): # Necessary to have them available in functions decorated with @safescope
    from ngsolve.meshes import MakeStructured2DMesh, MakeHexMesh
    import netgen.meshing as NetgenMeshing

    #from netgen.meshing import *
    #from netgen.meshing import *
    #from netgen.csg import *
    from netgen.geom2d import unit_square, SplineGeometry
    import netgen.geom2d as geom2d

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

    # Helper class to create 3d mesh with thrid dimension being the time
    from OTmeshing import OTMesh

    from tqdm.notebook import trange, tqdm


with Imports():
    #TODO aufteilen in dif und kein dif
    """
    def funcderiv2(u,v, tmpr, tmpf, nrmv, alph1, alph2):
        ### TODO Newton
        fval = np.array([u**3 + alph1/alph2*u*v**2 + 2*(alph1**2+alph1*tmpr)*u - 2*alph1**2*nrmv,
                        v**3 + alph2/alph1*u**2*v + 2*(alph2**2+alph2*tmpr)*v - 2*alph2**2*tmpf])
        derivval = np.array([[3*u**2 + alph1/alph2*v**2 + 2*(alph1**2+alph1*tmpr), 2*alph1/alph2*u*v],
                            [2*alph2/alph1*u*v, 3*v**2 + alph2/alph1*u**2 + 2*(alph2**2+alph2*tmpr)]])
        return fval, derivval
    """
    #Jetzt mit besseren Variablennamen, außerdem ist alph1 = 1
    def funcderivInv(rhos, Jxs, Jys, alpha_inv, alpha_f, r1, etarhos, etaJxs, etaJys, rhodat, bSource = False, fs = None, etafs = None, bDif=False, alpha_dif = None, gxs = None, gys = None, etagxs = None, etagys = None):
        #baralpha_inv = 2*(1/alpha_inv - alpha_inv/2)
        # TODO??? Noch Fallunterscheidungen wenn Parameterfehlen /try-except
        baralpha_inv = 1/alpha_inv**2
        norm_Js_sqaured = Jxs**2 + Jys**2
        Jsums = Jxs + Jys
        ones = np.ones_like(rhos)
        if bSource and not bDif:
            #K = (1/alpha_inv-alpha_inv/2)*(rhos + 0.5*(Jxs**2 + Jys**2) + 0.5*(1/alpha_f)*fs**2)
            weighted_norm_fs = 1/(2*alpha_f**2)*np.abs(fs)**2
            K = baralpha_inv*(rhos + 0.5*norm_Js_sqaured + weighted_norm_fs)

            fval = np.array([rhodat + K - r1*etarhos + r1*rhos,
                             rhodat*Jxs + Jxs*K - r1*etaJxs + r1*Jxs,
                             rhodat*Jys + Jys*K - r1*etaJys + r1*Jys,
                             (1/alpha_f**2)*rhodat*fs + (fs/alpha_f**2)*K - r1*etafs + r1*fs])
            
            #derivval = np.array([[(baralpha_inv + r1)*np.ones_like(rhos), baralpha_inv*Jsums, baralpha_inv*Jsums, (baralpha_inv/alpha_f)*fs],
            #                     [baralpha_inv*Jxs, rhodat + baralpha_inv*(K+Jxs*Jsums)+r1, baralpha_inv*Jxs*Jsums, 2*(baralpha_inv/alpha_f)*Jxs*fs],
            #                     [baralpha_inv*Jys, baralpha_inv*Jys*Jsums, rhodat + baralpha_inv*(K+Jys*Jsums)+r1, 2*(baralpha_inv/alpha_f)*Jys*fs],
            #                     [(baralpha_inv/alpha_f)*fs, (baralpha_inv/alpha_f)*fs*Jsums, (baralpha_inv/alpha_f)*fs*Jsums, 1/alpha_f*rhodat + (baralpha_inv/alpha_f)*K + baralpha_inv*(fs/alpha_f)**2 + r1]])
            derivval = np.array([
                [(baralpha_inv + r1)*ones,
                  baralpha_inv*Jxs, 
                  baralpha_inv*Jys,
                  baralpha_inv*fs/alpha_f**2],
                [baralpha_inv*Jxs, 
                 rhodat + K + baralpha_inv*Jxs**2 + r1, 
                 baralpha_inv*Jxs*Jys, 
                 baralpha_inv*Jxs * fs/alpha_f**2],
                [baralpha_inv*Jys,
                 baralpha_inv * Jxs * Jys,
                 rhodat + K + baralpha_inv * Jys**2 + r1,
                 baralpha_inv * Jys * fs/alpha_f**2],
                [baralpha_inv*fs/alpha_f**2,
                 baralpha_inv*fs*Jxs/alpha_f**2,
                 baralpha_inv*fs*Jys/alpha_f**2,
                 (rhodat + K)/alpha_f** + baralpha_inv*fs**2 / alpha_f**4 + r1]
                ])

        elif bSource and bDif:

            norm_gs = np.sqrt(gxs**2 + gys**2)
            weighted_norm_fs = fs**2/(2*alpha_f**2)
            weighted_norm_gs = norm_gs**2/(2*alpha_dif**2) 
            
            K = baralpha_inv*(rhos + 0.5*norm_Js_sqaured + weighted_norm_fs + weighted_norm_gs)
            
            fval = np.array([rhodat + K - r1*etarhos + r1*rhos,
                            rhodat*Jxs + Jxs*K - r1*etaJxs + r1*Jxs,
                            rhodat*Jys + Jys*K - r1*etaJys + r1*Jys,
                            (1/alpha_f**2)*rhodat*fs + (fs/alpha_f**2)*K - r1*etafs + r1*fs,
                            (1/alpha_dif**2)*rhodat*gxs + (1/alpha_dif**2)*K*gxs - r1*etagxs + r1*gxs,
                            (1/alpha_dif**2)*rhodat*gys + (1/alpha_dif**2)*K*gys - r1*etagys + r1*gys])

            derivval = np.array([
            [(baralpha_inv + r1) * ones,
             baralpha_inv * Jxs,
             baralpha_inv * Jys,
             baralpha_inv * fs / alpha_f**2,
             baralpha_inv * gxs / alpha_dif**2,
             baralpha_inv * gys / alpha_f**2],
            [baralpha_inv * Jxs,
             rhodat + K + baralpha_inv * Jxs**2 + r1,
             baralpha_inv * Jxs * Jys,
             baralpha_inv * Jxs * fs / alpha_f**2,
             baralpha_inv * Jxs * gxs / alpha_dif**2,
             baralpha_inv * Jxs * gys / alpha_dif**2],
            [baralpha_inv * Jys,
             baralpha_inv * Jxs * Jys,
             rhodat + K + baralpha_inv * Jys**2 + r1,
             baralpha_inv * Jys * fs / alpha_f**2,
             baralpha_inv * Jys * gxs / alpha_dif**2,
             baralpha_inv * Jys * gys / alpha_dif**2],
            [baralpha_inv * fs / alpha_f**2,
             baralpha_inv * fs * Jxs / alpha_f**2,
             baralpha_inv * fs * Jys / alpha_f**2,
             (rhodat + K)/alpha_f**2 + baralpha_inv * fs**2 / alpha_f**4 + r1,
             baralpha_inv * fs * gxs/(alpha_f**2 * alpha_dif**2),
             baralpha_inv * fs * gys/(alpha_f**2 * alpha_dif**2)],
            [baralpha_inv * gxs / alpha_dif**2,
             baralpha_inv * gxs * Jxs / alpha_dif**2,
             baralpha_inv * gxs * Jys / alpha_dif**2,
             baralpha_inv * gxs * fs/(alpha_dif**2 * alpha_f**2),
             (rhodat + K) / alpha_dif**2 + baralpha_inv * gxs**2 / alpha_dif**4+ r1,
             baralpha_inv * gxs * gys / alpha_dif**4],
            [baralpha_inv * gys / alpha_dif**2,
             baralpha_inv * gys * Jxs / alpha_dif**2,
             baralpha_inv * gys * Jys / alpha_dif**2,
             baralpha_inv * gys * fs/(alpha_dif**2 * alpha_f**2),
             baralpha_inv * gxs * gys / alpha_dif**4,
             (rhodat + K) / alpha_dif**2+baralpha_inv * gys**2 / alpha_dif**4+r1]
        ])

        elif not bSource and bDif:
            norm_gs = np.sqrt(gxs**2 + gys**2)
            #weighted_norm_fs = 1/(2*alpha_f**2)*np.abs(fs)**2
            weighted_norm_gs = 1/(2*alpha_dif**2)*norm_gs**2 
            
            K = baralpha_inv*(rhos + 0.5*norm_Js_sqaured + weighted_norm_gs)#+ weighted_norm_fs + weighted_norm_gs)
                        
            fval = np.array([rhodat + K - r1*etarhos + r1*rhos,
                            rhodat*Jxs + Jxs*K - r1*etaJxs + r1*Jxs,
                            rhodat*Jys + Jys*K - r1*etaJys + r1*Jys,
                            #(1/alpha_f**2)*rhodat*fs + (fs/alpha_f**2)*K - r1*etafs + r1*fs,
                            (1/alpha_dif**2)*rhodat*gxs + (1/alpha_dif**2)*K*gxs - r1*etagxs + r1*gxs,
                            (1/alpha_dif**2)*rhodat*gys + (1/alpha_dif**2)*K*gys - r1*etagys + r1*gys])

            derivval = derivval = np.array([
            [(baralpha_inv + r1) * ones,
             baralpha_inv * Jxs,
             baralpha_inv * Jys,
             baralpha_inv * gxs / alpha_dif**2,
             baralpha_inv * gys / alpha_dif**2],
            [baralpha_inv * Jxs,
             rhodat + K + baralpha_inv * Jxs**2 + r1,
             baralpha_inv * Jxs * Jys,
             baralpha_inv * Jxs * gxs / alpha_dif**2,
             baralpha_inv * Jxs * gys / alpha_dif**2],
            [baralpha_inv * Jys,
             baralpha_inv * Jxs * Jys,
             rhodat + K + baralpha_inv * Jys**2 + r1,
             baralpha_inv * Jys * gxs / alpha_dif**2,
             baralpha_inv * Jys * gys / alpha_dif**2],
            [baralpha_inv * gxs / alpha_dif**2,
             baralpha_inv * gxs * Jxs / alpha_dif**2,
             baralpha_inv * gxs * Jys / alpha_dif**2,
             (rhodat + K) / alpha_dif**2+ baralpha_inv * gxs**2 / alpha_dif**4+ r1,
             baralpha_inv * gxs * gys / alpha_dif**4],
            [baralpha_inv * gys / alpha_dif**2,
             baralpha_inv * gys * Jxs / alpha_dif**2,
             baralpha_inv * gys * Jys / alpha_dif**2,
             baralpha_inv * gxs * gys / alpha_dif**4,
             (rhodat + K) / alpha_dif**2+ baralpha_inv * gys**2 / alpha_dif**4+ r1]
        ])

        else: # No Source, no Diffusion
            K = baralpha_inv*(rhos + 0.5*(Jxs**2 + Jys**2))
            fval = np.array([rhodat + K - r1*etarhos + r1*rhos,
                             rhodat*Jxs + Jxs*K - r1*etaJxs + r1*Jxs,
                             rhodat*Jys + Jys*K - r1*etaJys + r1*Jys])

            derivval = np.array([
                [(baralpha_inv + r1)*ones, 
                 baralpha_inv*Jxs, 
                 baralpha_inv*Jys],
                [baralpha_inv*Jxs, 
                 rhodat + K + baralpha_inv* Jxs**2 +r1, 
                 baralpha_inv*Jxs*Jys],
                [baralpha_inv*Jys, 
                 baralpha_inv*Jys*Jxs, 
                 rhodat + K + baralpha_inv*Jys**2+r1]])

        return fval, derivval



    def funcderiv2(x, y, eta_rhos, eta_Js_norm, eta_fs_norm, alpha_f):
        fval = np.array([x**3 + x*y**2/alpha_f**2 + 2*(1+ eta_rhos)*x - 2*eta_Js_norm,
                         x**2*y + y**3/alpha_f**2 + 2*(alpha_f**2 + eta_rhos)*y - 2*alpha_f**2*eta_fs_norm])

        derivval = np.array([
            [3*x**2 + y**2 / alpha_f**2 + 2*(1+eta_rhos),
            2*x*y/alpha_f**2],
            [2*x*y,
             x**2 + 3*y**2 / alpha_f**2 + 2*(alpha_f**2 + eta_rhos)]
            ])

        return fval, derivval

    def funcderiv2_dif(x, y, u, eta_rhos, eta_Js_norm, eta_fs_norm, eta_gs_norm, alpha_dif, alpha_f):
        fval = np.array([x**3 + (x*y**2)/alpha_f**2 + (x*u**2)/alpha_dif**2 + 2*(1+eta_rhos)*x - 2*eta_Js_norm,
                         x**2*y + y**3/alpha_f**2 + (y*u**2)/alpha_dif**2 + 2*(alpha_f**2 + eta_rhos)*y - 2*alpha_f**2*eta_fs_norm,
                         x**2*u + (y**2*u)/alpha_f**2 + u**3/alpha_dif**2 + 2*(alpha_dif**2+eta_rhos)*u - 2*alpha_dif**2*eta_gs_norm]) 
        
        derivval = np.array([
            [3*x**2 + y**2/alpha_f**2 + u**2/alpha_dif**2 + 2*(1+eta_rhos),
             2*x*y/alpha_f**2, 
             2*x*u/alpha_dif**2],
            [2*x*y,
            x**2 + 3*y**2/alpha_f**2 + u**2/alpha_dif**2 + 2*(alpha_f**2 + eta_rhos),
             2*y*u/alpha_dif**2],
            [2*x*u,
             2*y*u/alpha_f**2,
             x**2 + y**2/alpha_f**2 + 3*u**2/alpha_dif**2 + 2*(alpha_dif**2 + eta_rhos)]
            ])
        return fval, derivval

    def funcderiv_no_source_dif(x, u, eta_rhos, eta_Js_norm, eta_gs_norm, alpha_dif):
        
        fval = np.array([x**3 + x*u**2/alpha_dif**2 + 2*(1+ eta_rhos)*x - 2*eta_Js_norm,
                         x**2*u + u**3 / alpha_dif**2 + 2*(alpha_dif**2 + eta_rho)*u - 2*alpha_dif**2*eta_g_norm])

        derivval = np.array([
            [3*x**2 + u**2/alpha_dif**2 + 2*(1+eta_rhos),
             2*x*u/alpha_dif**2],
            [2*x*u,
             x**2 + 3*u**2 / alpha_dif**2 + 2*(alpha_dif**2 + eta_rhos)]
            ])
    """
    def funcderivInv_new(rhos, Jxs, Jys, alpha_inv, alpha_f, alpha_dif, r1, etarhos, etaJxs, etaJys, rhodat, 
                         bSource = False, fs = None, etafs = None, gs = None, etags = None):
            #Das als Parameter?
            gxs = None 
            gys = None 
            gs_norm = gxs**2 + gys**2
            
            etagxs = None 
            etagys = None 
            baralpha_inv = 2*(1/alpha_inv - alpha_inv/2)
            Jsums = Jxs + Jys
            if bSource:
                """
                Schauen, ob in der Theorie richtig mit gs_norm**2 manchmal und manchmal nicht
                """
                K_sqaured = (1/alpha_inv-alpha_inv/2)*(rhos + 0.5*(Jxs**2 + Jys**2) + 0.5*(1/alpha_f)*fs**2 + 1/(2*alpha_dif)* gs_norm**2) 
                K = (1/alpha_inv-alpha_inv/2)*(rhos + 0.5*(Jxs**2 + Jys**2) + 0.5*(1/alpha_f)*fs**2 + 1/(2*alpha_dif)* gs_norm) 
                fval = np.array([rhodat + 2*K_sqaured - r1*etarhos + r1*rhos,
                                 rhodat*Jxs + 2*Jxs*K + r1(Jxs - etaJxs),
                                 rhodat*Jys + 2*Jys*K + r1*(Jys - etaJys),
                                 (1/alpha_f)*rhodat*fs + 1/alpha_f * K *fs + r1*(fs - etafs),
                                 1/alpha_dif * rhodat*gxs + 1/alpha_dif*K_sqaured*gxs + r1*(gxs-etagxs),
                                 1/alpha_dif * rhodat*gys + 1/alpha_dif*K_sqaured*gys + r1*(gys-etagys)
                                 ])
                #TODO
                derivval = np.array([[(baralpha_inv + r1)*np.ones_like(rhos), baralpha_inv*Jsums, baralpha_inv*Jsums, (baralpha_inv/alpha_f)*fs],
                                     [baralpha_inv*Jxs, rhodat + baralpha_inv*(K+Jxs*Jsums)+r1, baralpha_inv*Jxs*Jsums, 2*(baralpha_inv/alpha_f)*Jxs*fs],
                                     [baralpha_inv*Jys, baralpha_inv*Jys*Jsums, rhodat + baralpha_inv*(K+Jys*Jsums)+r1, 2*(baralpha_inv/alpha_f)*Jys*fs],
                                     [(baralpha_inv/alpha_f)*fs, (baralpha_inv/alpha_f)*fs*Jsums, (baralpha_inv/alpha_f)*fs*Jsums, 1/alpha_f*rhodat + (baralpha_inv/alpha_f)*K + baralpha_inv*(fs/alpha_f)**2 + r1]])
                


            else:
                K = (1/alpha_inv-alpha_inv/2)*(rhos + 0.5*(Jxs**2 + Jys**2) + 1/(2*alpha_dif)*gs_norm)
                K_squared = K = (1/alpha_inv-alpha_inv/2)*(rhos + 0.5*(Jxs**2 + Jys**2) + 1/(2*alpha_dif)*gs_norm**2)
                fval = np.array([rhodat + 2*K_sqaured - r1*etarhos + r1*rhos,
                                rhodat*Jxs + 2*Jxs*K + r1(Jxs - etaJxs),
                                rhodat*Jys + 2*Jys*K + r1*(Jys - etaJys),
                                                 #(1/alpha_f)*rhodat*fs + 1/alpha_f * K *fs + r1*(fs - etafs),
                                1/alpha_dif * rhodat*gxs + 1/alpha_dif*K_sqaured*gxs + r1*(gxs-etagxs),
                                1/alpha_dif * rhodat*gys + 1/alpha_dif*K_sqaured*gys + r1*(gys-etagys)])
                #TODO
                derivval = np.array([[(baralpha_inv + r1)*np.ones_like(rhos), baralpha_inv*Jsums, baralpha_inv*Jsums],
                                     [baralpha_inv*Jxs, rhodat + baralpha_inv*(K+Jxs*Jsums)+r1, baralpha_inv*Jxs*Jsums],
                                     [baralpha_inv*Jys, baralpha_inv*Jys*Jsums, rhodat + baralpha_inv*(K+Jys*Jsums)+r1]])
                
            return fval, derivval
    """

@safescope
def SolveStokes(mesh2d, alpha_wind = 1, mu_val=1):
    V = VectorH1(mesh2d, order=1, dirichlet="curve|inflow")
    Q = H1(mesh2d, order=1)
    X = V*Q
    
    (u,p),(v,q) = X.TnT()

    stokes = mu_val*InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx + div(v)*p*dx
    a = BilinearForm(stokes).Assemble()

    gf = GridFunction(X)
    gfu, gfp = gf.components

    #uin = CF ( (1.5*4*y*(0.41-y)/(0.41*0.41), 0, 0) )
    #uin = CF ( ((1-0*.5*y), 0, 0) )
    uin = alpha_wind*CF ( ((1-0*.5*y), 0) )
    gfu.Set(uin, definedon=mesh2d.Boundaries("inflow"))

    res = -a.mat * gf.vec
    inv = a.mat.Inverse(freedofs=X.FreeDofs()) #, inverse="pardiso")
    gf.vec.data += inv * res

    #Draw(gfu)

    return gfu


def CreateVariables_new(VV, WW):#, WW): #VV H1, WW L2

    # "testfunction"
    phi = GridFunction(VV)
    theta = GridFunction(VV*VV) #TODO Null-Randbedingungen?

    # primale Variablen
    rho = GridFunction(WW) # doch VV?
    Jx = GridFunction(WW)
    Jy = GridFunction(WW)
    f = GridFunction(WW)
    gx = GridFunction(WW)
    gy = GridFunction(WW)

    # duale Vairbalen
    rhos = GridFunction(WW) # doch VV?
    Jxs = GridFunction(WW)
    Jys = GridFunction(WW)
    fs = GridFunction(WW)
    gxs = GridFunction(WW)
    gys = GridFunction(WW)
    
    dphi = GridFunction(WW*WW*WW)

    rho0 = GridFunction(WW)
    rho1 = GridFunction(WW)

    return phi, theta, rho, Jx, Jy, f, gx, gy, rhos, Jxs, Jys, fs, gxs, gys, dphi, rho0, rho1


@safescope 
def assembleBilinearForm_new(V, r1, bWind, bSource, bDif, vWindX, vWindY, tol=1e-4):
    """
    Unterschiedliche Räume zusammenführen zum Produktraum.
    Gradienten bestimmen, Fallunterscheidungen machen (Wind, Source)
    Evtl Syntax mit CF von oben übernehmen für Leserlichkeit?
    """
    # Das als Fallunterscheidung, wenn bDif dann den Produktraum. Oder eher diese Funktion bei der Fallunterscheidung aufrufen?
    # Dann den Produktraum noch zurückgeben
    #TODO Fallunterscheidung; Checke Raumdimensionen
    if bDif:
        V = FESpace([V, V*V])
        (phi, theta), (psi, zeta) = V.TnT()
    else:
        (phi, psi) = V.TnT()

    # Räumlicher Gradient (hier müsste schon x,y,t Koordinaten sein?)
    #gradx_phi = CF((grad(phi)[0], grad(phi)[1]))
    #gradx_psi = CF((grad(psi)[0], grad(psi)[1]))

    grad_phi = grad(phi)
    grad_psi = grad(psi)

    gradx_phi = grad_phi[0]
    grady_phi = grad_phi[1]

    gradx_psi = grad_psi[0]
    grady_psi = grad_psi[1]
    dt_phi = grad(phi)[2]
    dt_psi = grad(psi)[2]


    # Divergenz-Wind-Term
    if bWind:
        vWind = CF((vWindX, vWindY))
        wind_phi = InnerProduct(vWind, CF((gradx_phi, grady_phi))) 
        wind_psi = InnerProduct(vWind, CF((gradx_psi, grady_psi)))
    else:
        wind_phi = 0
        wind_psi = 0


    # Bilinear-Form 
    aprod = BilinearForm(V, condense = True)

    if bDif:
        div_theta = grad(theta[0])[0] + grad(theta[1])[1]
        div_zeta = grad(zeta[0])[0] + grad(zeta[1])[1]
        mixed_trial = dt_phi + wind_phi - div_theta 
        mixed_test = dt_psi + wind_psi - div_zeta 
        aprod += r1 * mixed_trial * mixed_test * dx
        aprod += r1 * (theta[0] * zeta[0] + theta[1] + zeta[1]) * dx
    # Einzelne Terme hinzufügen 
    aprod += r1 * (gradx_phi * gradx_psi + grady_phi * grady_psi) * dx 

    if bSource:
        aprod += r1 * phi * psi * dx
    else:
        aprod += tol * phi * psi * dx

    aprod.Assemble() 

    invaprod = aprod.mat.Inverse(X.FreeDofs(True), inverse = "pardiso")
    return invaprod #V, aprod

@safescope 
def SolvePhiTheta_new(V, r1, rho0, rho1, rho, Jx, Jy, gx, gy, rhos, Jxs, Jys, gxs, gys, source, sources, invaprod, bWind, bSource, bInitialFinal, vWindX = 0, vWindY = 0, debug = False):
    ### TODO 
    # Dimension von X gibt Fallunterscheidung von bDif oder nicht
    # Dann den Produktraum noch zurückgeben
    if debug == True:
        print(f"Dimension {V.dim}, Typ: {type(V)}")
        #dim = X.dim # oder instanceof?
    #X = FESpace([Vpsi, Vzeta]) # Direkt X übergeben?
    if isinstance(V, FESpace): #Syntax#TODO Andere Methode für Fallunterscheidung nehmen?
        psi, zeta = V.TestFunction()
        #psi = Vpsi.TestFunction()
        #zeta = Vzeta.TestFunction                                                                                                                                                  
        #pdb.set_trace()
        fbulk = LinearForm(V)
        zeta_x, zeta_y = zeta 
        grad_psi = CF((grad(psi)[0], grad(psi)[1]))
        dt_psi = grad(psi)[2]
        # bulk rhs
        #muhvec = CF((Jx + vWindX*rho, Jy + vWindY*rho, rho))
        div_zeta = CF(grad(zeta_x)[0] + grad(zeta_y)[1])


        # Auflösen??
        #muhvec = CF((Jx, Jy, rho))
        #qhvec = CF((Jxs, Jys, rhos))
        gvec = CF((gx, gy))
        gsvec = CF((gxs, gys))
        J_rhs = CF((r1* Jxs -Jx, r1*Jys-Jy))
        fbulk += (InnerProduct(grad_psi, J_rhs))*dx#(r1*qhvec - 1*muhvec)*grad(psi)*dx
        if bWind: 
            fbulk += (InnerProduct(CF((vWindX, vWindY)),grad_psi) + dt_psi - div_zeta)*(r1*rhos-rho)*dx
        else: 
            fbulk += (dt_psi - div_zeta)*(r1*rhos-rho)*dx 
        if bSource: 
            fbulk += (r1*sources - source)*psi*dx # irs_dx # source term
        #if bDif: # Immer aktiv?
        #fbulk += InnerProduct(zeta,(gvec - r1*gsvec))*dx # Als InnerProduct schreiben?
        fbulk += (zeta_x * (gvec[0] - r1*gsvec[0]) + zeta_y * (gvec[1] - r1*gsvec[1]))*dx
        if bInitialFinal: #TODO Theorie
            mesh = V.mesh
            fbulk += (-1)*rho0*psi.Trace()*ds(definedon=mesh.Boundaries("bottom")) # implements initial condition
            fbulk += 1*rho1*psi.Trace()*ds(definedon=mesh.Boundaries("top")) # implements final condition
        
        # ??? 
        fbulk.Assemble()
        newvec = GridFunction(V)
        gfuprod = GridFunction(V)
            
        newvec.vec.data = fbulk.vec
        gfuprod.vec.data = invaprod * newvec.vec
            
        return gfuprod.vec 
    else: 
        # Wenn nicht Diffusion, die Funktion ohne Diffusion
        return SolvePhi(V, r1, rho0, rho1, rho, Jx, Jy, rhos, Jxs, Jys, source, sources, invaprod, bWind, bSource, bInitialFinial, vWindX, vWindY)


def interpolatedPhi(phih, dphih):
    #TODO Erweitern um Theta ? Eigene Funktion schreiben
    dphih.components[0].Set(grad(phih)[0])
    dphih.components[1].Set(grad(phih)[1])
    dphih.components[2].Set(grad(phih)[2])

    return 0

def interpolatedTheta(theta, dtheta):
    dtheta.components[0].Set(grad(theta)[0])
    dtheta.components[1].Set(grad(theta)[1])
    dtheta.components[2].Set(grad(theta)[2])

    return 0


@safescope
def UpdatePhiTheta(r1, rho, dphih, phiL2, Jx, Jy, gx, gy, source, rhos, Jxs, Jys, gxs, gys, sources, bWind, bSource, bInverse, sensor_marker, alpha_f, alpha_inv, alpha_dif, vWindX, vWindY, gfInvDat, theta):

    bDif = theta is not None
    ### TODO Fallunterscheidungen mit Theta -> bDif implizit drin
    if bInverse:
        indNot = sensor_marker
    else:
        indNot = BitArray(rho.vec.size)
        indNot.Clear()
    #theta = 0 # Platzhalter
    if bDif:
        div_theta = grad(theta[0])[0] + grad(theta[1])[1]

    etarhos = dphih.components[2].vec.FV().NumPy() + 1/r1*rho.vec.FV().NumPy() #TODO ändern (alle etas)
    
    if bWind:
        etarhos += dphih.components[0].vec.FV().NumPy()*vWindX.vec.FV().NumPy() + dphih.components[1].vec.FV().NumPy()*vWindY.vec.FV().NumPy()
    # Hier noch eine Bedingung für Diffusion?
    if bDif:
        etarhos -= div_theta

    etaJxs = dphih.components[0].vec.FV().NumPy() + 1/r1*Jx.vec.FV().NumPy()
    etaJys = dphih.components[1].vec.FV().NumPy() + 1/r1*Jy.vec.FV().NumPy()
    # Bedingung für Diffusion
    if bDif:
        etagxs = -theta.components[0].vec.FV().NumPy() + 1/r1*gx.vec.FV().NumPy()
        etagys = -theta.components[1].vec.FV().NumPy() + 1/r1*gy.vec.FV().NumPy()
    if bSource:
        etafs = phiL2.vec.FV().NumPy() + 1/r1*source.vec.FV().NumPy()

        ## Check admissibility bulk.
        ###Updated
        if bDif:
            PosIndexG = etarhos + .5*(etaJxs**2 + etaJys**2 + etafs**2/alpha_f**2 + etagxs**2/alpha_dif**2 + etagys**2/alpha_dif**2) >0
        else: 
            PosIndexG = etarhos + .5*(etaJxs**2 + etaJys**2 + etafs**2/alpha_f**2) >0
        ## Update dual variables curve
        if np.any(PosIndexG)==True:

            eta_rho = etarhos[PosIndexG].copy()
            eta_Jx = etaJxs[PosIndexG].copy()
            eta_Jy = etaJys[PosIndexG].copy()
            eta_f = etafs[PosIndexG].copy()

            eta_J_nrm = np.sqrt(bgx**2+bgy**2)
            eta_f_nrm = np.abs(eta_f)
            # Solve the system using Newton's method
            #mu01d = nrm_b1d.copy() # Wieso hier nochmal die copy TODO!!!
            #nu01d = cg.copy()
            #kappa01d = nrm_d1d.copy()
            err1d = 1
            x_sol = eta_J_nrm.copy()
            y_sol = eta_f_nrm.copy()
            if bDif:
                eta_gx = etagxs[PosIndexG].copy()
                eta_gy = etagys[PosIndexG].copy()
                eta_g_nrm = np.sqrt(dgx**2+dgy**2)
                u_sol = eta_g_nrm.copy()

            """
            ag = etarhos[PosIndexG].copy()
            bgx = etaJxs[PosIndexG].copy()
            bgy = etaJys[PosIndexG].copy()
            cg = etafs[PosIndexG].copy()
            dgx = etagxs[PosIndexG].copy()
            dgy = etagys[PosIndexG].copy()

            nrm_b1d = (bgx**2+bgy**2)**0.5
            nrm_c1d = #TODO
            nrm_d1d = (dgx**2+dgy**2)**0.5 
            # Solve the system using Newton's method
            mu01d = nrm_b1d.copy() # Wieso hier nochmal die copy TODO!!!
            nu01d = cg.copy()
            kappa01d = nrm_d1d.copy()
            err1d = 1
            """
            for i in range(np.shape(ag)[0]):
                #TODO Fallunterscheidung notwendig? wieso nicht auch wenn y-Term == 0? Dann mehrere Unterscheidungen bei Neuerung? Vorerst keine Fallunterscheidung und direkt alle machen
                """
                if nrm_b1d[i]==0: # wenn x-Term 0
                    print("zero encountered")
                    nrm_b1d[i] = 1
                    mu01d[i] = 0
                    funcderivtest = lambda c: np.array([c**3 + 2*(alpha_f**2+alpha_f*ag[i])*c - 2*alpha_f**2*cg[i], 3*c**2 + 2*(alpha_f**2+alpha_f*ag[i])]) #Prüfe y Term, wenn x=0
                    solroottest = root(funcderivtest, nu01d[i], jac=True, method='hybr') # TODO Wieso das eine Argument nu01d?
                    nu01d[i] = solroottest.x[0]
                    print(solroottest.message)
                else:
                    funcderivparam2 = lambda c: funcderiv2(c[0], c[1], ag[i], cg[i], nrm_b1d[i], 1, alpha_f) # TODO Hier Newton mit allen Termen
                    zold = np.array([mu01d[i], nu01d[i]])
                    solrootcurve = root(funcderivparam2, zold, jac=True, method='hybr')
                    #print(solrootcurve.message)
                    znew = solrootcurve.x
                    mu01d[i] = znew[0]
                    nu01d[i] = znew[1]
                """
                #funcderivparam2 = lambda c: funcderiv2(c[0], c[1], ag[i], cg[i], nrm_b1d[i], 1, alpha_f) # TODO Hier Newton mit allen Termen
                if bDif:
                    funcderivparam2 = lambda c: funcderiv2_dif(c[0], c[1], c[2], eta_rho[i], eta_J_nrm[i], eta_f_nrm[i], eta_g_nrm, alpha_f, alpha_dif)

                    zold = np.array([x_sol[i], y_sol[i], u_sol[i]])
                else: 
                    funcderivparam2 = lambda c: funcderiv2(c[0], c[1], eta_rho[i], eta_J_nrm[i], eta_f_nrm[i], alpha_f) 
                    zold = np.array([x_sol[i], y_sol[i]])
                solrootcurve = root(funcderivparam2, zold, jac = True, method = "hybr")
                znew = solrootcurve.x
                x_sol[i] = znew[0]
                y_sol[i] = znew[1]
                if bDif:
                    u_sol[i] = znew[2]
            # Update
            if bDif:
                #x = norm Js / mu01d, y = norm fs / nu01d, u = norm gs/ nrm_d1d
                etarhos[PosIndexG] = -0.5*(x_sol**2 + y_sol**2/alpha_f**2 + u_sol**2/alpha_dif**2)
            else:
                etarhos[PosIndexG] = -0.5*(x_sol**2 + y:sol**2/alpha_f**2)

            etaJxs[PosIndexG] = x_sol * eta_Jx/eta_J_nrm
            etaJys[PosIndexG] = x_sol * eta_Jy/eta_J_nrm
            etafs[PosIndexG] = y_sol * eta_f/eta_f_nrm #oder lieber np.sign(eta_f)?
            
            if bDif:
                etagxs[PosIndexG] = u_sol * eta_gx/eta_g_nrm
                etagys[PosIndexG] = u_sol * eta_gy/eta_g_nrm
        

        #sources.vec.data[~indNot] = etafs[~indNot]
        sources.vec.data = etafs #TODO Wieso wird hier nur sources.vec.data gemacht und nicht auch bei den anderen?
        
    else: # Case without source
        PosIndexB = etarhos + .5*(etaJxs**2 + etaJys**2) > 0
        if bDif:
            PosIndexB = etarhos + .5*(etaJxs**2 + etaJys**2 + etagxs**2/alpha_dif**2 + etagys**2/alpha_dif**2) >0
        iter0 = 0
        
        ## Update dual variables bulk
        if np.any(PosIndexB)==True:
            
            # print("there are non admissible in bulk")
            """
            a = etarhos[PosIndexB].copy()
            b1 = etaJxs[PosIndexB].copy()
            b2 = etaJys[PosIndexB].copy()
            nrm_b = (b1**2+b2**2)**0.5
            """
            eta_rho = etarhos[PosIndexB].copy()
            eta_Jx = etaJxs[PosIndexB].copy()
            eta_Jy = etaJys[PosIndexB].copy()
            eta_J_nrm = (eta_Jx**2 + eta_Jy**2)**0.5 #vielleicht np.sqrt?
            x_sol = eta_J_nrm.copy()
            if bDif:
                eta_gx = etagxs[PosIndexB].copy()
                eta_gy = etagys[PosIndexB].copy()
                eta_g_nrm = (eta_gx**2 + eta_gy**2)**0.5
                u_sol = eta_g_nrm.copy()
                for i in range(len(eta_rho)):

                    funcderivparam2 = lambda c: funcderiv_no_source_dif(c[0], c[1], eta_rho[i], eta_J_nrm[i], eta_g_nrm[i], alpha_dif)
                    zold = np.array([x_sol[i], u_sol[i]])
                    sol = root(funcderivparam2, zold, jac=True, method="hybr")
                    #z_new = solrootcurve.x
                    x_sol[i] = sol.x[0]
                    u_sol[i] = sol.x[1]
            else:

            # FIND (unique) largest real root of cubic poly
            # x^3 + 2(alpha+1) x - 2 |beta| = 0
            # Use Newton's method.
            #mu0 = nrm_b.copy()

                err = 1
                while err > 1e-6 and iter0<20:
                    iter0 += 1
                    #mu1 = mu0 - (mu0**3 + 2*(a+1)*mu0 - 2*nrm_b)/(3*mu0**2+2*(a+1)) # x = x - f(x)/f'(x)
                        x_new = x_sol - (x_sol**3 + 2*(1+eta_rho)*x_sol - 2*eta_J_nrm) / (3*x_sol**2 + 2*(1+ eta_rho))
                    #err = abs(mu1-mu0).max()
                    #mu0 = mu1
                    err = abs(x_new - x_sol).max()
                    x_sol = x_new

            if bDif:
                etarhos[PosIndexB] = -0.5*(x_sol**2 + u_sol**2/alpha_dif**2)
                etagxs[PosIndexB] = u_sol * eta_gx/eta_g_nrm
                etagys[PosIndexB] = u_sol * eta_gy/eta_g_nrm
            else:
                etarhos[PosIndexB] = -0.5*x_sol**2
            etaJxs[PosIndexB] = x_sol * eta_Jx/eta_J_nrm
            etaJys[PosIndexB] = x_sol * eta_Jy/eta_J_nrm

            #etarhos[PosIndexB] = -0.5*mu0**2
            #etaJxs[PosIndexB] = mu0*b1/nrm_b
            #etaJys[PosIndexB] = mu0*b2/nrm_b
   
    
    rhos.vec.data = etarhos
    Jxs.vec.data = etaJxs
    Jys.vec.data = etaJys
    if bDif:
        gxs.vec.data = etagxs
        gys.vec.data = etagys

    
    if bInverse:
        ### TODO hier auch nochmal mit Diffusion
        for i in range(len(indNot)):
            if indNot[i]:
                ### TODO Wieso nicht einfach ausschließlich in der Funktion und dort dann die Fallunterscheidung? Ist dann ja doppelt
                if bSource:
                    #TODO Fallunterscheidung bDif
                    if bDif:
                        funcderivInv = lambda c: funcderivInv(c[0], c[1], c[2], alpha_inv, alpha_f, r1, etarhos[i], etaJxs[i], etaJys[i], gfInvDat.vec.FV().NumPy()[i], bSource=True, fs = c[3], etafs = etafs[i], bDif=True, alpha_dif = alpha_dif, gxs = gxs[i], gys = gys[i], etagxs = etagxs[i], etagys = etagys[i])
                        zold = np.array([rhos.vec.FV().NumPy()[i], Jxs.vec.FV().NumPy()[i], Jys.vec.FV().NumPy()[i], sources.vec.FV().NumPy()[i], gxs.vec.FV().NumPy()[i], gys.vec.FV().NumPy()[i]])
                    else:
                        funcderivInv2 = lambda c: funcderivInv(c[0], c[1], c[2], alpha_inv, alpha_f, r1, etarhos[i], etaJxs[i], etaJys[i],  gfInvDat.vec.FV().NumPy()[i], bSource=True, fs = c[3], etafs = etafs[i])
                    
                        zold = np.array([rhos.vec.FV().NumPy()[i], Jxs.vec.FV().NumPy()[i], Jys.vec.FV().NumPy()[i], gxs.vec.FV().NumPy()[i], gys.vec.FV().NumPy()[i]])                
                else: #No Source
                    #TODO Fallunterscheidung bDif
                    if bDif:
                        funcderivInv = lambda c: funcderivInv(c[0], c[1], c[2], alpha_inv, alpha_f, r1, etarhos[i], etaJxs[i], etaJys[i], gfInvDat.vec.FV().NumPy()[i],bDif=True, alpha_dif = alpha_dif, gxs = gxs[i], gys = gys[i], etagxs = etagxs[i], etagys = etagys[i])
                        zold = np.array([rhos.vec.FV().NumPy()[i], Jxs.vec.FV().NumPy()[i], Jys.vec.FV().NumPy()[i], gxs.vec.FV().NumPy()[i], gys.vec.FV().NumPy()[i]])
                    else:
                        funcderivInv2 = lambda c: funcderivInv(c[0], c[1], c[2], alpha_inv, alpha_f, r1, etarhos[i], etaJxs[i], etaJys[i], gfInvDat.vec.FV().NumPy()[i]) 

                        zold = np.array([rhos.vec.FV().NumPy()[i], Jxs.vec.FV().NumPy()[i], Jys.vec.FV().NumPy()[i]])
                
                solrootcurve = root(funcderivInv2, zold, jac=True, method='hybr')        
                
                rhos.vec.data[i] = solrootcurve.x[0]
                Jxs.vec.data[i] = solrootcurve.x[1]
                Jys.vec.data[i] = solrootcurve.x[2]
                if bSource and not bDif:
                    sources.vec.data[i] = solrootcurve.x[3]
                elif bDif and not bSource:
                    gxs.vec.data[i] = solrootcurve.x[3]
                    gys.vec.data[i] = solrootcurve.x[4]
                elif bDif and bSource:
                    sources.vec.data[i] = solrootcurve.x[3]
                    gxs.vec.data[i] = solrootcurve.x[4]
                    gys.vec.data[i] = solrootcurve.x[5]
                else:
                    pass

            
    return 0



def ALG2_new(V, W, r1, rho, rho0, rho1, Jx, Jy, gx, gy, rhos, Jxs, Jys, gxs, gys, source, sources,
               ndof3d, invaprod, bWind, bSource, bInitialFinal, bDif, vWindX, vWindY, scaler1=1, iterMax=100, tol=1e-6):
    #TODO Alles anpassen
    """
    bDif unnötig?; fehlt gx,gy,gxs,gys, Definition theta 
    """ 
    pdb.set_trace()
    
    if bDif:
        V = FESpace([V,(V*V)]) 
        phiTheta = GridFunction(V)
        phih = phiTheta.components[0]
        thetah = phiTheta.components[1]
        phiL2 = GridFunction(W)
        dphih = GridFunction(W*W*W)
        thetaL2 = GridFunction(W*W)
        divtheta = GridFunction(W)

    else: 
        phiL2 = GridFunction(W)
        
    iter0 = 0
    duality_gaps = [] 
    lam_avg = 0

    err0 = 1
    tau = 1 #?

    with TaskManager():
        while iter0 < iterMax+1:
            # TODO anpassen mit theta
            ## STEP A: Solve for new (phi,phi1d) and theta

            if bInitialFinial:
                # TODO Wieso klappt das mit rho0cf als nicht globale Variable?
                phiTheta.vec.data = SolvePhiTheta_new(V, r1, rho0cf, rho1cf, rho, Jx, Jy, gx, gy, rhos, Jxs, Jys, gxs, gys, source, sources, invaprod, bWind, bSource, bInitialFinal, vWindX = 0, vWindY = 0) #bDif steckt dabei in der Dimension von V drin
            else:
                phiTheta.vec.data = SolvePhiTheta_new(X, r1, rho0, rho1, rho, Jx, Jy, gx, gy, rhos, Jxs, Jys, gxs, gys, source, sources, invaprod, bWind, bSource, bInitialFinal, vWindX = 0, vWindY = 0)                

            if bDif:
                phih = phiTheta.components[0]
                thetah = phiTheta.components[1]
        
                theta_x = thetah.components[0]
                theta_y = thetah.components[1]
            phiL2.Set(phih)       

            # Das unverändert lassen, aber in if Bedingung packen? Nochmal genau anschauen TODO
            lam_avg = lam_avg + tau*r1*(Integrate(rho1, mesh, definedon=mesh.Boundaries("top")) - Integrate(rho0, mesh, definedon=mesh.Boundaries("bottom"))) 
            if bSource:
                lam_avg += tau*r1*bSource*Integrate(source, mesh)
            # Fallunterscheidung mit Diffusion noch? TODO

            
            ## STEP B: solve pointwise nonlinear eqn (USE NEWTON)
            #pdb.set_trace()
            interpolatedPhi(phih, dphih)            
            if bDif:
                interpolatedTheta(thetah, dtheta) #TODO Argumente prüfen

            UpdatePhi(r1, rho, dphih, phiL2, Jx, Jy, source, rhos, Jxs, Jys, sources, bWind, bSource, bInverse, sensor_marker, alpha_f, alpha_inv, vWindX, vWindY, gfInvDat)

            ## STEPC: 
            rho.vec.FV().NumPy()[:] += tau*r1*(dphih.components[2].vec.FV().NumPy() + vWindX.vec.FV().NumPy()*dphih.components[0].vec.FV().NumPy() + vWindY.vec.FV().NumPy()*dphih.components[1].vec.FV().NumPy() - rhos.vec.FV().NumPy())
            rho.vec.FV().NumPy()[:] += tau*r1*(dphih.components[2].vec.FV().NumPy() + vWindX.vec.FV().NumPy()*dphih.components[0].vec.FV().NumPy() + vWindY.vec.FV().NumPy()*dphih.components[1].vec.FV().NumPy() - rhos.vec.FV().NumPy()) # Hier noch in die Klammer Nabla * Lambda
              
            Jx.vec.data += tau*r1*(dphih.components[0].vec - Jxs.vec)#(dphih.components[0].vec.FV().NumPy() - Jxs.vec.FV().NumPy())
            Jy.vec.data += tau*r1*(dphih.components[1].vec - Jys.vec)#(dphih.components[1].vec.FV().NumPy() - Jys.vec.FV().NumPy())

            #TODO Das in if packen (was ist die Condition?)
            if bDif:
                gx.vec.data += tau*r1*(-theta_x - gxs.vec)
                gy.vec.data += tau*r1*(-theta_y - gys.vec)

            if not bInitialFinial: # update initial- and final data
                #TODO auch in der Theorie
                rho0.vec.FV().NumPy()[:] += tau*r1*(-phiL2.vec.FV().NumPy() + lam_avg)
                rho1.vec.FV().NumPy()[:] += tau*r1*(phiL2.vec.FV().NumPy() - lam_avg)
            
            if bSource:
                source.vec.FV().NumPy()[:] += tau*r1*(phiL2.vec.FV().NumPy() - sources.vec.FV().NumPy())# TODO Lambda hinzufügen? - lam_avg) 

            if iter0%iterFlag==0:
                err0 = max(np.abs(dphih.components[2].vec.FV().NumPy() + dphih.components[0].vec.FV().NumPy()*vWindX.vec.FV().NumPy() + dphih.components[1].vec.FV().NumPy()*vWindY.vec.FV().NumPy() - rhos.vec.FV().NumPy()).max(),np.abs(dphih.components[0].vec.FV().NumPy() - Jxs.vec.FV().NumPy()).max(),np.abs(dphih.components[1].vec.FV().NumPy() - Jys.vec.FV().NumPy()).max())
                if bSource:
                    err0 = max(err0, np.abs(phiL2.vec.FV().NumPy() - sources.vec.FV().NumPy()).max())
                
                rhs_gap = (Integrate(rho1, mesh, definedon=mesh.Boundaries("top")) - Integrate(rho0, mesh, definedon=mesh.Boundaries("bottom")) + Integrate(source, mesh))

                print("iter = ", iter0, "/", maxiter, ", rhs integral = ", rhs_gap , "lam_avg = ", lam_avg, " duality gap = ", err0, " gap in f^* = ", np.abs(phiL2.vec.FV().NumPy() - sources.vec.FV().NumPy()).max(), "source integral = ", Integrate(source, mesh), "Gap in rho", np.abs(dphih.components[2].vec.FV().NumPy() - rhos.vec.FV().NumPy() + vWindX.vec.FV().NumPy()*dphih.components[0].vec.FV().NumPy() + vWindY.vec.FV().NumPy()*dphih.components[1].vec.FV().NumPy()).max(), end="\r")
                                
                duality_gaps.append(err0)
                
                if err0 < tol: # FIXME 
                    print("converged")
                    # return anpassen
                    return rho, rho0, rho1, phih, Jx, Jy, Jxs, Jys, source, sources, iter0, err0, duality_gaps

            iter0 += 1
        print("max number of iterations reached.") 
    # return anpassen
    return rho, rho0, rho1, phih, Jx, Jy, Jxs, Jys, source, sources, iter0, err0, duality_gaps


if __name__ == "__main__":
    bWind = True
    bSource = True
    bInitialFinial = True
    bInverse = True
    bDif = True

    alpha_f = 1
    alpha_inv = 1
    alpha_wind = 1 #???
    alpha_dif = 1

    vtk_output = False
    # Create 2d and 1d meshes
    otmesh = OTMesh(h=0.05, nz = 40, holes = False)

    mesh = otmesh.mesh3d
    mesh2d = otmesh.mesh2d
    nz = otmesh.nz

    order=1
    orderm=1
    orderI=1
    ### TODO auf die Randbedingungen des Gradienten achten!
    V = H1(mesh, order=orderm) ## FIXME: Dirichlet conditions? -> do not change things
    W = L2(mesh, order=0) #IntegrationRuleSpace(mesh, order=orderm) ## FIXME: Dirichlet conditions? -> do not change things
    ndof3d = W.ndof

    W = IntegrationRuleSpace(mesh, order=orderm) ## FIXME: Dirichlet conditions? -> do not change things
    irs_dx = dx(intrules=W.GetIntegrationRules())

    np_xy = 30
    sensor_points_xy = [ (0.7,0.25+0.5*i/np_xy) for i in range(np_xy) ] # Must have length nz

    sensor_points = []
    for i in range(2*nz):
        for p in sensor_points_xy:
            sensor_points.append([p[0], p[1], i/(2*nz)])
            sensor_points.append([p[0]-otmesh.h/2, p[1], i/(2*nz)])
            sensor_points.append([p[0]+otmesh.h/2, p[1], i/(2*nz)])

    sensor_marker = BitArray(mesh.ne)  # mesh.ne = number of elements
    sensor_marker.Clear()

    for pt in sensor_points:
        el_nr = mesh(pt[0],pt[1],pt[2]) # mesh2d(pt[0],pt[1]) = 2d point
        sensor_marker.Set(el_nr.nr) # with i the number of the element
        
    # Set up inverse problem data
    gfInvDat = GridFunction(W)
    gfInvDat.Set(IfPos(z-.1, IfPos(.9-z,z*(1-z),0),0), definedonelements=sensor_marker)
    gfInvDat.vec.data /= Integrate(gfInvDat, mesh)    
    gfInvDat.vec.data *= .5

    #DrawAnimate2d(gfInvDat, V, mesh2d, nz)
    gfWind = SolveStokes(mesh2d, alpha_wind=alpha_wind, mu_val=1)

    tol0=1e-10
    iterFlag=10
    sol = "direct"
    eps0=0

    n = specialcf.normal(mesh.dim)
    phiEx = 0.5*(x+y)-0.25*y
    phi, theta, rho, Jx, Jy, f, gx, gy, rhos, Jxs, Jys, fs, gxs, gys, dphi, rho0, rho1 = CreateVariables_new(V, W)
    lam_avg = 0 # Lagrange multiplier for RHS of Poisson to integrate to zero

    source = GridFunction(W)
    sources = GridFunction(W)

    vWindX = GridFunction(W)
    vWindY = GridFunction(W)
    # Set up "wind" vectorfield
    if bWind:
        vWindX.Set(gfWind.components[0])
        vWindY.Set(gfWind.components[1])
    else:
        vWindX.Set(0*x)
        vWindY.Set(0*x)

    sigma = .1

    tt3 =(x-(.35*(1-z)+.7*z))**2+(y-(.35*(1-z)+.7*z))**2
    rhoEx = 0*exp(-(1/2/sigma**2)*tt3) + 0*1
    
    if bInitialFinial:
        rho0cf = rhoEx
        rho1cf = rhoEx

    else:
        rho0 = GridFunction(W)
        rho1 = GridFunction(W)

    mEx = CF((.5*rhoEx, .5*rhoEx))

    # initial data
    rho.Interpolate(rhoEx) # .Set produces neg. values -> why?
    Jx.Set(0*mEx[0]) 
    Jy.Set(0*mEx[1]) 
    rhos.Set(0*phiEx.Diff(z))
    Jxs.Set(0*phiEx.Diff(x))
    Jys.Set(0*phiEx.Diff(y))
    source.Set(0*x)
    if not bInitialFinial:
        rho0.Set(0*x)
        rho1.Set(0*x)


    r1 = 1
    sr1 = 1
    toldiff = 1e-4
    tolcvg = 1e-5
    maxiter = 500 # 50000

    invaprod = assembleBilinearForm_new(V, r1, bWind, bSource, vWindX, vWindY, tol=toldiff)
    # was genau ist das dif argmuent?
    dif = 0
    rho, rho0, rho1, phih, Jx, Jy, Jxs, Jys, source, Thesources, iter0, err0, duality_gaps = ALG2_new(V, W, r1, rho, rho0, rho1, Jx, Jy, gx, gy, source, sources, dif, rhos, Jxs, Jys, gxs, gys, ndof3d, invaprod, bWind, bSource, bInitialFinial, bDif, vWindX, vWindY, scaler1=sr1, iterMax=maxiter, tol=tolcvg)
