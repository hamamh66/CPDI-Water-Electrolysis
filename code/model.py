"""Polarization models, calibration and levelized-cost operating-point selection."""
import numpy as np
from scipy.optimize import least_squares, brentq, minimize_scalar
F=96485.332; R_GAS=8.314462; E_TN=1.481
import os; HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'..','data','')
def E_rev(T): return 1.50342-9.956e-4*T+2.5e-7*T**2
def V_cell(j,p,T):
    """Lumped two-electrode cell: p=(log j0, b, R). b is the asinh prefactor in V; R in ohm cm2."""
    j0=np.exp(p[0]); return E_rev(T)+p[1]*np.arcsinh(j/(2*j0))+p[2]*j
def dVdj(j,p,T):
    j0=np.exp(p[0]); return p[1]/np.sqrt(j**2+4*j0**2)+p[2]
def fit(j,V,T,p0=(np.log(1e-4),0.05,0.2)):
    r=least_squares(lambda p:V_cell(j,p,T)-V,p0,bounds=([np.log(1e-12),1e-3,0.0],[np.log(1.0),0.3,5.0]),x_scale=[1,0.01,0.1])
    return r.x
def rmse(j,V,T,p): return float(np.sqrt(np.mean((V_cell(j,p,T)-V)**2)))
def load():
    D={}
    a=np.loadtxt(DATA+'pem_garcia_valverde_353K.csv',delimiter=','); D['PEM']=dict(T=353.15,j=a[:,0],V=a[:,1])
    for k,f in [('PEM_343','pem_garcia_valverde_343K.csv'),('PEM_333','pem_garcia_valverde_333K.csv')]:
        a=np.loadtxt(DATA+f,delimiter=','); D[k]=dict(T=float(f[-8:-5])+0.15,j=a[:,0],V=a[:,1])
    a=np.loadtxt(DATA+'alk_zerogap_raneyNi_80C.csv',delimiter=',',comments='#'); D['AZG_days']=dict(T=353.15,j=a[:,0]/1000/25.0,V=a[:,1:])
    rows=[l.strip().split(',') for l in open(DATA+'aem_nife_comm.csv') if not l.startswith('#')]
    v=np.array([float(r[0]) for r in rows]); j50=np.array([float(r[1]) for r in rows]); j80=np.array([float(r[2]) if r[2] else np.nan for r in rows])
    D['AEM50']=dict(T=323.15,j=j50,V=v); m=~np.isnan(j80); D['AEM80']=dict(T=353.15,j=j80[m],V=v[m])
    return D
# coverage and finite-gap alkaline (derived archetype)
K_VB=0.023*(1e4)**0.3   # Vogt-Balzer theta=0.023*i^0.3 with i in A/m2, converted to j in A/cm2
def theta_vb(j,k=K_VB,n=0.3): return np.minimum(k*np.asarray(j,float)**n,0.95)
def V_fg(j,p,T,gap_asr=0.3,k_eps=0.25,k=K_VB,n=0.3):
    """zero-gap-calibrated electrodes plus an electrolyte gap; area blocking through j_eff; Bruggeman voidage in the gap only."""
    j=np.asarray(j,float); th=theta_vb(j,k,n); jeff=j/(1-th); j0=np.exp(p[0]); eps=np.minimum(k_eps*j,0.6)
    return E_rev(T)+p[1]*np.arcsinh(jeff/(2*j0))+p[2]*j+gap_asr*j/(1-eps)**1.5
def eta_F(j,jx): return j/(j+jx)
def U(j,Vfun,Gamma,jx):
    """cost-equivalent voltage (V): LCOH = 26.59*c_el*U"""
    return (Vfun(j)+Gamma/j)/eta_F(j,jx)
def optimum(Vfun,Gamma,jx,jlo,jhi,cons=()):
    """minimise U on [jlo,jhi] subject to cons: list of (name, g) with g(j)<=0 feasible and g increasing in j."""
    ub=jhi; active='domain'
    for name,g in cons:
        if g(jlo)>0: return dict(j=np.nan,binding='infeasible')
        if g(ub)>0:
            ub=brentq(g,jlo,ub); active=name
    grid=np.geomspace(jlo,ub,4000); u=U(grid,Vfun,Gamma,jx); i=int(np.argmin(u))
    a=grid[max(i-1,0)]; b=grid[min(i+1,len(grid)-1)]
    r=minimize_scalar(lambda x:U(x,Vfun,Gamma,jx),bounds=(a,b),method='bounded',options=dict(xatol=1e-10))
    j=r.x; binding='none'
    if abs(j-ub)<1e-6*ub or i==len(grid)-1: j=ub; binding=active
    if i==0: j=jlo; binding='lower domain'
    # unconstrained (within domain) optimum for the cost of the constraint
    return dict(j=float(j),binding=binding,U=float(U(j,Vfun,Gamma,jx)),V=float(Vfun(j)),ub=float(ub))
