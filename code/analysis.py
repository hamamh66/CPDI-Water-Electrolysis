from model import *
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=os.path.join(HERE,'..','output',''); FIG=os.path.join(HERE,'..','figures','')
os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({'pdf.fonttype':42,'font.size':8.5,'axes.grid':True,'grid.alpha':0.3,'font.family':'serif','mathtext.fontset':'dejavuserif','legend.fontsize':7.2})
rng=np.random.default_rng(42); D=load(); res={}
az=D['AZG_days']; D['AZG']=dict(T=az['T'],j=az['j'],V=az['V'][:,2:].mean(1),sd=az['V'][:,2:].std(1,ddof=1))
ARCH=[('AZG','Alkaline, zero gap',353.15),('PEM','PEM',353.15),('AEM80','AEM, 80 $^\\circ$C',353.15),('AEM50','AEM, 50 $^\\circ$C',323.15)]
COL={'AFG':'#1b6ca8','AZG':'#2a9d8f','PEM':'#c1121f','AEM80':'#7b2cbf','AEM50':'#b5838d'}
JX=None
# Faradaic efficiency from the AEM dataset (zenodo.15125443, Fig. S13)
fe_j=np.array([0.25,0.5,0.75,1.0,1.25,1.5]); fe=np.array([98.4,99.8,99.6,100.0,100.1,99.9])/100
jx=float(least_squares(lambda x:fe_j/(fe_j+x[0])-fe,[0.003],bounds=([0],[0.1])).x[0]); res['jx']=jx; JX=jx
P={}; BOOT={}; cal={}
for key,name,T in ARCH:
    d=D[key]; o=np.argsort(d['j']); j=d['j'][o]; V=d['V'][o]
    jf,Vf,jv,Vv=j[0::2],V[0::2],j[1::2],V[1::2]
    pf=fit(jf,Vf,T); pa=fit(j,V,T)
    m=j<=0.6*j.max(); pe=fit(j[m],V[m],T)
    # residual bootstrap on the full-data fit
    r=V-V_cell(j,pa,T); B=[]
    for _ in range(1000):
        B.append(fit(j,V_cell(j,pa,T)+rng.choice(r,len(r),replace=True),T,p0=pa))
    B=np.array(B); P[key]=pa; BOOT[key]=B
    cal[key]=dict(name=name,T=T,n=len(j),jmin=float(j.min()),jmax=float(j.max()),j0=float(np.exp(pa[0])),b=float(pa[1]),R=float(pa[2]),
        j0_ci=[float(np.exp(np.percentile(B[:,0],5))),float(np.exp(np.percentile(B[:,0],95)))],b_ci=[float(np.percentile(B[:,1],5)),float(np.percentile(B[:,1],95))],R_ci=[float(np.percentile(B[:,2],5)),float(np.percentile(B[:,2],95))],
        corr=np.corrcoef(B.T).round(3).tolist(),rmse_all=1e3*rmse(j,V,T,pa),rmse_fit=1e3*rmse(jf,Vf,T,pf),rmse_holdout=1e3*rmse(jv,Vv,T,pf),
        rmse_extrap=1e3*rmse(j[~m],V[~m],T,pe),extrap_from=float(j[m].max()),maxabs_holdout=1e3*float(np.max(np.abs(V_cell(jv,pf,T)-Vv))))
cal['AZG']['day_sd_mV']=float(1e3*D['AZG']['sd'].mean())
# day-to-day transfer for the zero-gap cell: fit day 4, predict days 5 and 6
p4=fit(az['j'],az['V'][:,3],353.15); cal['AZG']['rmse_day4_to_5']=1e3*rmse(az['j'],az['V'][:,4],353.15,p4); cal['AZG']['rmse_day4_to_6']=1e3*rmse(az['j'],az['V'][:,5],353.15,p4)
# temperature transfer check for PEM: parameters at 80 C applied to 70 and 60 C data (no refit)
for k in ['PEM_343','PEM_333']:
    cal['PEM']['rmse_transfer_'+k]=1e3*rmse(D[k]['j'],D[k]['V'],D[k]['T'],P['PEM'])
res['calibration']=cal
# ---------------- operating-point program
JLO,JHI=0.02,6.0; E_MIN=None
def Vf_of(key,p=None,**kw):
    if key=='AFG':
        pp=P['AZG'] if p is None else p; return lambda j: V_fg(j,pp,353.15,**kw)
    pp=P[key] if p is None else p; T=dict(ARCH_T)[key]; return lambda j: V_cell(j,pp,T)
ARCH_T=[(k,T) for k,_,T in ARCH]
def solve(key,Gamma,p=None,theta_max=None,e_min=E_MIN,jx=None,fgkw={},m_deg=0.0,jref=1.0):
    jx=JX if jx is None else jx
    Vf=Vf_of(key,p,**fgkw); cons=[] if e_min is None else [('efficiency',lambda j:Vf(j)-E_TN/e_min)]
    if key=='AFG' and theta_max is not None:
        k=fgkw.get('k',K_VB); n=fgkw.get('n',0.3); cons.append(('coverage',lambda j:theta_vb(j,k,n)-theta_max))
    if m_deg>0:
        Vd=lambda j: Vf(j)+Gamma*((j/jref)**m_deg-1)/j   # capital term scaled by (j/jref)^m
        o=optimum(Vd,Gamma,jx,JLO,JHI,cons); o['V']=float(Vf(o['j'])); return o
    return optimum(Vf,Gamma,jx,JLO,JHI,cons)
KEYS=['AFG','AZG','PEM','AEM80','AEM50']; NAMES={'AFG':'Alkaline, finite gap (derived)','AZG':'Alkaline, zero gap','PEM':'PEM','AEM80':'AEM, 80 C','AEM50':'AEM, 50 C'}
JMAXDATA={'AFG':cal['AZG']['jmax'],'AZG':cal['AZG']['jmax'],'PEM':cal['PEM']['jmax'],'AEM80':cal['AEM80']['jmax'],'AEM50':cal['AEM50']['jmax']}
G=np.geomspace(0.01,3,61); TH0=0.5
curves={}
for key in KEYS:
    js=[];bd=[];us=[];vs=[]
    for g in G:
        o=solve(key,g,theta_max=TH0); js.append(o['j']);bd.append(o['binding']);us.append(o['U']);vs.append(o['V'])
    # bootstrap band
    bk='AZG' if key=='AFG' else key; Bs=BOOT[bk][:300]
    band=np.array([[solve(key,g,p=pb,theta_max=TH0)['j'] for g in G[::4]] for pb in Bs])
    curves[key]=dict(G=G.tolist(),j=js,binding=bd,U=us,V=vs,Gb=G[::4].tolist(),lo=np.percentile(band,5,0).tolist(),hi=np.percentile(band,95,0).tolist())
res['curves']=curves
# representative table
GT=[0.05,0.2,0.5,1.0,2.0]; tab=[]
for key in KEYS:
    bk='AZG' if key=='AFG' else key
    for g in GT:
        o=solve(key,g,theta_max=TH0); b,Rr=(P[bk][1],P[bk][2]+(0.3 if key=='AFG' else 0))
        jb=np.array([solve(key,g,p=pb,theta_max=TH0)['j'] for pb in BOOT[bk][:500]])
        th=float(theta_vb(o['j'])) if key=='AFG' else None
        tab.append(dict(key=key,Gamma=g,j=o['j'],V=o['V'],etaE=E_TN/o['V'],U=o['U'],binding=o['binding'],lo=float(np.percentile(jb,5)),hi=float(np.percentile(jb,95)),
            closed=float((np.sqrt(b*b+4*Rr*g)-b)/(2*Rr)),extrapolated=bool(o['j']>JMAXDATA[key]),theta=th,kWh_per_kg=26.59*o['V']/eta_F(o['j'],JX)))
res['table']=tab
# efficiency-floor sweep: Gamma at which the floor starts to bind
eff={}
for key in KEYS:
    eff[key]={}
    for em in [0.62,0.70,0.75,0.80]:
        Vf=Vf_of(key); 
        try: jE=brentq(lambda j:Vf(j)-E_TN/em,JLO,JHI)
        except ValueError: jE=None
        gc=None
        if jE is not None:
            f=lambda lg: solve(key,10**lg,e_min=None,theta_max=None)['j']-jE
            gc=float(10**brentq(f,-2.5,2.5)) if f(-2.5)<0<f(2.5) else None
        eff[key][str(em)]=dict(jE=jE,Gamma_c=gc)
res['eff_floor']=eff
# finite-gap: coverage ceiling sweep and regime map
THS=np.linspace(0.20,0.60,41); GG=np.geomspace(0.01,3,121); reg=np.zeros((len(THS),len(GG)),int); jmap=np.zeros_like(reg,float)
code={'none':0,'coverage':1,'efficiency':2,'domain':3,'lower domain':4}
for a,th in enumerate(THS):
    for b_,g in enumerate(GG):
        o=solve('AFG',g,theta_max=th); reg[a,b_]=code[o['binding']]; jmap[a,b_]=o['j']
Vf=Vf_of('AFG'); gcrit=[]
for th in THS:
    jt=(th/K_VB)**(1/0.3); f=lambda lg: solve('AFG',10**lg,theta_max=None)['j']-jt
    gcrit.append(float(10**brentq(f,-3,3)) if f(-3)<0<f(3) else None)
res['afg_cover']=dict(theta=THS.tolist(),j_theta=((THS/K_VB)**(1/0.3)).tolist(),Gamma_c=gcrit)
# cost of the coverage ceiling (relative increase of U) at selected (theta_max, Gamma)
pen=[]
for th in [0.25,0.30,0.35,0.40,0.50]:
    for g in [0.2,0.5,1.0,2.0]:
        oc=solve('AFG',g,theta_max=th); ou=solve('AFG',g,theta_max=None)
        pen.append(dict(theta_max=th,Gamma=g,j_c=oc['j'],j_u=ou['j'],binding=oc['binding'],dU_pct=100*(oc['U']/ou['U']-1)))
res['afg_penalty']=pen
# sensitivity of the finite-gap optimum to derived-archetype assumptions at Gamma=0.5
sens=[]
for nm,kw in [('reference',{}),('gap ASR 0.1',dict(gap_asr=0.1)),('gap ASR 0.5',dict(gap_asr=0.5)),('k_eps 0',dict(k_eps=0.0)),('k_eps 0.45',dict(k_eps=0.45)),
              ('k_theta x0.5',dict(k=0.5*K_VB)),('k_theta x2',dict(k=2*K_VB)),('k_theta 0.75 (former prior)',dict(k=0.75)),('n_theta 0.2',dict(n=0.2)),('n_theta 0.4',dict(n=0.4))]:
    for g in [0.2,0.5,2.0]:
        o=solve('AFG',g,theta_max=TH0,fgkw=kw); sens.append(dict(case=nm,Gamma=g,j=o['j'],binding=o['binding'],U=o['U'],theta=float(theta_vb(o['j'],kw.get('k',K_VB),kw.get('n',0.3)))))
res['afg_sens']=sens
res['K_VB']=K_VB; res['j_theta_05_published']=float((0.5/K_VB)**(1/0.3)); res['j_theta_05_former']=float((0.5/0.75)**(1/0.3))
# crossover and degradation scenarios
sc=[]
for key in KEYS:
    for g in [0.2,1.0]:
        for jxv in [0.0,JX,0.01,0.02]:
            o=solve(key,g,theta_max=TH0,jx=jxv); sc.append(dict(key=key,Gamma=g,kind='jx',val=jxv,j=o['j'],binding=o['binding']))
        for md in [0.0,0.25,0.5]:
            o=solve(key,g,theta_max=TH0,m_deg=md,jref=1.0); sc.append(dict(key=key,Gamma=g,kind='m_deg',val=md,j=o['j'],binding=o['binding']))
res['scenarios']=sc
# joint perturbation: bootstrap parameters x lognormal Gamma (CV 0.3) at nominal Gamma
mc=[]
for key in KEYS:
    bk='AZG' if key=='AFG' else key
    for g0 in [0.2,1.0]:
        c=0.3; s=np.sqrt(np.log(1+c*c)); gs=g0*np.exp(rng.normal(-0.5*s*s,s,1000)); Bs=BOOT[bk]
        out=[solve(key,gs[i],p=Bs[i],theta_max=TH0) for i in range(1000)]
        jj=np.array([o['j'] for o in out]); bb=[o['binding'] for o in out]
        mc.append(dict(key=key,Gamma=g0,p5=float(np.percentile(jj,5)),p50=float(np.percentile(jj,50)),p95=float(np.percentile(jj,95)),
            freq={k:bb.count(k)/1000 for k in set(bb)},n_invalid=int(np.isnan(jj).sum()),frac_extrap=float(np.mean(jj>JMAXDATA[key]))))
res['mc']=mc
# valid domain: cost ratio at which the optimum reaches the upper end of the calibrated range
vd={}
for key in KEYS:
    f=lambda lg: solve(key,10**lg,theta_max=None)['j']-JMAXDATA[key]
    vd[key]=dict(jmin=cal['AZG' if key=='AFG' else key]['jmin'],jmax=JMAXDATA[key],Gamma_data=float(10**brentq(f,-3,2)),n_table_extrap=int(sum(t['extrapolated'] for t in tab if t['key']==key)))
res['valid_domain']=vd
json.dump(res,open(OUT+'results.json','w'),indent=1,default=float)
np.savez(OUT+'regime_map.npz',THS=THS,GG=GG,reg=reg,jmap=jmap)
# ================= figures
def save(n): plt.savefig(FIG+n+'.pdf',bbox_inches='tight'); plt.savefig(FIG+n+'.png',dpi=300,bbox_inches='tight'); plt.close()
fig,axs=plt.subplots(2,4,figsize=(7.2,3.5),sharex='col',gridspec_kw=dict(height_ratios=[3,1.2],hspace=0.08,wspace=0.42))
for c,(key,name,T) in enumerate(ARCH):
    d=D[key]; o=np.argsort(d['j']); j=d['j'][o]; V=d['V'][o]; jj=np.linspace(j.min(),j.max(),300); ax=axs[0,c]
    pf=fit(j[0::2],V[0::2],T)
    ax.plot(jj,V_cell(jj,pf,T),'-',color=COL[key],lw=1.3,label='model (fitted points)')
    ax.plot(j[0::2],V[0::2],'o',color=COL[key],ms=3.5,label='fitted'); ax.plot(j[1::2],V[1::2],'s',mfc='none',color='k',ms=3.5,label='held out')
    ax.set_title(f'({"abcd"[c]}) {name}',loc='left',fontsize=8)
    a2=axs[1,c]; a2.axhline(0,color='#555',lw=0.6); a2.plot(j[0::2],1e3*(V[0::2]-V_cell(j[0::2],pf,T)),'o',color=COL[key],ms=3); a2.plot(j[1::2],1e3*(V[1::2]-V_cell(j[1::2],pf,T)),'s',mfc='none',color='k',ms=3)
    a2.set_xlabel(r'$j$ (A cm$^{-2}$)'); a2.set_ylim(-25,25)
axs[0,0].set_ylabel('Cell voltage (V)'); axs[1,0].set_ylabel('Residual (mV)'); h,l=axs[0,0].get_legend_handles_labels(); [x.set_color('k') for x in []]; fig.legend(h,['model (fitted to filled points)','fitted points','held-out points'],loc='upper center',ncol=3,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,1.04))
save('fig_calibration')
fig,axs=plt.subplots(1,4,figsize=(7.2,2.3),sharey=True)
for c,key in enumerate(['AFG','AZG','PEM','AEM80']):
    ax=axs[c]; jj=np.geomspace(0.03,6,400); Vf=Vf_of(key)
    for g,ls in zip([0.05,0.2,1.0],[':','--','-']):
        u=U(jj,Vf,g,JX); ax.plot(jj,u,ls,color=COL[key],lw=1.3,label=rf'$\Gamma={g:g}$'); o=solve(key,g,theta_max=TH0); ax.plot(o['j'],o['U'],'o',color=COL[key],ms=4,mfc='w' if o['j']>JMAXDATA[key] else COL[key])
    ax.axvspan(JMAXDATA[key],6,color='#999',alpha=0.15,lw=0); ax.set_xscale('log'); ax.set_ylim(1.5,4.5); ax.set_xlabel(r'$j$ (A cm$^{-2}$)')
    ax.set_title(f'({"abcd"[c]}) '+{'AFG':'Alkaline, finite gap','AZG':'Alkaline, zero gap','PEM':'PEM','AEM80':'AEM, 80 $^\\circ$C'}[key],loc='left',fontsize=8)
axs[0].set_ylabel(r'Cost-equivalent voltage $U$ (V)')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color='k',ls=l,lw=1.3,label=rf'$\Gamma={g:g}$ W cm$^{{-2}}$') for g,l in zip([0.05,0.2,1.0],[':','--','-'])]+[Line2D([],[],color='k',marker='o',ls='',ms=4,label='optimum (calibrated range)'),Line2D([],[],color='k',marker='o',mfc='w',ls='',ms=4,label='optimum (extrapolated)')],loc='upper center',ncol=5,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,1.08))
save('fig_objective')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for key in KEYS:
    c=curves[key]; jv=np.array(c['j']); ex=jv>JMAXDATA[key]
    ax[0].plot(np.array(c['G'])[~ex],jv[~ex],'-',color=COL[key],lw=1.5,label=NAMES[key].replace(' C',' $^\\circ$C')); ax[0].plot(np.array(c['G'])[ex],jv[ex],':',color=COL[key],lw=1.5)
    ax[0].fill_between(c['Gb'],c['lo'],c['hi'],color=COL[key],alpha=0.15,lw=0)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'Cost ratio $\Gamma$ (W cm$^{-2}$)'); ax[0].set_ylabel(r'$j^{*}$ (A cm$^{-2}$)'); ax[0].legend(fontsize=6.3,loc='upper left'); ax[0].set_title('(a) cost-optimal current density',loc='left',fontsize=8)
for key in KEYS:
    c=curves[key]; ax[1].plot(c['G'],E_TN/np.array(c['V']),'-',color=COL[key],lw=1.5)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'Cost ratio $\Gamma$ (W cm$^{-2}$)'); ax[1].set_ylabel(r'Voltage efficiency at $j^{*}$'); ax[1].set_title('(b) voltage efficiency at the optimum',loc='left',fontsize=8)
save('fig_optimum_vs_gamma')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
from matplotlib.colors import ListedColormap
cm=ListedColormap(['#e9ecef','#1b6ca8','#e9c46a','#f4a261'])
ax[0].pcolormesh(GG,THS,np.clip(reg,0,3),cmap=cm,vmin=0,vmax=3,shading='auto'); ax[0].set_xscale('log')
gc=np.array([g if (g is not None and g<=3) else np.nan for g in gcrit]); ax[0].plot(gc,THS,'k-',lw=1.0); ax[0].set_xlim(0.01,3)
cs=ax[0].contour(GG,THS,jmap,levels=[0.2,0.4,0.8,1.6],colors='w',linewidths=0.7); ax[0].clabel(cs,fmt='%g',fontsize=6)
ax[0].set_xlabel(r'Cost ratio $\Gamma$ (W cm$^{-2}$)'); ax[0].set_ylabel(r'Coverage ceiling $\theta_{\max}$'); ax[0].set_title('(a) binding limit, finite-gap alkaline',loc='left',fontsize=8)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color='#e9ecef',label='interior optimum'),Patch(color='#1b6ca8',label='coverage ceiling binds'),Patch(color='#f4a261',label='upper domain bound')],fontsize=6.3,loc='upper left')
for th,ls in zip([0.25,0.30,0.40],['-','--',':']):
    pu=[100*(solve('AFG',g,theta_max=th)['U']/solve('AFG',g,theta_max=None)['U']-1) for g in GG[::3]]
    ax[1].plot(GG[::3],pu,ls,color=COL['AFG'],lw=1.4,label=rf'$\theta_{{\max}}={th:.2f}$')
ax[1].set_xscale('log'); ax[1].set_xlabel(r'Cost ratio $\Gamma$ (W cm$^{-2}$)'); ax[1].set_ylabel(r'Cost increase due to ceiling (%)'); ax[1].legend(fontsize=6.5); ax[1].set_title('(b) cost of the coverage ceiling',loc='left',fontsize=8)
save('fig_coverage_regimes')
print(json.dumps({k:{kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!='corr'} for k,v in cal.items()},indent=0)[:3000])
print('jx',jx)
for t in tab: print(t['key'],t['Gamma'],'j* %.3f [%.3f,%.3f] closed %.3f V %.3f etaE %.3f U %.3f %s extrap=%s theta=%s'%(t['j'],t['lo'],t['hi'],t['closed'],t['V'],t['etaE'],t['U'],t['binding'],t['extrapolated'],t['theta']))
print('eff floor',json.dumps(eff,default=float)[:1500])
print('afg cover', [(round(a,2),round(b,3),None if c is None else round(c,3)) for a,b,c in zip(THS[::5],((THS/K_VB)**(1/0.3))[::5],gcrit[::5])])
for p_ in pen: print(p_)
for s_ in sens: print(s_)
for m_ in mc: print(m_)
print([ (s['key'],s['Gamma'],s['kind'],s['val'],round(s['j'],3),s['binding']) for s in sc])
