import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
AQUI=os.path.dirname(os.path.abspath(__file__))
CSV=next((c for c in [os.path.join(AQUI,"por_semilla.csv"),os.path.join(AQUI,"analysis_revision","por_semilla.csv"),os.path.join(AQUI,"data","por_semilla.csv")] if os.path.exists(c)),None)
if CSV is None: raise SystemExit("Pon por_semilla.csv junto a este script o en analysis_revision/")
d=pd.read_csv(CSV); b=d[d.beta==1]
G=[("beta1_pinn","PINN\nModel-2"),("beta1_qcpinn","QCPINN\nCascade"),("beta1_none","Control\nsin centro"),("beta1_cos","Control\ncoseno"),("beta1_ctrl","Control\nlineal")]
col={"beta1_pinn":"#4C72B0","beta1_qcpinn":"#C44E52","beta1_none":"#8C8C8C","beta1_cos":"#8C8C8C","beta1_ctrl":"#8C8C8C"}
g=lambda n,c="L2_u_global_pct": b[b.grupo==n][c].dropna().values

def welch_df(a,b):
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
    return (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
rng=np.random.default_rng(0); out=os.path.join(AQUI,"figuras")+os.sep; os.makedirs(out,exist_ok=True)
# 17 boxplot
fig,ax=plt.subplots(figsize=(6.4,3.6))
for i,(k,l) in enumerate(G):
    v=g(k); ax.boxplot(v,positions=[i],widths=.5,showfliers=False,medianprops=dict(color="k"),boxprops=dict(color=col[k]),whiskerprops=dict(color=col[k]),capprops=dict(color=col[k]))
    ax.scatter(i+rng.uniform(-.12,.12,len(v)),v,s=14,color=col[k],alpha=.8,zorder=3)
ax.axhline(0.0051,color="k",lw=.8,ls=":"); ax.set_yscale("log")
ax.set_xticks(range(len(G))); ax.set_xticklabels([l for _,l in G]); ax.set_ylabel(r"$\varepsilon_{L_2}(u)$ [%] ($\beta=1$)")
fig.tight_layout(); fig.savefig(out+"17_masa_boxplot.pdf"); plt.close(fig)
# 18 regiones
cols=[("L2_u_global_pct","Global"),("L2_u_interior_pct","Interior"),("L2_u_boundary_pct","Frontera"),("L2_u_t0_pct",r"$t=0$")]
fig,ax=plt.subplots(figsize=(6.8,3.6)); w=.2
for j,(c,l) in enumerate(cols):
    m=[g(k,c).mean() for k,_ in G]; s=[g(k,c).std(ddof=1) for k,_ in G]
    ax.bar(np.arange(len(G))+(j-1.5)*w,m,w,yerr=s,capsize=2,label=l,color=["#2F3E5C","#4C72B0","#8FB3D9","#C9D9EC"][j])
ax.set_yscale("log"); ax.set_xticks(range(len(G))); ax.set_xticklabels([l for _,l in G]); ax.set_ylabel(r"$\varepsilon_{L_2}(u)$ [%] ($\beta=1$)"); ax.legend(frameon=False,ncol=4,fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(out+"18_masa_regiones.pdf"); plt.close(fig)
# 19 contrastes
q=g("beta1_qcpinn"); fig,ax=plt.subplots(figsize=(6.2,2.8)); rows=[]
for i,(k,l) in enumerate(G):
    if k=="beta1_qcpinn": continue
    x=g(k); t=stats.ttest_ind(q,x,equal_var=False); df=welch_df(q,x); dif=q.mean()-x.mean(); se=np.sqrt(q.var(ddof=1)/len(q)+x.var(ddof=1)/len(x)); ci=stats.t.ppf(.975,df)*se
    rows.append((k,dif,dif-ci,dif+ci,t.pvalue))
for i,(k,dif,lo,hi,p) in enumerate(rows):
    ax.errorbar(dif,i,xerr=[[dif-lo],[hi-dif]],fmt="o",color="#C44E52",capsize=3)
ax.axvline(0,color="k",lw=.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([dict(G)[r[0]].replace("\n"," ") for r in rows]); ax.invert_yaxis()
ax.set_xlabel(r"QCPINN menos configuración, $\varepsilon_{L_2}(u)$ [pp] ($\beta=1$)"); fig.tight_layout(); fig.savefig(out+"19_masa_contrastes.pdf"); plt.close(fig)
# 20 energia por semilla
E=[("energia_cls","PINN\nAdam"),("energia_dv","QCPINN\nAdam"),("energia_cls_lbfgs","PINN\nL-BFGS"),("energia_dv_lbfgs","QCPINN\nL-BFGS")]
fig,ax=plt.subplots(figsize=(6.2,3.4))
for i,(k,l) in enumerate(E):
    v=b[b.grupo==k].drift_rel.dropna().values; c="#4C72B0" if "cls" in k else "#C44E52"
    ax.scatter(i+rng.uniform(-.1,.1,len(v)),v,s=18,color=c,alpha=.85,zorder=3); ax.hlines(v.mean(),i-.25,i+.25,color="k",lw=1.4)
ax.axhline(2.3e-6,color="k",ls=":",lw=.8); ax.axhline(4.5e-6,color="k",ls="--",lw=.8)
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in E]); ax.set_ylabel("Deriva relativa de la energía")
fig.tight_layout(); fig.savefig(out+"20_energia_semillas.pdf"); plt.close(fig)
# numeros
print("contrastes u:"); [print(r[0],"%+.2f [%+.2f,%+.2f] p=%.3f"%r[1:]) for r in rows]
e=lambda n:b[b.grupo==n].drift_rel.dropna().values
for a,c in [("energia_dv","energia_cls"),("energia_dv_lbfgs","energia_cls_lbfgs")]:
    x,y=e(a),e(c); t=stats.ttest_ind(x,y,equal_var=False); dif=x.mean()-y.mean(); ci=stats.t.ppf(.975,welch_df(x,y))*np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
    print(a,c,"%.2e [%.2e,%.2e] p=%.4f mw=%.4f ratio=%.2f"%(dif,dif-ci,dif+ci,t.pvalue,stats.mannwhitneyu(x,y).pvalue,x.mean()/y.mean()))
print("vs FD", [ (k, e(k).mean()/2.3e-6) for k,_ in E])
