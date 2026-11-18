"""Plots from saved result files and schematics of the reviewed implementation."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/reports/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.titlesize':12,
                     'axes.labelsize':12,'legend.fontsize':11,'svg.fonttype':'none'})
BLUE,ORANGE,GREEN='#245b88','#c66526','#2b7a64'
G=json.loads((ROOT/'results/ga/part1_review.json').read_text())
N=np.load(ROOT/'results/nn/nn_model_line_search.npz')
P=json.loads((ROOT/'results/report_library_gp/comparison.json').read_text())
T=np.load(ROOT/'results/report_library_gp/test_predictions.npz')

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

def box(ax,x,y,w,h,label,color='#edf3f7',size=12):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.015,rounding_size=.025',facecolor=color,edgecolor=BLUE,lw=1))
    ax.text(x,y,label,ha='center',va='center',fontsize=size)

def arrow(ax,a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.1})
    if label: ax.text((a[0]+b[0])/2+.28,(a[1]+b[1])/2,label,fontsize=11,ha='left',va='center')

# Antenna geometry and project-level flow. Geometry is schematic, not an EM model.
fig,ax=plt.subplots(figsize=(6.5,1.9));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
for pts,c in [([(0.3,.65),(2.2,.65),(2.9,1.4),(1,1.4)],'#b7d3df'),
              ([(.3,.65),(2.2,.65),(2.2,.38),(.3,.38)],'#d9dbbf'),
              ([(2.2,.65),(2.9,1.4),(2.9,1.13),(2.2,.38)],'#c9caa5'),
              ([(.8,.9),(1.9,.9),(2.38,1.4),(1.28,1.4)],'#cf894f')]:
    ax.add_patch(Polygon(pts,facecolor=c,edgecolor='#354652',lw=1))
ax.text(1.5,2.65,'Patch geometry',ha='center',weight='bold')
ax.text(1.5,2.15,'Copper patch / FR-4',ha='center',fontsize=11)
ax.text(1.5,.05,'Ground plane; not to scale',ha='center',fontsize=11)
ax.annotate('',xy=(1.9,.81),xytext=(.8,.81),arrowprops={'arrowstyle':'<->'})
ax.text(1.35,.55,'W',ha='center')
ax.annotate('',xy=(2.49,1.42),xytext=(2.02,.92),arrowprops={'arrowstyle':'<->'})
ax.text(2.57,1.12,'L');ax.text(.12,.43,'h')
box(ax,4.1,2.25,1.35,.55,'GA')
box(ax,6.2,2.25,2.0,.55,'Analytical f')
box(ax,8.7,2.25,1.65,.55,'Best design')
arrow(ax,(4.8,2.25),(5.16,2.25));arrow(ax,(7.22,2.25),(7.85,2.25))
box(ax,4.1,.95,1.35,.75,'Sample\n+ label')
box(ax,6.2,.95,2.0,.75,'Train NN / GP')
box(ax,8.7,.95,1.65,.75,'PSO +\nverification')
arrow(ax,(4.8,.95),(5.16,.95));arrow(ax,(7.22,.95),(7.85,.95))
save(fig,'01_antenna_and_workflow')

fig,ax=plt.subplots(figsize=(6.5,1.85),layout='constrained')
best=min(G['runs'],key=lambda r:r['fitness'])
for r in G['runs']:
    ax.plot(r['history'],color=BLUE,alpha=.28,lw=.8)
ax.plot(best['history'],color=BLUE,lw=1.6,label=f"Best score: seed {best['actual_seed']}")
ax.set(xlabel='Generation',ylabel='Best objective',xlim=(0,100))
ax.legend(loc='upper right',frameon=False);ax.grid(alpha=.18)
save(fig,'02_ga_convergence')

fig,ax=plt.subplots(figsize=(6.5,1.25));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
for x,w,label in [(1,1.8,'W, L, h\n(mm)'),(3.3,1.9,'Standardize\n3 inputs'),(5.9,2.1,'32 tanh\nneurons'),(8.65,2.1,'Linear output\n→ GHz')]: box(ax,x,1.1,w,.8,label)
for a,b in [(1.92,2.33),(4.28,4.83),(6.98,7.58)]:arrow(ax,(a,1.1),(b,1.1))
ax.text(5,.18,'W₁: 32 × 3     b₁: 32 × 1     W₂: 1 × 32     b₂: 1 × 1     |     161 parameters',ha='center',fontsize=11)
save(fig,'03_nn_architecture')

fig,axs=plt.subplots(1,2,figsize=(6.5,2.15),layout='constrained')
epochs=np.arange(1,len(N['training_loss'])+1)
for key,label,c in [('training_loss','Training',BLUE),('validation_loss','Validation',ORANGE)]:
    axs[0].semilogy(epochs,np.sqrt(N[key])*1000,color=c,label=label,lw=1)
axs[0].set(xlabel='Epoch',ylabel='RMSE (MHz)',xticks=[0,15000,30000]);axs[0].legend(frameon=False,fontsize=10)
order=np.argsort(T['actual_GHz'])
err=(T['NN_GHz']-T['actual_GHz'])*1000
axs[1].scatter(T['actual_GHz'][order],err[order],s=9,color=BLUE)
axs[1].axhline(0,color='black',lw=.7)
axs[1].set(xlabel='Analytical f (GHz)',ylabel='NN error (MHz)')
for ax in axs:ax.grid(alpha=.18)
save(fig,'04_nn_training_error')

fig,axs=plt.subplots(1,2,figsize=(6.5,2.3),layout='constrained')
for ax,name in zip(axs,['NN','GP']):
    a=T['actual_GHz'];b=T[name+'_GHz'];ax.scatter(a,b,s=10,color=BLUE if name=='NN' else GREEN)
    ax.plot([a.min(),a.max()],[a.min(),a.max()],'k--',lw=.8)
    ax.set(title=name+(' (scikit-learn)' if name=='GP' else ''),xlabel='Analytical f (GHz)',ylabel='Predicted f (GHz)');ax.grid(alpha=.18)
save(fig,'05_test_parity')

fig,ax=plt.subplots(figsize=(6.5,1.8),layout='constrained')
for name,c in [('NN',BLUE),('GP',GREEN)]:
    histories=[np.loadtxt(ROOT/f'results/report_library_gp/{name}_seed_{s}_history.csv',delimiter=',',skiprows=1)[:,1] for s in range(1,6)]
    arr=np.array(histories)
    ax.fill_between(np.arange(arr.shape[1]),arr.min(0),arr.max(0),color=c,alpha=.14)
    ax.plot(np.median(arr,axis=0),color=c,label=name+(' (library)' if name=='GP' else ''),lw=1.5)
ax.set(xlabel='PSO iteration',ylabel='Best objective',xlim=(0,150));ax.grid(alpha=.18);ax.legend(frameon=False)
save(fig,'06_pso_convergence')

gpu=json.loads((ROOT/'results/pso_comparison/comparison.json').read_text())
cpu=json.loads((ROOT/'results/pso_comparison_cpu/comparison.json').read_text())
times=[[r['search_seconds'] for r in d['runs'] if r['model']=='NN'] for d in [gpu,cpu]]
fig,ax=plt.subplots(figsize=(6.5,2.2),layout='constrained')
means=[np.mean(t) for t in times]
ax.bar(['GPU NN inference','CPU NN inference'],means,color=[ORANGE,BLUE],width=.55)
for i,t in enumerate(times):ax.scatter(np.repeat(i,len(t)),t,color='black',s=12,zorder=3)
for i,m in enumerate(means):ax.text(i,max(times[i])+.12,f'{m:.3f} s',ha='center')
ax.set(ylabel='PSO search time (s)',ylim=(0,max(max(t) for t in times)*1.22),title='Same saved NN; five seeds per backend')
save(fig,'07_cpu_gpu_latency')

# Standalone detailed flowcharts for reuse in the report/presentation.
flows={
'08_ga_flowchart':('GeneticAlgorithm.py',[
'Seed RNG; initialize 6 designs; repair geometry','Evaluate analytical frequency and objective',
'Copy best distinct elite; tournament selection','Arithmetic crossover; Gaussian mutation; repair',
'Replace duplicates; fill next generation; score',
'Record best; after 100 generations check change < 1%',
'Return best design and history']),
'09_nn_flowchart':('NeuralNetwork.py',[
'Load 600 samples; shared 420 / 90 / 90 split','Fit training scaling; Xavier weights; zero biases',
'Forward pass: 3 inputs → 32 tanh → 1 linear','MSE and handwritten backpropagation',
'Strong Wolfe line search; update all parameters','Validation loss; save best weights; check stop',
'Restore best; save scaling and weights; test']),
'10_pso_flowchart':('PSO.py',[
'Load fixed surrogate; seed 40 particles and velocities','Repair each particle; score with surrogate objective',
'Initialize personal bests and global best','Update and clip velocity; move and repair positions',
'Score; update personal and global bests; save history',
'150 minimum; stop below 1% change or at 1000',
'Analytically verify design; save results; next seed'])}
for filename,(name,steps) in flows.items():
    fig,ax=plt.subplots(figsize=(6.5,6.0));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,10),ylim=(0,9));ax.axis('off');ax.text(5,8.65,name,ha='center',weight='bold',fontsize=15)
    ys=np.linspace(7.85,1.0,7)
    for i,(y,s) in enumerate(zip(ys,steps)):
        box(ax,4.8,y,8,.7,s,size=12)
        if i<6:arrow(ax,(4.8,y-.37),(4.8,ys[i+1]+.37),'Stop' if i==5 else None)
    back=3 if filename=='10_pso_flowchart' else 2
    ax.plot([8.83,9.4,9.4,8.83],[ys[5],ys[5],ys[back],ys[back]],color=BLUE,lw=1)
    arrow(ax,(9.4,ys[back]),(8.83,ys[back]));ax.text(9.5,(ys[5]+ys[back])/2,'Continue',rotation=90,va='center',fontsize=11)
    save(fig,filename)

sys.path.insert(0,str(ROOT/'src'))
from PatchFormula import RMA
for r in G['runs']:
    a=RMA(*r['design_mm']);assert a.geometryCheck()
    assert abs(a.ResonanceF_GHz-r['frequency_GHz'])<1e-12
    assert np.all(np.diff(r['history'])<=1e-12)
for r in P['runs']:
    a=RMA(*r['design_mm']);assert a.geometryCheck()
    assert abs(a.ResonanceF_GHz-r['analytical_GHz'])<1e-12
    hist=np.loadtxt(ROOT/f"results/report_library_gp/{r['model']}_seed_{r['seed']}_history.csv",delimiter=',',skiprows=1)
    assert np.all(np.diff(hist[:,1])<=1e-12)
    assert np.allclose(hist[-1,2:],r['design_mm'])
summary={'ga_best':best,'ga_mean_error_MHz':float(np.mean([r['target_error_MHz'] for r in G['runs']])),
         'library_comparison':P['summary'],'gpu_nn_mean_seconds':means[0],'cpu_nn_mean_seconds':means[1],
         'gpu_cpu_ratio':means[0]/means[1],
         'checks':'Analytical final frequencies, geometries, monotonic histories and final CSV positions verified.'}
summary['ga_best']={k:v for k,v in best.items() if k!='history'}
(OUT.parent/'figure_audit.json').write_text(json.dumps(summary,indent=2))
html='<html><meta charset="utf-8"><title>EE5606 report figures</title><style>body{max-width:1100px;margin:40px auto;font:18px Georgia}img{max-width:100%;border:1px solid #ddd}article{margin:40px 0}</style><h1>EE5606 report figures</h1><p>Plots use saved experiment data. Schematics describe the reviewed code.</p>'
for p in sorted(OUT.glob('*.png')):
    html+=f'<article><h2>{p.stem.replace("_"," ")}</h2><a href="{p.stem}.svg">SVG</a> · <a href="{p.name}">PNG</a><br><img src="{p.name}"></article>'
(OUT/'index.html').write_text(html+'</html>',encoding='utf-8')
print(json.dumps(summary,indent=2))
