"""Publication figures from frozen measurements; exports vector PDF/SVG and600dpi PNG.

No model fitting or network calls. Pass explicit data and output paths.
"""
from pathlib import Path
import argparse,json,tarfile,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
BLUE='#2864A0';ORANGE='#C66A2B';DARK='#263647';GRAY='#788795';GRID='#E3E8ED';COLORS={'mcp':BLUE,'native':ORANGE};NAMES={'mcp':'Deterministic MCP','native':'Native Bedrock'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10.3,'axes.labelsize':9.2,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.edgecolor':'#77828D','axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':DARK,'text.color':DARK,'xtick.color':DARK,'ytick.color':DARK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'lines.linewidth':1.6,'lines.markersize':4.8})
CMAP=LinearSegmentedColormap.from_list('paper_blue',['#F1F5FA','#A6C7E2','#2864A0','#103A65'])
def save(fig,name,dest):
 from matplotlib.text import Text
 for artist in fig.findobj(Text):
  if artist.get_fontsize()<8.3:artist.set_fontsize(8.3)
 for ax in fig.axes:
  leg=ax.get_legend()
  if leg is not None:leg.set_frame_on(True);leg.get_frame().set_facecolor('white');leg.get_frame().set_edgecolor(GRID);leg.get_frame().set_alpha(.96)
 for ext in ('pdf','svg','png'):fig.savefig(dest/(name+'.'+ext),bbox_inches='tight',pad_inches=.07)
 plt.close(fig)
def base(ax,chance=None):
 ax.grid(axis='y',color=GRID,lw=.65,zorder=0);ax.set_axisbelow(True)
 if chance is not None:ax.axhline(chance,color=GRAY,ls=(0,(4,3)),lw=.9,zorder=1)
def matrix(ax,values,rows,columns,title):
 image=ax.pcolormesh(np.arange(len(columns)+1),np.arange(len(rows)+1),values,cmap=CMAP,vmin=0,vmax=100,edgecolors='white',linewidth=.8)
 ax.set_xticks(np.arange(len(columns))+.5,columns);ax.set_yticks(np.arange(len(rows))+.5,rows);ax.invert_yaxis();ax.tick_params(length=0,pad=5);ax.set_title(title,loc='left',fontweight='bold',pad=10)
 for i in range(len(rows)):
  for j in range(len(columns)):ax.text(j+.5,i+.5,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>=65 else DARK,fontsize=9)
 for spine in ax.spines.values():spine.set_visible(False)
 return image

def build(summary,evaluation,archive,engineering,dest,ablation_archive=None,engineering_captures=None):
 dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'r:gz') as t:
  reports={m:json.load(t.extractfile(f'outputs/{m}/report.json')) for m in ('rf','knn512')};raw=json.load(t.extractfile('outputs/rf_bundle/input.json'))
 allrows=raw['rows'];common=set.intersection(*[{r['session'] for r in allrows if r['cohort']==c} for c in ('mcp','native')]);rows=[r for r in allrows if r['session'] in common]
 # Session-complete marginal and joint disclosure.
 fig,axs=plt.subplots(1,3,figsize=(6.9,2.8),sharey=True,layout='constrained')
 for i,(ax,target,title) in enumerate(zip(axs,('class','scope','joint'),('Operation class','Target count','Joint class + scope'))):
  base(ax,100/9 if target=='joint' else 100/3)
  for off,c,mark in [(-.08,'mcp','o'),(.08,'native','s')]:
   z=[next(r for r in summary['loso'] if r['cohort']==c and r['target']==target and r['view']==v) for v in ('ct','ca','ctca')];y=np.array([r['balanced_accuracy']*100 for r in z]);ci=np.array([r['conditional_session_bootstrap_95'] for r in z])*100
   ax.errorbar(np.arange(3)+off,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,color=COLORS[c],capsize=3,ms=5,label=NAMES[c],zorder=3)
  ax.set_xticks(range(3),['Tool','Assurance','Combined'],rotation=20);ax.set_ylim(0,106);ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontweight='bold');ax.set_yticks([0,25,50,75,100])
 axs[0].set_ylabel('Balanced accuracy (%)');h,l=axs[1].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);save(fig,'class_scope_joint',dest)
 # Scope exposure is complementary across operation and channel.
 fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
 for i,(ax,c) in enumerate(zip(axs,('mcp','native'))):
  values=np.array([[next(r for r in summary['scope_by_operation_all_views'] if r['cohort']==c and r['class_id']==cl and r['view']==v)['balanced_accuracy']*100 for v in ('ct','ca','ctca')] for cl in ('I01','I02','I15')]);image=matrix(ax,values,['Quarantine\nI01','Release\nI02','Rule audit\nI15'],['Tool','Assurance','Combined'],f'({chr(97+i)}) {NAMES[c]}')
 fig.colorbar(image,ax=axs,fraction=.028,pad=.02,label='Scope balanced accuracy (%)');save(fig,'scope_complementarity',dest)
 # All held-out confusion counts, no selected seed.
 report=json.loads((evaluation/'report.json').read_text());fig,axs=plt.subplots(2,2,figsize=(6.7,5.6),layout='constrained')
 for i,c in enumerate(('mcp','native')):
  for j,target in enumerate(('class','scope')):
   labels=['I01','I02','I15'] if target=='class' else ['1','2','4'];fits=[r for r in report['results'] if r['kind']=='loso' and r['cohort']==c and r['target']==target and r['view']=='ca'];pred=[p for r in fits for p in json.loads((evaluation/'predictions'/(r['id']+'.json')).read_text())];counts=np.array([[sum(p['truth']==a and p['prediction']==b for p in pred) for b in labels] for a in labels]);values=counts/counts.sum(axis=1)[:,None]*100;ax=axs[i,j];im=matrix(ax,values,labels,labels,f'({chr(97+i*2+j)}) {"MCP" if c=="mcp" else "Native"}: {target}')
   for txt in list(ax.texts):txt.remove()
   for a in range(3):
    for b in range(3):ax.text(b+.5,a+.5,f'{values[a,b]:.1f}%\n(n={counts[a,b]})',ha='center',va='center',fontsize=8.3,color='white' if values[a,b]>=65 else DARK)
   ax.set_xlabel('Predicted label');ax.set_ylabel('True label')
 save(fig,'confusion_oof',dest)
 # Directional transfer across all measured management views.
 scenarios=[('mcp','native','session_holdout','M → N'),('native','mcp','session_holdout','N → M'),('mcp','mcp','two_to_four','M: 2 → 4'),('mcp','mcp','four_to_two','M: 4 → 2'),('native','native','two_to_four','N: 2 → 4'),('native','native','four_to_two','N: 4 → 2')]
 for target in ('class','scope'):
  fig,axs=plt.subplots(1,2,figsize=(6.7,4.0),layout='constrained')
  for i,(ax,method,name) in enumerate(zip(axs,('rf','knn512'),('Random forest','3-NN, 512 records'))):
   values=[]
   for src,dt,scenario,label in scenarios:
    line=[]
    for view in ('ct','ca','ctca'):
     fits=[r for r in reports[method]['results'] if r['source']==src and r['destination']==dt and r.get('scenario','session_holdout')==scenario and r['view']==view and r['target']==target and not r.get('permuted_training_labels',False) and r.get('metric','euclidean')=='euclidean'];assert len(fits)==5;line.append(statistics.mean(r['balanced_accuracy'] for r in fits)*100)
    values.append(line)
   im=matrix(ax,np.array(values),[s[3] for s in scenarios],['Tool','Assurance','Combined'],f'({chr(97+i)}) {name}')
  fig.colorbar(im,ax=axs,fraction=.028,pad=.02,label='Balanced accuracy (%)');save(fig,'transfer_'+target,dest)
 # Learning and partial-observation curves.
 for kind,key,name in [('learning','learning','learning_curves'),('prefix','prefix','prefix_evidence')]:
  n=2 if kind=='learning' else 3;fig,axs=plt.subplots(1,n,figsize=(6.7,3.0),layout='constrained')
  for j,target in enumerate(('class','scope')):
   ax=axs[j];base(ax,100/3)
   for c,marker in [('mcp','o'),('native','s')]:
    z=sorted([r for r in summary[key] if r['cohort']==c and r['target']==target],key=lambda r:r['budget']);x=np.array([r['budget'] for r in z]);y=np.array([r['mean']*100 for r in z]);ax.plot(x,y,marker=marker,color=COLORS[c],label=NAMES[c]);ax.fill_between(x,[r['min']*100 for r in z],[r['max']*100 for r in z],color=COLORS[c],alpha=.12,lw=0)
   ax.set_ylim(25,105);ax.set_title(f'({chr(97+j)}) {"Operation class" if target=="class" else "Target count"}',loc='left',fontweight='bold');ax.set_xlabel('Training sessions' if kind=='learning' else 'Max. assurance records')
   if kind=='prefix':ax.set_xscale('log',base=2);ax.set_xticks([4,16,64,256],['4','16','64','256'])
   else:ax.set_xticks([2,4,6,8,10]);ax.set_xlim(1.6,10.4)
  axs[0].set_ylabel('Balanced accuracy (%)');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
  if kind=='prefix':
   ax=axs[2];base(ax)
   for c,marker in [('mcp','o'),('native','s')]:
    z=sorted([r for r in summary[key] if r['cohort']==c and r['target']=='class'],key=lambda r:r['budget']);ax.plot([r['budget'] for r in z],[r['all_trace_ended_by_budget_fraction']*100 for r in z],marker=marker,color=COLORS[c])
   ax.set_xscale('log',base=2);ax.set_xticks([4,16,64,256],['4','16','64','256']);ax.set_ylim(-3,105);ax.set_title('(c) Trace completed',loc='left',fontweight='bold');ax.set_xlabel('Max. assurance records');ax.set_ylabel('Matched traces (%)')
  save(fig,name,dest)
 # Additional fixed classifier families, same data splits, current software.
 fig,axs=plt.subplots(1,2,figsize=(6.7,3.1),layout='constrained');methods=['RF','ExtraTrees','Logistic','RBF-SVM','HistGB']
 for j,target in enumerate(('class','scope')):
  ax=axs[j];base(ax,100/3)
  for c,off,marker in [('mcp',-.08,'o'),('native',.08,'s')]:
   z=[next(r for r in summary['baselines'] if r['cohort']==c and r['target']==target and r['method']==m) for m in methods];y=np.array([r['mean']*100 for r in z]);ax.errorbar(np.arange(5)+off,y,yerr=[y-np.array([r['min']*100 for r in z]),np.array([r['max']*100 for r in z])-y],fmt=marker,color=COLORS[c],capsize=3,label=NAMES[c]);
  ax.set_xticks(range(5),['RF','Extra\ntrees','Logistic','RBF\nSVM','Hist.\nboost']);ax.set_ylim(25,105);ax.set_title(f'({chr(97+j)}) {"Class" if target=="class" else "Scope"}',loc='left',fontweight='bold')
 axs[0].set_ylabel('Balanced accuracy (%)');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);save(fig,'classifier_families',dest)
 # Conditional null distributions, both-side blocked label permutations.
 fig,axs=plt.subplots(1,2,figsize=(6.7,2.9),layout='constrained')
 for i,(ax,r) in enumerate(zip(axs,summary['permutation'])):
  ax.hist(np.array(r['null_values'])*100,bins=np.arange(0,65,5),color='#B8CDDD',edgecolor='white',lw=.6);ax.axvline(r['observed']*100,color=ORANGE,lw=1.7);ax.axvline(100/3,color=GRAY,ls=(0,(4,3)),lw=.9);ax.set_xlim(0,105);ax.set_xlabel('Class balanced accuracy (%)');ax.set_title(f'({chr(97+i)}) {NAMES[r["cohort"]]}',loc='left',fontweight='bold');ax.text(.55,.87,f'Observed {r["observed"]*100:.1f}%\np = {r["conditional_permutation_p"]:.3f}',transform=ax.transAxes,fontsize=9,color=ORANGE,ha='left',va='top');base(ax)
 axs[0].set_ylabel('Null replicates (n = 199)');save(fig,'blocked_permutation',dest)
 # Historical feature ablations; fixed controls are not p-values.
 # The ablation input is colocated with the supplied analysis archive.
 ablation=ablation_archive or archive.parent/'ablation_results.tar.gz'
 with tarfile.open(ablation,'r:gz') as t:ar=json.load(t.extractfile('ablations/report.json'))
 fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained');ab=['count_only','size_only','timing_only','full','fixed_padding']
 for i,(ax,c) in enumerate(zip(axs,('mcp','native'))):
  base(ax,100/3)
  for control,off,color,marker,label in [(False,-.07,COLORS[c],'o','True training labels'),(True,.07,GRAY,'x','Permuted training labels')]:
   z=[next(r for r in ar['summary'] if r['cohort']==c and r['target']=='class' and r['ablation']==k and r['permuted']==control) for k in ab];y=np.array([r['balanced_accuracy_mean']*100 for r in z]);ax.errorbar(np.arange(5)+off,y,yerr=[y-np.array([r['min']*100 for r in z]),np.array([r['max']*100 for r in z])-y],fmt=marker,color=color,capsize=3,label=label)
  ax.set_xticks(range(5),['Count\nonly','Size\nonly','Timing\nonly','All','Length\nequalized']);ax.set_ylim(0,105);ax.set_title(f'({chr(97+i)}) {NAMES[c]}',loc='left',fontweight='bold')
 axs[0].set_ylabel('Class balanced accuracy (%)');h,l=axs[1].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);save(fig,'feature_ablations',dest)
 # Engineering: explicit units, active record cadence, scheduling tails, payload occupancy.
 eng=json.loads(engineering.read_text());erows=eng['rows'];fig,axs=plt.subplots(2,2,figsize=(6.7,5.5),layout='constrained');x=np.arange(8);runlabels=[f'S{i//4+1}R{i%4+1}' for i in x];palette=[BLUE]*4+[ORANGE]*4
 ax=axs[0,0];ax.bar(x,[r['response_body_bytes']/1024 for r in erows],color=palette,width=.65);base(ax);ax.set_ylim(0,85);ax.set_xticks(x,runlabels,rotation=45,ha='right');ax.set_ylabel('HTTP response body (KiB)');ax.set_title('(a) Preserved response bodies',loc='left',fontweight='bold')
 ax=axs[0,1]
 for i,r in enumerate(erows):
  capture_root=engineering_captures or engineering.parent
  candidates=(capture_root/r['batch']/'artifacts'/'readonly_runs'/r['run_id'],capture_root/r['batch']/r['run_id'])
  run_dir=next((p for p in candidates if (p/'tls_records.jsonl').is_file()),None);assert run_dir is not None
  path=run_dir/'tls_records.jsonl';events=[json.loads(s) for s in path.read_text().splitlines()];cell=[e for e in events if e['length_bytes']==4118];assert len(cell)==512
  ts=np.sort(np.array([e['timestamp_ns'] for e in cell],dtype=np.int64));rel=(ts-ts[0])/1e9;ax.step(rel,np.arange(1,513),where='post',color=palette[i],alpha=.4,lw=.8)
 ax.plot([0,2.56],[0,512],color=GRAY,ls=(0,(4,3)),lw=.9,label='Nominal aggregate cadence');ax.set_xlim(-.04,2.65);ax.set_ylim(0,535);base(ax);ax.set_xlabel('Time from first cell record (s)');ax.set_ylabel('Cumulative 4118 B TLS records');ax.set_title('(b) Eight captured tunnel traces',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=7.0,loc='lower right')
 ax=axs[1,0]
 for batch,color,label in [('batch_001',BLUE,'Controller session 1'),('batch_002',ORANGE,'Controller session 2')]:
  z=np.sort([v for r in erows if r['batch']==batch for v in r['cell_write_lateness_ms']]);ax.plot(z,np.arange(1,len(z)+1)/len(z),color=color,label=label)
 ax.set_xscale('log');ax.set_xlim(.015,3);ax.set_ylim(0,1.02);base(ax);ax.set_xlabel('Cell-write lateness (ms)');ax.set_ylabel('Empirical cumulative fraction');ax.set_title('(c) Scheduling error distribution',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=7.5,loc='lower right');ax.text(.04,.78,'P99 = 0.153 ms\nmax = 2.222 ms',transform=ax.transAxes,fontsize=8.3)
 ax=axs[1,1];ax.bar(x,[100*r['response_body_bytes']/(256*(4096-36)) for r in erows],color=palette,width=.65);base(ax);ax.set_ylim(0,10);ax.set_xticks(x,runlabels,rotation=45,ha='right');ax.set_ylabel('Response-body slot occupancy (%)');ax.set_title('(d) Engineering envelope utilization',loc='left',fontweight='bold');ax.text(.5,.88,'Denominator: 256 × 4060 B',transform=ax.transAxes,ha='center',fontsize=8.0)
 save(fig,'engineering_profile',dest)
 # Observable feature distributions, preserving within-session dependence in interpretation.
 fig,axs=plt.subplots(2,2,figsize=(6.7,5.1),layout='constrained');qcolors={1:BLUE,2:ORANGE,4:'#6F808F'}
 for i,c in enumerate(('mcp','native')):
  for j,(feature,label) in enumerate([('ca__event_count','Assurance TLS-record count'),('ca__bytes_direction_neg','Server-to-agent TLS bytes (KiB)')]):
   ax=axs[j,i]
   for k,q in enumerate((1,2,4)):
    vals=[[r[feature]/(1024 if j else 1) for r in rows if r['cohort']==c and r['class_id']==cl and r['scope_q']==q] for cl in ('I01','I02','I15')]
    bp=ax.boxplot(vals,positions=np.arange(3)+(k-1)*.22,widths=.18,patch_artist=True,showfliers=False,medianprops={'color':DARK,'linewidth':1.0},boxprops={'linewidth':.7},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
    for box in bp['boxes']:box.set_facecolor(qcolors[q]);box.set_alpha(.65)
   ax.set_xticks(range(3),['I01','I02','I15']);base(ax);ax.set_ylabel(label if i==0 else '');ax.set_title(f'({chr(97+j*2+i)}) {NAMES[c]}',loc='left',fontweight='bold')
   if j:ax.set_yscale('log')
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(facecolor=qcolors[q],alpha=.65,label=f'q = {q}') for q in (1,2,4)],frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3);save(fig,'observable_distributions',dest)
 # Architecture: measured channels are distinguished from the unmeasured model path.
 fig,ax=plt.subplots(figsize=(6.7,3.9));ax.set_xlim(0,10.9);ax.set_ylim(0,5.3);ax.axis('off')
 def box(cx,cy,w,h,text,color='#F1F5F8',edge='#98A7B5'):
  p=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0.025,rounding_size=0.05',facecolor=color,edgecolor=edge,linewidth=.8);ax.add_patch(p);ax.text(cx,cy,text,ha='center',va='center',fontsize=8.4)
 def arrow(x1,y1,x2,y2,text='',color=DARK,style='-',dy=.12):
  ax.annotate('',(x2,y2),(x1,y1),arrowprops={'arrowstyle':'<->','color':color,'linewidth':1.0,'linestyle':style,'shrinkA':0,'shrinkB':0});
  if text:ax.text((x1+x2)/2,(y1+y2)/2+dy,text,ha='center',va='bottom',fontsize=7.5,color=color)
 ax.text(.05,5.12,'(a) Measured agentic network-control pipeline',fontweight='bold',fontsize=10)
 box(.9,3.62,1.35,.72,'Agent');box(2.85,3.62,1.48,.72,'Tool service\nMCP / native');box(5.28,3.62,1.35,.72,'TLS proxy');box(7.43,3.62,1.3,.72,'ONOS 2.7');box(9.45,3.62,1.4,.72,'Mininet\n+ OVS')
 arrow(1.58,3.62,2.11,3.62);arrow(3.59,3.82,4.60,3.82,'$C_T$',BLUE,dy=.10);arrow(3.59,3.42,4.60,3.42,'$C_A$',ORANGE,dy=-.27);arrow(5.96,3.62,6.78,3.62,'REST',DARK,dy=.14);arrow(8.08,3.62,8.75,3.62)
 box(.9,4.65,1.35,.55,'Bedrock LLM',color='#FBFCFD');arrow(.9,3.99,.9,4.35,color=GRAY,style='--');ax.text(1.68,4.66,'$C_L$: not captured',fontsize=7.8,color=GRAY,va='center')
 box(5.28,4.75,1.9,.55,'Passive observer',color='#FFF8F1',edge=ORANGE);ax.plot([5.28,5.28],[4.46,4.08],color=GRAY,lw=.8,ls=':');ax.text(4.4,2.76,'Separate TLS 1.2 endpoints expose tool and assurance metadata; payloads remain encrypted',ha='center',fontsize=7.8,color=GRAY)
 ax.axhline(2.36,color=GRID,lw=.8);ax.text(.05,2.04,'(b) Separate read-only engineering qualification',fontweight='bold',fontsize=10)
 box(1.25,1.24,2.0,.70,'Agent-side shim');box(5.85,1.24,1.9,.70,'Bridge-side shim');box(9.0,1.24,1.6,.70,'Controller')
 arrow(2.28,1.24,4.87,1.24,'Loopback mTLS 1.3',BLUE,dy=.22);ax.text(3.57,.63,'4096 B cells · 10 ms · 256 rounds',ha='center',fontsize=7.8,color=BLUE);arrow(6.83,1.24,8.17,1.24,'HTTPS',DARK,dy=.18);ax.text(7.5,.63,'unshaped backend',ha='center',fontsize=7.8,color=GRAY);ax.text(5.0,.16,'One read-only GET; no remote mutations or measured class-hiding guarantee',ha='center',fontsize=7.8,color=GRAY)
 save(fig,'architecture',dest)
 print(json.dumps({'figures':13,'formats':['PDF','SVG','PNG600dpi'],'figure_directory':str(dest)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--ablation-archive',type=Path);p.add_argument('--engineering',type=Path,required=True);p.add_argument('--engineering-captures',type=Path);p.add_argument('--figure-dir',type=Path,required=True);a=p.parse_args();build(json.loads(a.summary.read_text()),a.evaluation,a.archive,a.engineering,a.figure_dir,a.ablation_archive,a.engineering_captures)
