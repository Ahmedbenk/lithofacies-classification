# Draws Figures 1, 3, 4, 5 and 6 of the paper.
#
# The numbers below (confusion-matrix counts and per-class recall/F1) were copied
# from the output of the pipeline scripts in this folder and are not recomputed
# here. To change a figure, rerun the relevant pipeline and update the values.
# Figure 2 (well-log plot) comes from generate_logplot_comparison.py.

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
DPI=300

# Figure 1: SEG blind-well confusion matrix (STUART + CRAWFORD)
seg_cm=np.array([[3,9,2,0,0,0,0,0,0],[13,88,11,0,1,1,0,3,0],[3,60,60,0,2,0,0,11,0],[0,0,0,45,4,22,12,4,0],[0,0,1,3,15,19,0,17,0],[0,3,3,6,19,71,4,53,9],[0,1,1,3,4,5,53,26,0],[0,4,4,0,6,19,7,103,2],[0,0,0,0,0,0,0,2,4]])
n=seg_cm/seg_cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(6,5.5));im=ax.imshow(n,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(9));ax.set_yticks(range(9));ax.set_xticklabels(range(1,10));ax.set_yticklabels(range(1,10))
ax.set_xlabel('Predicted facies');ax.set_ylabel('True facies');ax.set_title('SEG blind wells (STUART and CRAWFORD), row-normalized')
for i in range(9):
    for j in range(9):
        if seg_cm[i,j]>0: ax.text(j,i,str(seg_cm[i,j]),ha='center',va='center',color='white' if n[i,j]>0.5 else 'black',fontsize=8)
fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).set_label('Recall (row-normalized)');plt.tight_layout();plt.savefig('fig1_seg_confusion_matrix.png',dpi=DPI,bbox_inches='tight');plt.close()

# Figure 3: FORCE, effect of adding PEF and NPHI (cross-validated F1)
ab={'c':['Chalk','Anhydrite','Limestone','Marl','Tuff','Dolomite','Coal'],'a':[0.307555,0.450131,0.457511,0.283657,0.246603,0.024974,0.291231],'b':[0.383206,0.509340,0.474916,0.293033,0.263379,0.030445,0.265189],'d':[0.353749,0.469932,0.476194,0.285158,0.229974,0.021675,0.344946]}
fig,ax=plt.subplots(figsize=(9,5));x=np.arange(7);w=0.27
ax.bar(x-w,ab['a'],w,label='Baseline (no PEF)',color='#8C8C8C');ax.bar(x,ab['b'],w,label='+ PEF',color='#55A868');ax.bar(x+w,ab['d'],w,label='+ PEF + NPHI',color='#CCB974')
ax.set_xticks(x);ax.set_xticklabels(ab['c'],rotation=20,ha='right');ax.set_ylabel('F1 score');ax.set_title('FORCE: effect of adding PEF and NPHI (cross-validated)');ax.legend();plt.tight_layout();plt.savefig('fig3_pef_nphi_ablation.png',dpi=DPI,bbox_inches='tight');plt.close()

# Figure 4: FORCE leaderboard blind-set confusion matrix
fc=['Sandstone','Ss/Sh','Shale','Marl','Dolomite','Limestone','Chalk','Halite','Anhydrite','Tuff','Coal','Basement']
fm=np.array([[18474,1997,1391,249,11,676,7,0,0,996,247,0],[2737,6665,7323,204,5,232,3,0,0,260,129,0],[2088,9778,65820,3387,6,1463,1,0,0,1047,385,0],[391,649,389,830,0,1003,21,0,0,5,18,0],[15,23,287,17,0,65,0,0,0,3,6,0],[648,544,784,319,3,2238,106,62,0,91,2,1],[0,0,0,2,0,606,17,0,0,0,0,0],[0]*12,[1,4,34,0,0,86,0,0,0,0,0,0],[63,75,503,26,0,10,0,0,0,567,1,0],[65,76,79,2,0,5,0,0,0,0,463,0],[0]*12])
rs=fm.sum(1,keepdims=True);fn=fm/np.where(rs==0,1,rs)
fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(fn,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(12));ax.set_yticks(range(12));ax.set_xticklabels(fc,rotation=45,ha='right');ax.set_yticklabels(fc)
ax.set_xlabel('Predicted lithology');ax.set_ylabel('True lithology');ax.set_title('FORCE leaderboard blind set, row-normalized')
for i in range(12):
    for j in range(12):
        v=fm[i,j]
        if v>0: ax.text(j,i,f'{v}',ha='center',va='center',color='white' if fn[i,j]>0.5 else 'black',fontsize=6 if v<1000 else 7)
        if rs[i,0]==0 and i==j: ax.text(j,i,'n/a',ha='center',va='center',color='gray',fontsize=7,style='italic')
fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).set_label('Recall (row-normalized)');plt.tight_layout();plt.savefig('fig4_force_confusion_matrix.png',dpi=DPI,bbox_inches='tight');plt.close()

# Figure 5: FORCE, cross-validated vs blind-well recall by number of training wells
cw=[('Chalk',11,0.307334,0.0272),('Anhydrite',6,0.376959,0.0),('Dolomite',37,0.038507,0.0),('Coal',51,0.551309,0.671014),('Tuff',55,0.3735,0.455422),('Marl',73,0.352516,0.251059),('Sandstone',98,0.644329,0.768214)]
fig,ax=plt.subplots(figsize=(8,5));x=np.arange(7);w=0.35
ax.bar(x-w/2,[r[2] for r in cw],w,label='Cross-validated recall',color='#4C72B0');ax.bar(x+w/2,[r[3] for r in cw],w,label='Blind-well recall',color='#C44E52')
ax.set_xticks(x);ax.set_xticklabels([f"{r[0]}\n({r[1]} wells)" for r in cw]);ax.set_ylabel('Recall');ax.set_ylim(0,1);ax.set_title('FORCE: cross-validated vs. blind-well recall, by number of training wells');ax.legend();plt.tight_layout();plt.savefig('fig5_cv_vs_blind_recall.png',dpi=DPI,bbox_inches='tight');plt.close()

# Figure 6: SEG per-facies recall, calibrated baseline vs stacked hybrid
b=[0.169884,0.601626,0.588618,0.543478,0.152074,0.495671,0.316327,0.536145,0.788820];s=[0.166023,0.601626,0.582114,0.586957,0.046083,0.554113,0.428571,0.506024,0.366460]
fig,ax=plt.subplots(figsize=(8,5));x=np.arange(9);w=0.35
ax.bar(x-w/2,b,w,label='Calibrated single model',color='#4C72B0');ax.bar(x+w/2,s,w,label='Stacked hybrid',color='#C44E52')
ax.set_xticks(x);ax.set_xticklabels(range(1,10));ax.set_xlabel('Facies');ax.set_ylabel('Recall (leave-one-well-out)');ax.set_title('SEG: per-facies recall, baseline vs. stacked hybrid');ax.legend();plt.tight_layout();plt.savefig('fig6_seg_baseline_vs_nnstack.png',dpi=DPI,bbox_inches='tight');plt.close()
print('done')
