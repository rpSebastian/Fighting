import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import ones
import pandas as pd 

plt.rcParams['font.family'] = 'serif'

def readdata(name):
    """读取数据"""
    root='results/data/%s'%(name)
    taglist=['m0_training_loss','p0_eval_hp_diff','p0_eval_reward',
    'p0_eval_win_rate','p0_hp_diff','p0_reward','p0_win_rate']
    datadict={}
    for tag in taglist:
        path='%s/run-%s-tag-%s.csv'%(root,name,tag)
        val=pd.read_csv(path).loc[:, ['Step','Value']].values
        datadict[tag]=val
    return datadict

def smooth(a,n,mode='padding'):
    """滑动平均,n:滑动窗口尺寸,mode='padding'补全，其余参考concatenate函数"""
    if mode=='padding': #补全模式
        mode='valid'
        inp=np.concatenate((a[0]*np.ones(n//2),a,a[-1]*np.ones(n//2)))
    else:
        inp=a
    out=np.convolve(inp, np.ones((n,))/n, mode=mode)
    return out

def joint(inp1,inp2,smn=5,err=10):
    """拼接inp1与inp2,smn滑动窗口大小,err拼接容差"""
    startidx=np.where(abs(inp2[:,1]-smooth(inp1[:,1],smn)[-1])<err)[0][0]
    inp2[startidx:,0]=inp1[-1,0]+inp2[startidx:,0]-inp2[startidx,0]
    return np.concatenate((inp1,inp2[startidx:,:]))

def ansys(inp,smn=101,zp=0,zerr=5,cons=2,cont=0.4,top=5):
    """数据分析,smn滑动窗口大小,zp零点,zerr零点判断容差,cons收敛区间滑动尺寸,cont收敛判断阈值,top选取前几个点"""
    #平滑
    s=smooth(inp[:,1],smn)
    #零点
    zeroidx=np.where(abs(s-zp)<zerr)[0][0]
    zerostp=inp[zeroidx,0] #零点步数
    #收敛点
    grad=smooth(np.gradient(inp[:,1]),int(inp[:,1].size/cons),'valid')
    conidx=inp[:,1].size-(grad < cont).sum()-1
    conp=inp[conidx,:] #收敛步数
    con=inp[conidx:,1]
    #评价
    conmean=con.mean() #收敛均值
    constd=con.std() #收敛标准差
    conrate=conmean/conp[0] #收敛速率
    maxidx=conidx+con.argsort()[::-1][:top] 
    minidx=conidx+con.argsort()[::1][:top]
    maxp=inp[maxidx,:] #前top个最大点
    minp=inp[minidx,:] #后top个最小点
    return s,zerostp,conp,conmean,constd,conrate,maxp,minp

def latextable(inp,out):
    """输出Latex格式的表格"""
    for i in inp:
        try:
            if(int(i)==i):
                out=out+'%d&'%(i)
            else:
                out=out+'%.3f&'%(i)
        except:
            out=out+'%s&'%(i)
    out=out[:-1]+r'\\'
    return out

def plot_hdrd(inp,inp2,label,title,c,save=True,typ='svg'):
    """HP Diff和Reward"""
    plt.figure(figsize=(8,5))
    #配色方案
    cmap = plt.cm.get_cmap('tab10')
    c=c/10
    #读取评价数据
    if label=='HP Difference':
        s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(inp,cons=2.5)
    else:
        s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(inp,cons=1.5)
    print(zerostp,conp,conmean,constd,conrate)
    #完整数据
    plt.plot(inp[:,0],inp[:,1],color=cmap(c),linewidth=1,alpha = .2,)
    #平滑数据
    plt.plot(inp[:,0],s,color=cmap(c),linewidth=2,label='Training')
    plt.text((conp[0]+inp[:,0].max())/2,conmean*0.8,'Mean:%d'%(conmean),ha='center')
    #波动区间
    plt.fill_between(inp[:,0],smooth(s+abs(inp[:,1]-s),35),smooth(s-abs(inp[:,1]-s),35),color=cmap(c),alpha = 0.3)
    #评价数据
    plt.plot(inp2[:,0],smooth(inp2[:,1],25),'-o',ms = 5, markevery = 10,color=cmap(c+0.1),linewidth=2,label='Evaluation')
    #水平线
    plt.hlines(0, xmin = 0, xmax = inp[:,0][-1],linewidth=2,linestyle = '--',color='black',alpha = .5)
    #零点分割线
    plt.vlines(zerostp, ymin = inp[:,1].min(), ymax = inp[:,1].max(),linewidth=2,linestyle = '--',color='black',alpha = .5)
    plt.text(zerostp,inp[:,1].min(),' ZP:%d'%(zerostp))
    #收敛分割线
    plt.vlines(conp[0], ymin = inp[:,1].min(), ymax = inp[:,1].max(),linewidth=2,linestyle = '--',color='black',alpha = .5)
    plt.text(conp[0],inp[:,1].min(),' ROC:[%d,%d]'%(conp[0],inp[:,0].max()))
    #最值点
    plt.scatter(maxp[:,0], maxp[:,1], marker ="^",s = 20, color = 'r')
    plt.scatter(minp[:,0], minp[:,1], marker ="v",s = 20, color = 'b')
    plt.text(maxp[0,0],maxp[0,1],' %d'%(maxp[0,1]))
    plt.text(minp[0,0],minp[0,1],' %d'%(minp[0,1]))
    #图例等
    title='%s Of %s Method'%(label,title)
    plt.title(title)
    plt.legend(loc=2)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    if save:
        plt.savefig('results/pic/%s.%s'%(title,typ))

def plot_wr(inp,inp2,title,save=True,typ='svg'):
    """Win Rate"""
    plt.figure(figsize=(8,5))
    #评估
    s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(inp2,smn=25,zp=0.5,zerr=0.02)
    print(zerostp,conp,conmean,constd,conrate)
    #评估线
    plt.plot(inp2[:,0],s,'-o',ms = 5, markevery = 10,color='black',linewidth=2,label='Evaluation Of Win Rate')
    plt.text((conp[0]+inp2[:,0].max())/2,conmean*0.9,'Mean:%.2f'%(conmean),ha='center')
    #水平线
    plt.hlines(0.5, xmin = 0, xmax = inp2[:,0][-1],linewidth=2,linestyle = '--',color='black',alpha = .5)
    #零点分割线
    plt.vlines(zerostp, ymin = inp2[:,1].min(), ymax = inp2[:,1].max(),linewidth=2,linestyle = '--',color='black',alpha = .5)
    plt.text(zerostp,inp2[:,1].min()+0.02,' ZP:%d'%(zerostp))
    #收敛分割线
    plt.vlines(conp[0], ymin = inp2[:,1].min(), ymax = inp2[:,1].max(),linewidth=2,linestyle = '--',color='black',alpha = .5)
    plt.text(conp[0],inp2[:,1].min()+0.02,' ROC:[%d,%d]'%(conp[0],inp2[:,0].max()))
    #输赢统计
    winidx=np.where(inp[:,1]==1)[0]
    loseidx=np.where(inp[:,1]==0)[0]
    plt.bar(inp[winidx,0],inp[winidx,1]-0.5,bottom=0.5,width=20,color='crimson',alpha=.1,label='Win')
    plt.bar(inp[loseidx,0],inp[loseidx,1]+0.5,width=20,color='dodgerblue',alpha=.1,label='Lose')
    #图例等
    title='Win Rate Of %s Method'%(title)
    plt.title(title,y=1.08)
    plt.legend(bbox_to_anchor=(0.5, 1.05),loc=10,ncol=5,prop = {'size':10})
    plt.xlabel("Epoch")
    plt.ylabel('Win Rate')
    if save:
        plt.savefig('results/pic/%s.%s'%(title,typ))

def plot_loss(inp,title,c,save=True,typ='svg'):
    """Loss"""
    plt.figure(figsize=(8,5))
    #配色方案
    cmap = plt.cm.get_cmap('tab10')
    c=c/10
    #完整数据
    plt.plot(inp[:,0],inp[:,1],color=cmap(c),linewidth=1,alpha = .2,)
    #平滑数据
    s=smooth(inp[:,1],101)
    plt.plot(inp[:,0],s,color=cmap(c),linewidth=2)
    #波动区间
    plt.fill_between(inp[:,0],smooth(s+abs(inp[:,1]-s),35),smooth(s-abs(inp[:,1]-s),35),color=cmap(c),alpha = 0.3)
    #图例等
    title='Training Loss Of %s Method'%(title)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()
    if save:
        plt.savefig('results/pic/%s.%s'%(title,typ))

def plot_cmp(inpdict,label,c='Set2',save=True,typ='pdf'):
    """多方法对比"""
    plt.figure(figsize=(8,5))
    #配色方案
    cmap = plt.cm.get_cmap(c)
    marklist= ('o', 'v', 's', 'p', '*','D','X','^', '<', '>', '8', 'h', 'H', 'd', 'P' )
    #绘图
    idx=0
    out=r"""\begin{tabular}{cccccccc}
    \hline
    方法&零点&收敛点&收敛均值&收敛最大值&收敛最小值&收敛标准差&收敛速率\\
    \hline
    """

    for k,v in inpdict.items():
        #一些参数调整
        if label=='HP Difference':
            s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(v,smn=25,cons=15,zerr=8)
        elif label=='Reward':
            s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(v,smn=25,cons=1.4,zerr=1)
        elif label=='Win Rate':
            s,zerostp,conp,conmean,constd,conrate,maxp,minp=ansys(v,smn=25,zp=0.5,cons=2,zerr=0.05)
        
        out=latextable([k.replace('_',' '),zerostp,conp[0],conmean,maxp[0,1],minp[0,1],constd,conrate*100],out)
        
        plt.plot(v[:,0],s,'-%s'%(marklist[idx]),ms = 6, markevery = 20,color=cmap(idx/8),linewidth=2,label=k)
        scal=v[:,1].max()-v[:,1].min()
        if label=='Win Rate':
            plt.vlines(zerostp, ymin = 0.5-scal*0.1, ymax = 0.5+scal*0.1,linewidth=2,linestyle = '--',color=cmap(idx/8),alpha = .5)
        else:
            plt.vlines(zerostp, ymin = -scal*0.1, ymax = scal*0.1,linewidth=2,linestyle = '--',color=cmap(idx/8),alpha = .5)
        plt.vlines(conp[0], ymin = conp[1]-scal*0.1, ymax = conp[1]+scal*0.1,linewidth=2,linestyle = '--',color=cmap(idx/8),alpha = .5)
        idx+=1

    out=out+r"""
    \hline
    \end{tabular}"""
    #输出Latex Table
    print(out)
    #图例等
    title='%s Of Different Method'%(label)
    plt.title(title)
    plt.legend(loc=4)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel(label)
    if save:
        plt.savefig('results/pic/%s.%s'%(title,typ))


"""----消融实验绘图----"""
#DQN
# d=readdata('baseline')
# d['p0_eval_win_rate'][:,0]=d['p0_eval_win_rate'][:,0]*d['p0_win_rate'][-1,0]/d['p0_eval_win_rate'][-1,0]
# d['p0_eval_hp_diff'][:,0]=d['p0_eval_hp_diff'][:,0]*d['p0_hp_diff'][-1,0]/d['p0_eval_hp_diff'][-1,0]
# d['p0_eval_reward'][:,0]=d['p0_eval_reward'][:,0]*d['p0_reward'][-1,0]/d['p0_eval_reward'][-1,0]
# plot_loss(d['m0_training_loss'],'DQN',0)
# plot_wr(d['p0_win_rate'],d['p0_eval_win_rate'],'DQN')
# plot_hdrd(d['p0_hp_diff'],d['p0_eval_hp_diff'],'HP Difference','DQN',0)
# plot_hdrd(d['p0_reward'],d['p0_eval_reward'],'Reward','DQN',1)

#Double
# d=readdata('double long')
# plot_loss(d['m0_training_loss'],'Double',1)
# plot_wr(d['p0_win_rate'],d['p0_eval_win_rate'],'Double')
# plot_hdrd(d['p0_hp_diff'],d['p0_eval_hp_diff'],'HP Difference','Double',1)
# plot_hdrd(d['p0_reward'],d['p0_eval_reward'],'Reward','Double',2)

#Double+Dueling
# d=readdata('double_dueling')
# plot_loss(d['m0_training_loss'],'Double+Dueling',2)
# plot_wr(d['p0_win_rate'],d['p0_eval_win_rate'],'Double+Dueling')
# plot_hdrd(d['p0_hp_diff'],d['p0_eval_hp_diff'],'HP Difference','Double+Dueling',2)
# plot_hdrd(d['p0_reward'],d['p0_eval_reward'],'Reward','Double+Dueling',3)

#Double+Dueling+Noisy
# d=readdata('noisy_double_dueling_long')
# plot_loss(d['m0_training_loss'],'Double+Dueling+Noisy',3)
# plot_wr(d['p0_win_rate'],d['p0_eval_win_rate'],'Double+Dueling+Noisy')
# plot_hdrd(d['p0_hp_diff'],d['p0_eval_hp_diff'],'HP Difference','Double+Dueling+Noisy',3)
# plot_hdrd(d['p0_reward'],d['p0_eval_reward'],'Reward','Double+Dueling+Noisy',0)

#Double+Dueling+Noisy+NSteps
# d11=readdata('noisy_double_dueling_nsteps')
# d12=readdata('noisy_double_dueling_nsteps_continue')
# d13=readdata('noisy_double_dueling_nsteps_continue_2')
# tagname1=['p0_hp_diff','p0_reward','p0_win_rate','m0_training_loss']
# tagname2=['p0_eval_hp_diff','p0_eval_reward','p0_eval_win_rate']
# pend1=[0,160,10,0]
# pend2=[0,0,10,0]
# for i in range(4):
#     dj11=joint(d11[tagname1[i]],d12[tagname1[i]][pend1[i]:],smn=4,err=2)
#     dj12=joint(dj11,d13[tagname1[i]][pend1[i]:],smn=4,err=2)
#     if(i<3):
#         dj21=joint(d11[tagname2[i]],d12[tagname2[i]][pend2[i]:],smn=4,err=2)
#         dj22=joint(dj21,d12[tagname2[i]][pend2[i]:],smn=2,err=1)
#     if i==0:
#         plot_hdrd(dj12,dj22,'HP Difference','Double+Dueling+Noisy+NSteps',0)
#     elif i==1:
#         plot_hdrd(dj12,dj22,'Reward','Double+Dueling+Noisy+NSteps',1)
#     elif i==2:
#         plot_wr(dj12,dj22,'Double+Dueling+Noisy+NSteps')
#     elif i==3:
#         plot_loss(dj12,'Double+Dueling+Noisy+NSteps',0)

#Double+Dueling+Noisy+NSteps+Categorical
# d=readdata('all_new_long')
# plot_loss(d['m0_training_loss'],'Double+Dueling+Noisy+NSteps+Categorical',1)
# plot_wr(d['p0_win_rate'],d['p0_eval_win_rate'],'Double+Dueling+Noisy+NSteps+Categorical')
# plot_hdrd(d['p0_hp_diff'],d['p0_eval_hp_diff'],'HP Difference','Double+Dueling+Noisy+NSteps+Categorical',1)
# plot_hdrd(d['p0_reward'],d['p0_eval_reward'],'Reward','Double+Dueling+Noisy+NSteps+Categorical',2)

"""----方案对比图----"""

tagname=['p0_eval_hp_diff','p0_eval_reward','p0_eval_win_rate']
labellist=['HP Difference','Reward','Win Rate']
pend1=[5,5,1]
pend2=[0,0,10]
for i in range(3):
    inpdict={}
    pathlist=['baseline','double long','double_dueling','noisy_double_dueling_long','all_new_long']
    namelist=['DQN','+Double','+Dueling','+Noisy','+Categorical']
    for d in pathlist[:-1]:
        v=readdata(d)
        inpdict[namelist[pathlist.index(d)]]=v[tagname[i]]

    d11=readdata('noisy_double_dueling_nsteps')
    d12=readdata('noisy_double_dueling_nsteps_continue')
    d13=readdata('noisy_double_dueling_nsteps_continue_2')
    dj1=joint(d11[tagname[i]],d12[tagname[i]][pend2[i]:],smn=4,err=2)
    dj2=joint(dj1[:-pend1[i]],d13[tagname[i]][pend2[i]:],smn=2,err=1)
    inpdict['+NSteps']=dj2
    v=readdata(pathlist[-1])
    inpdict[namelist[-1]]=v[tagname[i]]
    plot_cmp(inpdict,labellist[i],save=False)


plt.show()