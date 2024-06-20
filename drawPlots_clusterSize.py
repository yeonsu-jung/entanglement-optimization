# %%
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from pathlib import Path
from matplotlib.patches import Polygon

def power_law(x,a,b):
    return a*x**b
def half_power_law(x,a):
        return a*x**0.5
def linear(x,a):
    return a*x
single_column_size = (2.5,1.75)
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def _draw(output_dict_list_repeated, savepath):
    single_column_size = (2.5,1.75)
    inset_size = (1.5,1.5)
    
    # use tex
    # plt.rc('text', usetex=True)
    
    rep_vals_rep = []
    fig,ax=plt.subplots(1,1,figsize=single_column_size)
    legend_count = 0
    for output_dict_list in output_dict_list_repeated:
        output_dict = output_dict_list[0]        
        ARs = []
        rep_vals = []
        for output_dict in output_dict_list:
            ARs.append(output_dict['AR'])
            
            tt = output_dict['actual_timeline']
            vv = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
            rep_vals.append(vv[-1])
            
            if legend_count < 18:
                ax.loglog(tt,vv,label=f'AR={output_dict["AR"]}')
                legend_count += 1
            else:
                ax.plot(tt,vv)                
        rep_vals_rep.append(rep_vals)
    plt.savefig(f'{savepath}/raw.png',dpi=300)
            
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for rep_vals in rep_vals_rep:
        ax.loglog(ARs,np.array(rep_vals),'o-')
    ax.axvline(100,linestyle='--',color='k')
    plt.savefig(f'{savepath}/raw2.png',dpi=300)

    avg_dta = np.mean(rep_vals_rep,axis=0)
    err_dta = np.std(rep_vals_rep,axis=0)

    ## 
    fig,ax=plt.subplots(1,1,figsize=single_column_size)
    ax.errorbar(ARs,avg_dta,err_dta,fmt='o-')
    popt,pcov = curve_fit(power_law,ARs,avg_dta)
    smoothx = np.linspace(ARs[0],ARs[-1],100)
    ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    cutoff = 100
    ARs = np.array(ARs)
    ARs_under = ARs[ARs < cutoff]
    avg_dta_under = avg_dta[ARs < cutoff]
    err_dta_under = err_dta[ARs < cutoff]

    ## fraction over time
    fig,ax=plt.subplots(1,1,figsize=single_column_size)
    plt.rcParams.update({'font.size': 8})
    ax.errorbar(ARs_under,avg_dta_under,err_dta_under,fmt='.')
    clr = ax.get_lines()[-1].get_color()
    popt,pcov = curve_fit(power_law,ARs_under,avg_dta_under)
    smoothx = np.linspace(ARs_under[0],ARs_under[-1],100)
    ax.plot(smoothx,power_law(smoothx,*popt),'--',label=fr'${popt[0]:.2f}x^{{{popt[1]:.2f}}}$',color=clr)

    ARs_over = ARs[ARs > cutoff]
    avg_dta_over = avg_dta[ARs > cutoff]
    err_dta_over = err_dta[ARs > cutoff]
    popt,pcov = curve_fit(power_law,ARs_over,avg_dta_over)
    smoothx = np.linspace(ARs_over[0],ARs_over[-1],100)
    ax.errorbar(ARs_over,avg_dta_over,err_dta_over,fmt='.')
    clr = ax.get_lines()[-1].get_color()
    ax.plot(smoothx,power_law(smoothx,*popt),'--',label=fr'${popt[0]:.2f}x^{{{popt[1]:.2f}}}$',color=clr)
    
    x_start = 150
    x_end = 250
    
    ax.axvline(100,linestyle='--',color='k',linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # axis linewidth
    ax.spines['bottom'].set_linewidth(0.5)

    plt.xlabel('$\\alpha$')
    plt.ylabel('$f$')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{savepath}/fraction_over_time.png',dpi=300)

    xx = np.abs(ARs_under - 100)
    yy = 1-avg_dta_under

    fig,ax=plt.subplots(1,1,figsize=(2,1.5))
    plt.rcParams.update({'font.size': 6})
    ax.plot(xx,yy,'.')
    ax.set_xlabel('$|\\alpha - 100|$')
    ax.set_ylabel('$1 - f$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    popt,pcov = curve_fit(half_power_law,xx,yy)    
    smoothx = np.linspace(xx[0],xx[-1],100)
    ax.plot(smoothx,half_power_law(smoothx,*popt),'--',linewidth=0.5,label=fr'${popt[0]:.2f}x^{{0.5}}$')
    
    
    popt,pcov=curve_fit(linear,xx,yy)
    ax.plot(smoothx,linear(smoothx,*popt),'--',linewidth=0.5,label=fr'${popt[0]:.2f}x$')
    
    
    popt,pcov=curve_fit(power_law,xx,yy)
    ax.plot(smoothx,power_law(smoothx,*popt),'--',linewidth=0.5,label=fr'${popt[0]:.2f}x^{{{popt[1]:.2f}}}$')
    
    ax.set_xticks([5,10,100])
    ax.set_yticks([0.1,1])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{savepath}/near_transition.png',dpi=300)

output_root = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'
# %%
pklpath = Path(f'{output_root}/Micromechanics-HangModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
savepath = f'{output_root}/{pklpath.parent.stem}'
os.makedirs(savepath,exist_ok=True)

_draw(output_dict_list_repeated,savepath)
# %%
pklpath = Path(f'{output_root}/Micromechanics-TickleModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
savepath = f'{output_root}/{pklpath.parent.stem}'
os.makedirs(savepath,exist_ok=True)

_draw(output_dict_list_repeated,savepath)
# %%
pklpath = Path(f'{output_root}/Micromechanics-TabModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
savepath = f'{output_root}/{pklpath.parent.stem}'
os.makedirs(savepath,exist_ok=True)
_draw(output_dict_list_repeated,savepath)
# %%

pklpath = Path(f'{output_root}/Micromechanics-HangModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated_hang = pickle.load(f)
    
pklpath = Path(f'{output_root}/Micromechanics-TickleModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated_tickle = pickle.load(f)
    
pklpath = Path(f'{output_root}/Micromechanics-TabModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated_tab = pickle.load(f)
    
pklpath = Path(f'{output_root}/Micromechanics-HeavyHangModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_repeated_heavyhang = pickle.load(f)
    
# %%

def _draw_f_plot(output_dict_list_repeated,ax,**kwargs):
    rep_vals_rep = []
    for output_dict_list in output_dict_list_repeated:
        
        ARs = []
        rep_vals = []
        
        for output_dict in output_dict_list:
            ARs.append(output_dict['AR'])
            tt = output_dict['actual_timeline']
            vv = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
            rep_vals.append(vv[-1])
        rep_vals_rep.append(rep_vals)
        
    rep_vals_rep = np.array(rep_vals_rep)
    avg_dta = np.mean(rep_vals_rep,axis=0)
    std_dta = np.std(rep_vals_rep,axis=0)
    
    ARs = np.array(ARs)
    
    # fig,ax=plt.subplots(1,1,figsize=single_column_size)
    ARs_under = ARs[ARs < 100]
    avg_dta_under = avg_dta[ARs < 100]
    std_dta_under = std_dta[ARs < 100]

    ARs_over = ARs[ARs >= 100]
    avg_dta_over = avg_dta[ARs >= 100]
    std_dta_over = std_dta[ARs >= 100]

    ax.errorbar(ARs_under,avg_dta_under,std_dta_under,**kwargs)
    clr = ax.get_lines()[-1].get_color()
    ax.errorbar(ARs_over,avg_dta_over,std_dta_over,color=clr,**kwargs)
    
single_column_size = (2.5*0.8,1.75*0.8)
omega = np.array([100,1])
A = 0.001
Fr = 4*np.pi**2*omega**2*A/0.5

ARs = []
for output_dict in output_dict_list_repeated_hang[0]:
    ARs.append(output_dict['AR'])
ARs_under = np.array(ARs)[np.array(ARs) < 100]
ARs_over = np.array(ARs)[np.array(ARs) >= 100]

figure_output_root = f'{output_root}/figures'
if not os.path.exists(figure_output_root):
    os.makedirs(figure_output_root)
fig,ax=plt.subplots(1,1,figsize=single_column_size)


ax.plot(ARs_under,np.ones_like(ARs_under),'.-',markersize=3,alpha=0.7,label='$a/g = 0$')
clr = ax.get_lines()[-1].get_color()
ax.plot(ARs_over,np.ones_like(ARs_over),'.-',markersize=3,alpha=0.7,label='$a/g = 0$',color=clr)

_draw_f_plot(output_dict_list_repeated_tickle,ax,fmt='s-',markersize=3,alpha=0.7,label=r'$a/g = 7.90\times 10^{-2}$')
_draw_f_plot(output_dict_list_repeated_tab,ax,fmt='o-',markersize=3,alpha=0.7,label=r'$a/g = 7.90\times 10^2$')

ax.axvline(100,linestyle='--',color='k',linewidth=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$f$')

# plt.legend()
# show legends for 1 3 5 
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[i] for i in [0,2,4]], [labels[i] for i in [0,2,4]],loc='lower right',fontsize=6)


plt.savefig(f'{figure_output_root}/f_plot.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
ax.plot(ARs_under,np.ones_like(ARs_under),'.-',markersize=3,alpha=0.7,label=r'$F/\rho_s g d^2 l = 0$')
clr = ax.get_lines()[-1].get_color()
ax.plot(ARs_over,np.ones_like(ARs_over),'.-',markersize=3,alpha=0.7,label=r'$F/\rho_s g d^2 l = 0$',color=clr)

_draw_f_plot(output_dict_list_repeated_hang,ax,fmt='s-',markersize=3,alpha=0.7,label=r'$F/\rho_s g d^2 l = 1$')
_draw_f_plot(output_dict_list_repeated_heavyhang,ax,fmt='o-',markersize=3,alpha=0.7,label=r'$F/\rho_s g d^2 l = 20$')
ax.axvline(100,linestyle='--',color='k',linewidth=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$f$')

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[i] for i in [0,2,4]], [labels[i] for i in [0,2,4]],loc='lower right',fontsize=6)

plt.savefig(f'{figure_output_root}/f-plot-for-hangtest.png',dpi=300,bbox_inches='tight')

# %%

# %%




    
# savepath = f'{output_root}/{pklpath.parent.stem}'
# os.makedirs(savepath,exist_ok=True)

# _draw(output_dict_list_repeated,savepath)




# %%
# pklpath = Path(f'{output_root}/Micromechanics-HangModelos/output_dict_list_repeated.pkl')
# pklpath = Path(f'{output_root}/Micromechanics-TickleModelos/output_dict_list_repeated.pkl')
# pklpath = Path(f'{output_root}/Micromechanics-TabModelos/output_dict_list_repeated.pkl')

