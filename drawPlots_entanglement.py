# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx

from data_io import import_all_log, parse_path_string    
from scipy.optimize import curve_fit
def power_law(x,a,b):
    return a*x**b


# %%
single_column_size = (2.5,1.75)


pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N125-AR25-Scale1_20240531-222435')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N250-AR50-Scale1_20240531-222435')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N375-AR75-Scale1_20240531-222436')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N500-AR100-Scale1_20240531-222436')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N625-AR125-Scale1_20240531-222434')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1000-AR200-Scale1_20240603-131308')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1500-AR300-Scale1_20240603-000746')

# %%
pathlist2 = []

pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N1000-AR200')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N1500-AR300')

class data_container:
    def __init__(self,dataphat,max_rows=100000):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)
        
max_rows = 1000000
data_container_list = []
for pth in pathlist2:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list.append(data_container(file,max_rows=max_rows))
        break

# %%
Ns = []
ARs = []
for pth in pathlist:
    search_result = re.search(r'N(\d+)[-_]AR(\d+)',pth)
    Ns.append(int(search_result.group(1)))
    ARs.append(int(search_result.group(2)))
# %%
output_dir = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/entanglement'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# %%
class entanglement_data_container:
    def __init__(self,dataphat):
        self.path = Path(dataphat)
        self.dataobj = np.load(self.path)
        self.total_entanglement_over_time = self.dataobj['total_entanglement_over_time']
        self.time = np.linspace(0,5,len(self.total_entanglement_over_time))
        
entanglement_data_container_list = []

for pth in pathlist:
    for datafile in Path(pth).rglob('**/*.npz'):
        print(datafile)
    entanglement_data_container_list.append(entanglement_data_container(datafile))
    
# %%
entanglement_data_container_list[0].dataobj.files
    
# %%
len(entanglement_data_container_list)
# %%

for i_,dc in enumerate(entanglement_data_container_list):
    N = Ns[i_]
    plt.plot(dc.time,dc.total_entanglement_over_time/N**2,label=f'AR={ARs[i_]}')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Normalized total entanglement')
plt.savefig(f'{output_dir}/entanglement-over-time.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
for i_,dc in enumerate(data_container_list):
    N = Ns[i_]
    contact_list = dc.contact_list
    contact_list = contact_list[:667]
    tt = dc.time_line[:667]
    
    num_contacts = [x for x in map(lambda x: len(x)//18,contact_list)]
    num_contacts = np.array(num_contacts)
    plt.plot(tt[::1],num_contacts[::1]/N*2,label=f'AR={ARs[i_]}',linewidth=0.5)
plt.legend(loc='lower right',fontsize=6)
plt.xlabel('$t$ (sec)')
plt.ylabel(r'$C/N$')
plt.xlim([0,10])
plt.savefig(f'{output_dir}/avg-number-of-contacts-per-rod.png',dpi=300,bbox_inches='tight')
# %%
coef_vars = []
for i_,dc in enumerate(entanglement_data_container_list):
    coef_var = {}
    efield = dc.dataobj['e_fields_over_time']
    cfield = dc.dataobj['c_fields_over_time']
    
    efield = efield[:667]
    cfield = cfield[:667]
    
    tt = np.linspace(0,5,len(efield))
    
    e_avg = np.nanmean(efield,axis=1)
    e_std = np.nanstd(efield,axis=1)
    
    c_avg = np.nanmean(cfield,axis=1)
    c_std = np.nanstd(cfield,axis=1)
    
    coef_var['e'] = e_std/e_avg
    coef_var['c'] = c_std/c_avg
    
    coef_vars.append(coef_var)
    
    # plt.plot(tt,e_std/e_avg,label=f'AR={ARs[i_]}',linewidth=0.5)
# %%
fig,axs=plt.subplots(1,2,figsize=(12,6))
for i_,cv in enumerate(coef_vars):
    axs[0].plot(tt,cv['e'],label=f'AR={ARs[i_]}',linewidth=0.5)
    axs[1].plot(tt,cv['c'],label=f'AR={ARs[i_]}',linewidth=0.5)    

axs[0].set_xlabel('$t$ (sec)')
axs[0].set_ylabel(r'$\sigma/\mu$')
axs[1].set_xlabel('$t$ (sec)')
axs[1].set_ylabel(r'$\sigma/\mu$')
axs[1].set_ylim([0,2])
plt.legend(loc='lower right',fontsize=6)

plt.savefig(f'{output_dir}/coef-var-over-time.png',dpi=300,bbox_inches='tight')

# %%


# %%
avg_entanglement = []
std_entanglement = []
for dc in entanglement_data_container_list:
    efield = dc.dataobj['e_fields_over_time']
    e_avg = np.nanmean(efield)
    e_std = np.nanstd(efield)
    
    avg_entanglement.append(e_avg)
    std_entanglement.append(e_std)
avg_entanglement = np.array(avg_entanglement)
std_entanglement = np.array(std_entanglement)
# %%
ARs = np.array([25,50,75,100,125,200,300])
plt.errorbar(ARs,avg_entanglement,yerr=std_entanglement,fmt='o')

# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

single_column_size = (2.5,1.75)
fig,ax=plt.subplots(1,1,figsize=single_column_size)
ax.plot(ARs,std_entanglement/avg_entanglement,'o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma/\mu$')
plt.savefig(f'{output_dir}/entanglement-sigma-over-mu.png',dpi=300,bbox_inches='tight')

# ax.set_xticks([25,50,100,200,300])
# ax.set_yticks([0.9,1.4])
# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
bins_for_log = np.logspace(-3,5,100)
# default color order
clr_order = plt.rcParams['axes.prop_cycle'].by_key()['color']

histogram_results = []

for i_,dc in enumerate(entanglement_data_container_list):
    N = Ns[i_]
    phi_fields = dc.dataobj['phi_fields_over_time']
    phi_fields = phi_fields[:667]
    e_fields = dc.dataobj['e_fields_over_time']
    e_fields = e_fields[:667]
    c_fields = dc.dataobj['c_fields_over_time']
    c_fields = c_fields[:667]
    
    phi_field_last_frame = phi_fields[-1]
    e_field_last_frame = e_fields[-1]
    c_field_last_frame = c_fields[-1]
    # dta = e_field_last_frame[~np.isnan(e_field_last_frame)]
    
    hist_data = {}
    
    dta = e_field_last_frame[e_field_last_frame>0]
    n,x,_ = ax.hist(dta,bins=bins_for_log,density=True,label=f'AR={ARs[i_]}',facecolor='none',edgecolor=clr_order[i_], linewidth=0.5)
    ax.clear()
    
    hist_data['e_field'] = (n,x)
    
    dta = c_field_last_frame[e_field_last_frame>0]
    n,x,_ = ax.hist(dta,bins=bins_for_log,density=True,label=f'AR={ARs[i_]}',facecolor='none',edgecolor=clr_order[i_], linewidth=0.5)
    
    hist_data['c_field'] = (n,x)
    histogram_results.append(hist_data)
    
    hist_data['max_e'] = np.nanmax(e_field_last_frame)
    hist_data['max_c'] = np.nanmax(c_field_last_frame)
    
    
ax.set_yscale('log')
ax.set_xscale('log')
# %%
# plot max e and c
max_e = []
max_c = []
for hist_result in histogram_results:
    max_e.append(hist_result['max_e'])
    max_c.append(hist_result['max_c'])
max_e = np.array(max_e)
max_c = np.array(max_c)

fig,axs=plt.subplots(1,2,figsize=(8,6))
axs[0].plot(ARs,max_e,'o')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$\max(e)$')
axs[1].plot(ARs,max_c,'o')
axs[1].set_xlabel(r'$\alpha$')
axs[1].set_ylabel(r'$\max(c)$')
plt.savefig(f'{output_dir}/max-e-max-c.png',dpi=300,bbox_inches='tight')

# %%





# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})

markers = ['o','s','^','v','<','>','D']
fig,ax=plt.subplots(1,1,figsize=(8,6))
# fontsize
plt.rcParams.update({'font.size': 8})
for i_,hist_result in enumerate(histogram_results):
    n,x = hist_result['e_field']
    
    xx = x[:-1]
    yy = n
    xx = xx[yy>0]
    yy = yy[yy>0]

    
    ax.loglog(xx,yy,markers[i_],markersize=3)
    clr = ax.get_lines()[-1].get_color()
    # popt,pcov=curve_fit(power_law,xx,yy)
    # ax.loglog(xx,power_law(xx,*popt))
    # print(popt)
    
    logx = np.log(xx)
    logy = np.log(yy)
    p = np.polyfit(logx,logy,1)
    
    lgd = r'$\alpha = %d$, $y = %.2f x^{%.2f}$' % (ARs[i_],np.exp(p[1]),p[0])
    ax.loglog(xx,np.exp(p[1])*xx**p[0],color=clr,label=lgd)

ax.set_xlabel(r'$e(\mathbf{x})$')
ax.set_ylabel('PDF')
plt.legend()
plt.savefig(f'{output_dir}/entanglement-power-law-distribution.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=(8,6))
# fontsize
plt.rcParams.update({'font.size': 8})
for i_,hist_result in enumerate(histogram_results):
    n,x = hist_result['c_field']
    
    xx = x[:-1]
    yy = n
    xx = xx[yy>0]
    yy = yy[yy>0]

    
    ax.loglog(xx,yy,'o-',markersize=3,label=f'AR={ARs[i_]}')
    clr = ax.get_lines()[-1].get_color()
    # popt,pcov=curve_fit(power_law,xx,yy)
    # ax.loglog(xx,power_law(xx,*popt))
    # print(popt)
    
    logx = np.log(xx)
    logy = np.log(yy)
    p = np.polyfit(logx,logy,1)
    
    lgd = r'$\alpha = %d$, $y = %.2f x^{%.2f}$' % (ARs[i_],np.exp(p[1]),p[0])
    # ax.loglog(xx,np.exp(p[1])*xx**p[0],color=clr,label=lgd)

ax.set_xlabel(r'$c(\mathbf{x})$')
ax.set_ylabel('PDF')
plt.legend()
plt.savefig(f'{output_dir}/contact-power-law-distribution.png',dpi=300,bbox_inches='tight')
    
# %%    



# %%
for i_,dc in enumerate(entanglement_data_container_list):
    N = Ns[i_]
    c_fields = dc.dataobj['c_fields_over_time']
    c_fields = c_fields[:667]
    
    c_field_last_frame = c_fields[-1]
    
    plt.hist(c_field_last_frame,bins=100,density=True,label=f'AR={ARs[i_]}',alpha=0.5)
    
# %%
frac_over_time = []
for dc,edc in zip(data_container_list,entanglement_data_container_list):
    N = dc.num_rods
    AR = dc.AR
    
    final_contact_list = dc.contact_list[667]
    final_contact_list = final_contact_list.reshape(-1,18)
    
    contact_ij = final_contact_list[:,4:6].astype(int)
    
    contact_graph = nx.Graph()
    contact_graph.add_nodes_from(np.arange(N))
    contact_graph.add_edges_from(contact_ij)
    
    clusters = list(nx.connected_components(contact_graph))
        
    # largest clusters
    largest_cluster = max(clusters,key=len)
    frac_over_time.append(len(largest_cluster)/N)
# %%

    
    

# %%

    
    

