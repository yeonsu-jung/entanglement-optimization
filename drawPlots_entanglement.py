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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# %%
class data_container:
    def __init__(self,dataphat,start_row=0,max_rows=100000,skip_rows=1):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,start_row=start_row,max_rows=max_rows,skip_rows=skip_rows)
        

# %%
single_column_size = (2.5,1.75)


pathlist = []

pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0125-AR025-Scale1_20240609-010903')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0250-AR050-Scale1_20240609-010903')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0375-AR075-Scale1_20240609-010902')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0500-AR100-Scale1_20240609-010902')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0525-AR105-Scale1_20240609-010902')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0550-AR110-Scale1_20240609-010902')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0575-AR115-Scale1_20240609-010902')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0600-AR120-Scale1_20240609-010858')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0625-AR125-Scale1_20240609-010858')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0750-AR150-Scale1_20240609-010858')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N0875-AR175-Scale1_20240609-010858')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N1000-AR200-Scale1_20240609-010856')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N1250-AR250-Scale1_20240609-010856')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/CheckEntangleModelo1/NonIntersectingBox-N1500-AR300-Scale1_20240609-010856')


# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N125-AR25-Scale1_20240531-222435')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N250-AR50-Scale1_20240531-222435')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N375-AR75-Scale1_20240531-222436')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N500-AR100-Scale1_20240531-222436')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N625-AR125-Scale1_20240531-222434')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1000-AR200-Scale1_20240603-131308')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1500-AR300-Scale1_20240603-000746')

# %%
pathlist2 = []


pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N1000-AR200')
pathlist2.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N1500-AR300')




        
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
pathlist3 = []
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0050_RUN_HangEECarrotCake5_N125_AR25')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N250_AR50')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N375_AR75')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0050_RUN_HangEECarrotCake5_N500_AR100')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N625_AR125')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N1000_AR200')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N1500_AR300')
# %%
pathlist3 = []

# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0875_AR175')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0575_AR115')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0600_AR120')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0625_AR125')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0750_AR150')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N1250_AR250')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0550_AR110')
# pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0525_AR105')

pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0125_AR025')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0250_AR050')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0375_AR075')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0500_AR100')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0625_AR125')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N1000_AR200')
pathlist3.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1755_RUN_HangEEModelo1_N1500_AR300')


data_container_list_hang = []
for pth in pathlist3:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list_hang.append(data_container(file,max_rows=max_rows))
        break
# %%
pathlist4 = []
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1051_RUN_PerturbEEModelo1_N0125_AR025')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0250_AR050')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0375_AR075')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0500_AR100')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0525_AR105')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0550_AR110')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0575_AR115')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0600_AR120')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0625_AR125')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0750_AR150')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0875_AR175')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1000_AR200')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1250_AR250')
pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240611-0134_RUN_PerturbEEModelo1_N0450_AR090')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240611-0135_RUN_PerturbEEModelo1_N0300_AR060')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240611-0135_RUN_PerturbEEModelo1_N0350_AR070')
# pathlist4.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240611-0135_RUN_PerturbEEModelo1_N0400_AR080')


data_container_list_tickle = []
for pth in pathlist4:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list_tickle.append(data_container(file,max_rows=max_rows))
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

fig,ax=plt.subplots(1,1,figsize=single_column_size)
# font size
plt.rcParams.update({'font.size': 8})

for i_,dc in enumerate(entanglement_data_container_list):
    N = Ns[i_]
    plt.plot(dc.time,dc.total_entanglement_over_time/N**2,label=fr'$\alpha={ARs[i_]}$')
plt.legend(fontsize=6)
plt.xlabel('Time (sec)')
plt.ylabel('Normalized total entanglement')
plt.savefig(f'{output_dir}/entanglement-over-time.png',dpi=300,bbox_inches='tight')

# %%
    

# %%
last_part_averaged = []
fig,ax=plt.subplots(1,1,figsize=single_column_size)
for i_,dc in enumerate(data_container_list):
    N = Ns[i_]
    contact_list = dc.contact_list
    contact_list = contact_list[:667]
    tt = dc.time_line[:667]
    
    num_contacts = [x for x in map(lambda x: len(x)//18,contact_list)]
    num_contacts = np.array(num_contacts)
    plt.plot(tt[::1],num_contacts[::1]/N*2,label=fr'$\alpha={ARs[i_]}$',linewidth=0.5)
    
    last_part_averaged.append(np.mean(num_contacts[-50:])/N*2)


plt.xlabel(r'$t$ (sec)')
plt.ylabel(r'$Z$')

plt.legend(loc='lower right',fontsize=4)
# plt.xlim([0,10])
plt.savefig(f'{output_dir}/avg-number-of-contacts-per-rod.png',dpi=300,bbox_inches='tight')
# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
plt.plot(last_part_averaged,'o-')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$Z$')
plt.savefig(f'{output_dir}/Z-over-alpha.png',dpi=300,bbox_inches='tight')


# %%
coef_vars = []
means = []
ARs = []
for i_,dc in enumerate(entanglement_data_container_list):
    AR = int(re.search(r'AR(\d+)',pathlist[i_]).group(1))
    ARs.append(AR)
    
    coef_var = {}
    efield = dc.dataobj['e_fields_over_time']
    cfield = dc.dataobj['c_fields_over_time']
    
    efield = efield
    cfield = cfield
    
    tt = np.linspace(0,5,len(efield))
    
    e_avg = np.nanmean(efield,axis=1)
    e_std = np.nanstd(efield,axis=1)
    
    c_avg = np.nanmean(cfield,axis=1)
    c_std = np.nanstd(cfield,axis=1)
    
    coef_var['e'] = e_std/e_avg
    coef_var['c'] = c_std/c_avg
    
    coef_vars.append(coef_var)
    means.append(e_avg)
    
# %%
single_column_size = (1.5,1)
fig,ax=plt.subplots(1,1,figsize=single_column_size)
#fontsize
plt.rcParams.update({'font.size': 8})
tt = np.linspace(0,5,len(efield))
for e_avg in means:
    plt.plot(tt,e_avg,label=f'AR={ARs[i_]}',linewidth=0.5)
ax.set_xlabel(r'$t$ (sec)')
ax.set_ylabel(r'$\mu(e)$')
plt.savefig(f'{output_dir}/entanglement-over-time-mean.png',dpi=300,bbox_inches='tight')

    
    # plt.plot(tt,e_std/e_avg,label=f'AR={ARs[i_]}',linewidth=0.5)
# %%
dt = data_container_list[0].time_line[1] - data_container_list[0].time_line[0]

last_part_averaged = []
fig,ax=plt.subplots(1,1,figsize=single_column_size)
for i_,cv in enumerate(coef_vars):
    ax.plot(tt,cv['e'],label=rf'$\alpha={ARs[i_]}$',linewidth=0.5)
    last_part_averaged.append(np.mean(cv['e'][tt > 2]))
                              
ax.set_xlabel(r'$t$ (sec)')
ax.set_ylabel(r'$\sigma/\mu$')
plt.yticks(rotation=90)

# plt.legend(loc='lower left',fontsize=6)
plt.savefig(f'{output_dir}/coef-var-over-time-e.png',dpi=300,bbox_inches='tight')
# %%

    
fig,ax=plt.subplots(1,1,figsize=single_column_size)
plt.plot(ARs,last_part_averaged,'o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma/\mu$')

# %%
fig,axs=plt.subplots(1,2,figsize=(12,6))
for i_,cv in enumerate(coef_vars):
    axs[0].plot(tt,cv['e'],label=f'AR={ARs[i_]}',linewidth=0.5)
    axs[1].plot(tt,cv['c'],label=f'AR={ARs[i_]}',linewidth=0.5)    

axs[0].set_xlabel(r'$t$ (sec)')
axs[0].set_ylabel(r'$\sigma/\mu$')
axs[1].set_xlabel(r'$t$ (sec)')
axs[1].set_ylabel(r'$\sigma/\mu$')
axs[1].set_ylim([0,2])
plt.legend(loc='lower right',fontsize=6)


plt.savefig(f'{output_dir}/coef-var-over-time-e-and-c.png',dpi=300,bbox_inches='tight')

# %%


# %%
avg_entanglement = []
std_entanglement = []

avg_contact = []
std_contact = []

for dc in entanglement_data_container_list:
    efield = dc.dataobj['e_fields_over_time']
    e_avg = np.nanmean(efield)
    e_std = np.nanstd(efield)
    
    cfield = dc.dataobj['c_fields_over_time']
    c_avg = np.nanmean(cfield)
    c_std = np.nanstd(cfield)
    
    avg_entanglement.append(e_avg)
    std_entanglement.append(e_std)
    
    avg_contact.append(c_avg)
    std_contact.append(c_std)
    
avg_entanglement = np.array(avg_entanglement)
std_entanglement = np.array(std_entanglement)

avg_contact = np.array(avg_contact)
std_contact = np.array(std_contact)
# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
plt.errorbar(np.array([25,50,75,100,125,200,300]),avg_entanglement,yerr=std_entanglement,fmt='o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\mu$')
plt.savefig(f'{output_dir}/entanglement-over-alpha-with-error.png',dpi=300,bbox_inches='tight')

# %%
single_column_size = (1.3,1.1)
fig,ax=plt.subplots(1,1,figsize=single_column_size)
ax.plot(np.array([25,50,75,100,125,200,300]),std_entanglement/avg_entanglement,'o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma/\mu$')
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.savefig(f'{output_dir}/entanglement-sigma-over-mu.png',dpi=300,bbox_inches='tight')

cve = std_entanglement/avg_entanglement


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
    # phi_fields = phi_fields[:667]
    e_fields = dc.dataobj['e_fields_over_time']
    # e_fields = e_fields[:667]
    c_fields = dc.dataobj['c_fields_over_time']
    # c_fields = c_fields[:667]
    
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

fig,axs=plt.subplots(1,2,figsize=(12,6))
axs[0].plot(ARs,max_e,'o')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$\max(e)$')
axs[1].plot(ARs,max_c,'o')
axs[1].set_xlabel(r'$\alpha$')
axs[1].set_ylabel(r'$\max(c)$')
plt.savefig(f'{output_dir}/max-e-max-c.png',dpi=300,bbox_inches='tight')

# %%
fig,axs=plt.subplots(1,2,figsize=(12,6))
axs[0].plot(ARs,max_e/avg_entanglement,'o')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$\max(e)$')

axs[1].plot(ARs,max_c/avg_entanglement,'o')
axs[1].set_xlabel(r'$\alpha$')
axs[1].set_ylabel(r'$\max(c)$')
plt.savefig(f'{output_dir}/max-e-max-c-normalized.png',dpi=300,bbox_inches='tight')
# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
ax.plot(ARs,max_e/avg_entanglement,'o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\max(e)/\langle e \rangle$')
plt.savefig(f'{output_dir}/max-e-over-alpha-normalized.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
ax.plot(ARs,max_c/avg_contact,'o')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\max(c)/\langle c \rangle$')
plt.savefig(f'{output_dir}/max-c-over-alpha-normalized.png',dpi=300,bbox_inches='tight')
# %%


# %%
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
plt.savefig(f'{output_dir}/entanglement-power-law-distribution-big-panel.png',dpi=300,bbox_inches='tight')

# %%
single_column_size = (1.3,1.1)
fig,ax=plt.subplots(1,1,figsize=single_column_size)
# fontsize
plt.rcParams.update({'font.size': 8})

for i_,hist_result in enumerate(histogram_results):
    n,x = hist_result['e_field']
    
    xx = x[:-1]
    yy = n
    xx = xx[yy>0]
    yy = yy[yy>0]

    
    ax.loglog(xx,yy,'.-',markersize=1,label=fr'$\alpha={ARs[i_]}$',linewidth=0.5)
    # clr = ax.get_lines()[-1].get_color()
    
    # popt,pcov=curve_fit(power_law,xx,yy)
    # ax.loglog(xx,power_law(xx,*popt))
    # print(popt)
    
    # logx = np.log(xx)
    # logy = np.log(yy)
    # p = np.polyfit(logx,logy,1)    
    # lgd = r'$\alpha = %d$, $y = %.2f x^{%.2f}$' % (ARs[i_],np.exp(p[1]),p[0])    
    # ax.loglog(xx,np.exp(p[1])*xx**p[0],color=clr,label=lgd)

ax.set_xlabel(r'$e$')
ax.set_ylabel(r'$p(e)$')

# plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=90, ha='right')

# plt.legend(fontsize=6)
plt.savefig(f'{output_dir}/entanglement-power-law-distribution.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
# fontsize
plt.rcParams.update({'font.size': 8})

for i_,hist_result in enumerate(histogram_results):
    n,x = hist_result['c_field']
    
    xx = x[:-1]
    yy = n
    xx = xx[yy>0]
    yy = yy[yy>0]

    
    ax.loglog(xx,yy,'.-',markersize=1,label=fr'$\alpha={ARs[i_]}$',linewidth=0.5)
    # clr = ax.get_lines()[-1].get_color()
    
    # popt,pcov=curve_fit(power_law,xx,yy)
    # ax.loglog(xx,power_law(xx,*popt))
    # print(popt)
    
    # logx = np.log(xx)
    # logy = np.log(yy)
    # p = np.polyfit(logx,logy,1)    
    # lgd = r'$\alpha = %d$, $y = %.2f x^{%.2f}$' % (ARs[i_],np.exp(p[1]),p[0])    
    # ax.loglog(xx,np.exp(p[1])*xx**p[0],color=clr,label=lgd)

ax.set_xlabel(r'$c$')
ax.set_ylabel(r'$p(c)$')
plt.yticks(rotation=90, ha='right')
plt.legend(fontsize=4)
plt.savefig(f'{output_dir}/contact-power-law-distribution.png',dpi=300,bbox_inches='tight')

    
# %%
for i_,dc in enumerate(entanglement_data_container_list):
    N = Ns[i_]
    c_fields = dc.dataobj['c_fields_over_time']
    c_fields = c_fields
    c_field_last_frame = c_fields[-1]
    plt.hist(c_field_last_frame,bins=100,density=True,label=f'AR={ARs[i_]}',alpha=0.5)
    
# %%
frac_over_time = []
for dc,edc in zip(data_container_list,entanglement_data_container_list):
    N = dc.num_rods
    AR = dc.AR
    
    final_contact_list = dc.contact_list[-1]
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
# f for hang
frac_over_time_hang = []
for dc in data_container_list_hang:
    N = dc.num_rods
    AR = dc.AR
    
    final_contact_list = dc.contact_list[-1]
    final_contact_list = final_contact_list.reshape(-1,18)
    
    contact_ij = final_contact_list[:,4:6].astype(int)
    
    contact_graph = nx.Graph()
    contact_graph.add_nodes_from(np.arange(N))
    contact_graph.add_edges_from(contact_ij)
    
    clusters = list(nx.connected_components(contact_graph))
        
    # largest clusters
    largest_cluster = max(clusters,key=len)
    frac_over_time_hang.append(len(largest_cluster)/N)
    
# %%
ARs = []
frac_over_time_tickle = []
for dc in data_container_list_tickle:
    N = dc.num_rods
    AR = dc.AR
    ARs.append(AR)
    
    final_contact_list = dc.contact_list[-1]
    final_contact_list = final_contact_list.reshape(-1,18)
    
    contact_ij = final_contact_list[:,4:6].astype(int)
    
    contact_graph = nx.Graph()
    contact_graph.add_nodes_from(np.arange(N))
    contact_graph.add_edges_from(contact_ij)
    
    clusters = list(nx.connected_components(contact_graph))
        
    # largest clusters
    largest_cluster = max(clusters,key=len)
    frac_over_time_tickle.append(len(largest_cluster)/N)

# %%

fig,ax=plt.subplots(1,1,figsize=single_column_size)
plt.plot(ARs,frac_over_time_hang,'o')
plt.xlabel(fr'$\alpha$')
plt.ylabel(fr'$f$')


# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
cve = std_entanglement/avg_entanglement
plt.plot(cve,frac_over_time_hang,'o')
plt.xlabel(fr'$\sigma(e)/\mu(e)$')
plt.ylabel(fr'$f$')
# %%
fig,ax=plt.subplots(1,1,figsize=single_column_size)
plt.plot(ARs,frac_over_time_hang,'o',label=r'Static test')
plt.plot(ARs,frac_over_time_tickle,'o', label=r'Dynamic test ($a/g = 1$)')
ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f$')
plt.legend(fontsize=6)
plt.savefig(f'{output_dir}/f-over-alpha.png',dpi=300,bbox_inches='tight')

# %%
output_root = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'
pklpath = Path(f'{output_root}/Micromechanics-HangModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_hang = pickle.load(f)
    
# %%
len(output_dict_list_hang)
# %%
f_list_repeated = []
for output_dict_list in output_dict_list_hang:
    
    ARs = []
    f_list = []
    for output_dict in output_dict_list:
        ARs.append(output_dict['AR'])
        frac_over_time = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
        f = frac_over_time[-1]
        f_list.append(f)
    
    f_list_repeated.append(f_list)
    
# %%       
for f_lsit in f_list_repeated:
    plt.plot(ARs,f_list,'o-')

# %%
output_dict_list_hang[0][0].keys()

# %%

pklpath = Path(f'{output_root}/Micromechanics-TickleModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_tickle = pickle.load(f)
pklpath = Path(f'{output_root}/Micromechanics-TabModelos/output_dict_list_repeated.pkl')
with open(pklpath,'rb') as f:
    output_dict_list_tab = pickle.load(f)

# %%



# %%
for i_,dc in enumerate(entanglement_data_container_list):
    
    txt = dc.path.parent.parent.stem
    search_result = re.search(r'N(\d+)[-_]AR(\d+)',txt)
    N = int(search_result.group(1))
    AR = int(search_result.group(2))
    
    if AR == 50:
                
        phi_fields = dc.dataobj['phi_fields_over_time']
        # phi_fields = phi_fields[:667]
        e_fields = dc.dataobj['e_fields_over_time']
        # e_fields = e_fields[:667]
        c_fields = dc.dataobj['c_fields_over_time']
        # c_fields = c_fields[:667]
        
        phi_field_last_frame = phi_fields[-1]
        e_field_last_frame = e_fields[-1]
        c_field_last_frame = c_fields[-1]
        # dta = e_field_last_frame[~np.isnan(e_field_last_frame)]
        

        image_size = np.round((e_field_last_frame.shape[0])**(1/3)).astype(int)
        phi_volume = phi_field_last_frame.reshape(image_size,image_size,image_size)
        e_volume = e_field_last_frame.reshape(image_size,image_size,image_size)
        c_volume = c_field_last_frame.reshape(image_size,image_size,image_size)

        c_volume[np.isnan(e_volume)] = np.nan

        phi_image = np.mean(phi_volume,axis=0)
        phi_image = np.flipud(phi_image.T)
        e_image = np.mean(e_volume,axis=0)
        e_image = np.flipud(e_image.T)
        c_image = np.mean(c_volume,axis=0)
        c_image = np.flipud(c_image.T)


        double_column_size = (5,3.5)

        fig,axs=plt.subplots(1,3,figsize=double_column_size)
        axs = axs.flatten()
        # colorbar below
        fig.colorbar(axs[0].imshow(phi_image,cmap='coolwarm'),
                    ax=axs[0],orientation='horizontal')

        fig.colorbar(axs[1].imshow(e_image,cmap='coolwarm'),
                        ax=axs[1],orientation='horizontal')

        fig.colorbar(axs[2].imshow(c_image,cmap='coolwarm'),
                        ax=axs[2],orientation='horizontal')

        for ax in axs:
            ax.axis('off')

        plt.savefig(f'{output_dir}/phi-e-c-image_AR{AR}.png',dpi=300,bbox_inches='tight')


# %%




pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1423_RUN_PerturbCalmEEModelo1_N0750_AR150_g0.5_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1412_RUN_PerturbCalmEEModelo1_N0500_AR100_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g10_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1408_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g2_freq0.1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq3')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq1')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1/20240620-1343_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq0.1')

# class data_container:
#     def __init__(self,dataphat,max_rows=100000):
#         self.path = Path(dataphat)
#         out = parse_path_string(self.path)
#         # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
#         self.file_id = out[0]
#         self.surfix = out[1]
#         self.num_rods = out[2]
#         self.AR = out[3]
#         self.datetime_string = out[4]
#         self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)
# %%        
max_rows = 1000000
data_container_list = []

# (g, freq)
data_container_dict = {}

for pth in pathlist:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        
        # data_container_list.append(data_container(file,max_rows=max_rows))
        
        exp_id = pth.split('/')[-1]
        search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)
        
        N = int(search_result.group(1))
        AR = int(search_result.group(2))
        g = float(search_result.group(3))
        freq = float(search_result.group(5))
        
        data_container_list.append(data_container(file,max_rows=500))
        
        print(N,AR,g,freq)
        
        
# %%
data_entry = data_container_list[-1]
tt = data_entry.time_line
vv = data_entry.contact_list[-1].reshape(-1,18)

contact_ij = vv[:,4:6].astype(int)

# contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
curr_nodes = data_entry.node_list[-1]
graph = nx.Graph()
graph.add_nodes_from(range(len(curr_nodes)))
graph.add_edges_from(contact_ij)
clusters = list(nx.connected_components(graph))

# largest clusters
largest_cluster = max(clusters,key=len)
f = len(largest_cluster)/len(curr_nodes)


    
# %%



            