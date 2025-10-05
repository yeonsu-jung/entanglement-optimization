# %%
import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.optimize import curve_fit

def power_law(x,a,b):
    return a*x**b

def linear(x,a,b):
    return a*x + b

def saturation(x,a,b):
    return a*(1-np.exp(-b*x))

def decay(x,b):
    return np.exp(-b*x)


root_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/'
# analysis_id = 'Micromechanics-HangModelos'
analysis_id = 'Micromechanics-Modelos'
output_path = f'{root_dir}/{analysis_id}'
read_file_path = f'{output_path}/output_dict_list_repeated.pkl'
with open(read_file_path,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
# %%
output_dict_list = output_dict_list_repeated[0]
len(output_dict_list)
# %%


fig,ax=plt.subplots(1,1,figsize=(6,3))
last_values_repeated = []
untanglement_repeated = []
for output_dict_list in output_dict_list_repeated:
    k = 0
    last_values = []
    untanglement_list_wrt_AR = []
    for output_dict in output_dict_list:
        timeline = output_dict['actual_timeline']
        lk_mat_over_time = output_dict['lk_mat_over_time']    
        lk_mat0 = lk_mat_over_time[0]
        lk_mat0_flatten = lk_mat0[np.triu_indices(lk_mat0.shape[0],k=1)]
        
        break
    
    break
        
        
# %%

output_dict_list = output_dict_list_repeated[0]
output_dict = output_dict_list[17]
lk_mat_over_time = output_dict['lk_mat_over_time']

# %%
len(lk_mat_over_time)
lkm0 = lk_mat_over_time[55][np.triu_indices(lk_mat_over_time[55].shape[0],k=1)]
lkm1 = lk_mat_over_time[56][np.triu_indices(lk_mat_over_time[56].shape[0],k=1)]

# %%
sign_changes_fraction = []
entangled_sign_changes_fraction = []
for i in range(len(lk_mat_over_time)-1):
    lkm0 = lk_mat_over_time[i][np.triu_indices(lk_mat_over_time[i].shape[0],k=1)]
    lkm1 = lk_mat_over_time[i+1][np.triu_indices(lk_mat_over_time[i+1].shape[0],k=1)]
    
    lkm0_entangled = lkm0[lkm0>0.1]
    lkm1_entangled = lkm1[lkm0>0.1]
    
    sign_changes_fraction.append(np.sum(np.sign(lkm0) == np.sign(lkm1))/len(lkm0))
    entangled_sign_changes_fraction.append(np.sum(np.sign(lkm0_entangled) == np.sign(lkm1_entangled))/len(lkm0_entangled))
    
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

figure_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Figures'

fig,ax=plt.subplots(1,1,figsize=(4,3))
for i,output_dict in enumerate(output_dict_list):
    
    
    plt.plot(output_dict['actual_timeline'][:-1],sign_changes_fraction)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Fraction of sign changes')
    plt.savefig(f'{figure_path}/sign_changes_alpha300.png',dpi=300,bbox_inches='tight')

# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))
plt.plot(output_dict['actual_timeline'][:-1],entangled_sign_changes_fraction)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Fraction of sign changes')
ax.set_ylim([0,1.2])
plt.savefig(f'{figure_path}/entangled_sign_changes_alpha300.png',dpi=300,bbox_inches='tight')


# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))

threshold_list = np.linspace(0.01,0.1,10)
for threshold in threshold_list:
    sign_changes_fraction = []
    entangled_sign_changes_fraction = []
    for i in range(len(lk_mat_over_time)-1):
        lkm0 = lk_mat_over_time[i][np.triu_indices(lk_mat_over_time[i].shape[0],k=1)]
        lkm1 = lk_mat_over_time[i+1][np.triu_indices(lk_mat_over_time[i+1].shape[0],k=1)]
        
        lkm0_entangled = lkm0[lkm0>threshold]
        lkm1_entangled = lkm1[lkm0>threshold]
        
        sign_changes_fraction.append(np.sum(np.sign(lkm0) == np.sign(lkm1))/len(lkm0))
        entangled_sign_changes_fraction.append(np.sum(np.sign(lkm0_entangled) == np.sign(lkm1_entangled))/len(lkm0_entangled))
        
    plt.plot(output_dict['actual_timeline'][:-1],entangled_sign_changes_fraction)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Fraction of sign changes')
    ax.set_ylim([0,1.2])
plt.savefig(f'{figure_path}/entangled_sign_changes_threshold_test.png',dpi=300,bbox_inches='tight')