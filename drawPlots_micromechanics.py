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
all_gathered_wrt_reps = []
for output_dict_list in output_dict_list_repeated:
    at_each_rep = []
    for output_dict in output_dict_list:
        at_each_rep.append(output_dict['fraction_of_nodes_in_largest_cluster_over_time'])
        # plt.loglog(output_dict['actual_timeline'],output_dict['fraction_of_nodes_in_largest_cluster_over_time'],'o-')
    all_gathered_wrt_reps.append(at_each_rep)
        
timeline = output_dict['actual_timeline']
        
# %%
ARs = []

output_dict_list = output_dict_list_repeated[0]
for output_dict in output_dict_list:
    ARs.append(output_dict['AR'])
ARs = np.array(ARs)
# %%
# ARs = [25,50,60,70,75,80,90,100,105,110,115,120,125,150,175,200,250,300]


# %%
dta = np.mean(all_gathered_wrt_reps,axis=0)
dta = dta.transpose()
err = np.std(all_gathered_wrt_reps,axis=0)
err = err.transpose()

# %%

# 20 markers
markers = ['o','s','^','v','<','>','1','2','3','4','8','p','P','*','h','H','+','x','X','D']
decay_constats = []
fig,ax=plt.subplots(1,1,figsize=(12,6))
for i_ in range(dta.shape[1]):
    ax.errorbar(timeline,dta[:,i_],err[:,i_],fmt=markers[i_],label=f'{ARs[i_]}')
    
    clr = ax.get_lines()[-1].get_color()
    popt,pcov = curve_fit(decay,timeline,dta[:,i_])
    decay_constats.append(popt[0])
    smoothx = np.linspace(0,5,100)
    smoothy = decay(smoothx,*popt)
    ax.plot(smoothx,smoothy,color=clr)
    
# ax.set_yscale('log')
# ax.set_xscale('log')

plt.xlabel('Time (sec)')
plt.ylabel('Fraction of nodes in the largest cluster')
plt.legend()
plt.savefig(f'{output_path}/collapse.png',dpi=300)
# %%
plt.semilogy(ARs,decay_constats,'o')
# %%
def backward_fitting_powerlaw(xx,yy,num_points):
    # fit decay to the last 20 points
    xx_cut = xx[-num_points:]
    yy_cut = yy[-num_points:]
    popt,pcov = curve_fit( power_law,xx_cut,yy_cut)
    return -popt[1]

# %%
output_dict_list = output_dict_list_repeated[0]


decay_constats_repeated = []
clrs = np.random.rand(len(output_dict_list),3)
start_point = 0
fig=plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
last_values_repeated = []
legend_count = 0
for output_dict_list in output_dict_list_repeated:
    at_each_rep = []    
    decay_constants = []
    last_values = []
    
    for i_,output_dict in enumerate(output_dict_list):
        xx = output_dict['actual_timeline']
        # put 0 in the first entry
        xx = np.hstack([0,xx])

        yy = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
        yy = np.hstack([1,yy])
        
        start_point = 50
        xx_cut = xx[start_point:]
        yy_cut = yy[start_point:]
        
        # popt,pcov = curve_fit( decay,xx_cut,yy_cut)
        # decay_const = popt[0]
        decay_const = backward_fitting_powerlaw(xx,yy,50)
        decay_constants.append(decay_const)
        
        if legend_count < 18:
            plt.plot(xx_cut, yy_cut, '-', color=clrs[i_], label=f'{ARs[i_]}')
            legend_count += 1
        else:
            plt.plot(xx_cut, yy_cut, '-', color=clrs[i_])
            
        last_values.append(yy_cut[-1])
        
    decay_constats_repeated.append(decay_constants)
    last_values_repeated.append(last_values)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel('Time (sec)')
plt.ylabel('Fraction of nodes in the largest cluster')
plt.savefig(f'{output_path}/collapse_raw.png',dpi=300)

decay_constants_avg = np.mean(decay_constats_repeated,axis=0)
decay_constants_err = np.std(decay_constats_repeated,axis=0)

fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.errorbar(ARs,decay_constants_avg,decay_constants_err,fmt='o')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('AR')
ax.set_ylabel('Collapse exponent')
# tight layout
plt.tight_layout()
plt.savefig(f'{output_path}/collapse_constant.png',dpi=300)

# %%
five_sec_collapse_avg = np.mean(last_values_repeated,axis=0)
five_sec_collapse_err = np.std(last_values_repeated,axis=0)

fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.errorbar(ARs,five_sec_collapse_avg,five_sec_collapse_err,fmt='o')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('AR')
ax.set_ylabel('Cluster size at t=5s')
# tight layout
plt.tight_layout()
plt.savefig(f'{output_path}/five_seconds_survival.png',dpi=300)
# %%
five_sec_collapse_avg

# %%
ind_AR100 = np.where(ARs == 100)[0][0] + 1

plt.errorbar(ARs[:ind_AR100],five_sec_collapse_avg[:ind_AR100],five_sec_collapse_err[:ind_AR100],fmt='o')
# popt,pcov = curve_fit(power_law,ARs[:ind_AR100],five_sec_collapse_avg[:ind_AR100])
# popt,pcov = curve_fit(,ARs[:ind_AR100],five_sec_collapse_avg[:ind_AR100])
plt.loglog(ARs[:ind_AR100],power_law(ARs[:ind_AR100],*popt))

# %%
from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

# plot sigmoid function
x = np.linspace(-10,10,100)
y = sigmoid(x, 1, 0, 1, 0)
plt.plot(x,y)

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
        original_sign = np.sign(lk_mat0_flatten)
        
        lk_mat0_sum = np.sum(np.abs(lk_mat0_flatten))

        hamming_dist_list = []
        untanglement_list = []
        for lkm in lk_mat_over_time[1:]:
            lkm_flatten = lkm[np.triu_indices(lkm.shape[0],k=1)]
            current_sign = np.sign(lkm_flatten)
            
            naive_diff = lkm_flatten - lk_mat0_flatten
            # untanglement = np.abs(lkm_flatten)/np.abs(lk_mat0_flatten)
            # if original sign was positive, positive naive_diff means increase in entanglement
            # if original sign was negative, positive naive_diff means decrease in entanglement
            actual_diff = naive_diff * original_sign
            hamming_dist = np.sum(current_sign != original_sign)
            hamming_dist_list.append(hamming_dist/len(lkm_flatten))            
            untanglement_list.append(np.sum(np.abs(actual_diff))/lk_mat0_sum)
            
        # ax.plot(timeline[:-1],hamming_dist_list,label=f'{ARs[k]}')
        ax.plot(timeline[:-1],untanglement_list,label=f'{ARs[k]}')
        last_values.append(hamming_dist_list[-1])
        untanglement_list_wrt_AR.append(untanglement_list)
        
        k = k + 1
    last_values_repeated.append(last_values)
    untanglement_repeated.append(untanglement_list_wrt_AR)
    
ax.set_xlim([0,5])
# legend outside
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (sec)')
plt.ylabel('Untanglement')

# %%
for last_values in last_values_repeated:
    plt.loglog(ARs,last_values,'o')
plt.xlabel('AR')
plt.ylabel('Fraction of lk# sign changes at t=5s')
plt.savefig(f'{output_path}/sign_changes.png',dpi=300)

# %%
tme = timeline
dta = np.mean(untanglement_repeated,axis=0).transpose()
# put 0 for the first time point
dta = np.vstack([np.zeros(dta.shape[1]),dta])
err = np.std(untanglement_repeated,axis=0).transpose()
err = np.vstack([np.zeros(err.shape[1]),err])
# cutoff

cutoff_point = 5
tme = tme[cutoff_point:]
tme = tme - tme[0]
dta = dta[cutoff_point:]
dta = dta - dta[0,:]
err = err[cutoff_point:]

# %%
saturation_constant = []
power_law_constant = []
fig,ax=plt.subplots(1,1,figsize=(12,8))
for i in range(dta.shape[1]):
    ax.errorbar(tme,dta[:,i],err[:,i],fmt='o',label=f'{ARs[i]}')
    # get color
    color = ax.get_lines()[-1].get_color()

    popt,pcov = curve_fit(power_law,tme,dta[:,i])
    power_law_constant.append(popt[1])
    
    popt,pcov = curve_fit(saturation,tme,dta[:,i],bounds=([0,0],[1,1]))
    saturation_constant.append(popt[1])
    
    xx = np.linspace(0,5,100)
    yy = saturation(xx,*popt)
    
    ax.plot(xx,yy,color=color)
    
    
# legend outside
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (sec)')
plt.ylabel('Untanglement')
plt.savefig(f'{output_path}/untanglement.png',dpi=300)

# %%
plt.loglog(ARs,saturation_constant,'o')


# %%
err = np.std(last_values_repeated,axis=0)
fig,ax=plt.subplots(1,1,figsize=(6,3))
ax.errorbar(ARs,last_values,np.array(err),fmt='o')
# log scale
# ax.set_yscale('log')
# ax.set_xscale('log')


# %%
ARs_float = np.array(ARs).astype(float)
popt,pcov = curve_fit(power_law,ARs_float,last_values)

xx = np.log(ARs_float/ARs_float[0])
yy = np.log(last_values/last_values[0])

popt2,pcov2 = curve_fit(linear, xx, yy)

plt.loglog(ARs_float,last_values,'o')
plt.xlabel('AR')
plt.ylabel('Fraction of lk# sign changes at t=5s')