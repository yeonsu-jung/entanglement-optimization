# %%
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
def power_law(x,a,b):
    return a*x**b
# %%
filepath = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/EntangleCarrotCake5/total_contact_over_time_data.pkl'
with open(filepath,'rb') as f:
    data = pickle.load(f)
    
# %%

# %%
fig,ax=plt.subplots(1,1,figsize=(3,2.5))
# font size
plt.rcParams.update({'font.size': 8})
for dta in data:
    # dta.keys()
    # search_result = re.search(r'N(\d+)-AR(\d+)',dta['file_id'])
    search_result = re.search(r'N(\d+)-AR(\d+)',dta['file_id'])
    N = int(search_result.group(1))
    AR = int(search_result.group(2))
    tme = dta['time_line']    
    yy = np.array(dta['total_number_of_contacts'])/N*2
    ax.plot(tme, yy,label=f'N={N}, AR={AR}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('No. of contacts per rod')
plt.tight_layout()
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/EntangleCarrotCake5/total_contact_over_time.png',dpi=300)

# %%

dta25 = np.load('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N125-AR25-Scale1_20240531-222435/data_20240602-232647/all_fields_over_time.npz')
dta300 = np.load('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1500-AR300-Scale1_20240603-000746/data_20240604-031230/all_fields_over_time.npz')

# %%
name_str = ['phi_fields_over_time',
            'S_fields_over_time',
            'e_fields_over_time',
            'c_fields_over_time']
# %%
num_time_points = dta25['phi_fields_over_time'].shape[0]
num_grids25 = int(np.round((dta25['phi_fields_over_time'].shape[1])**(1/3)))
num_grids300 = int(np.round((dta300['phi_fields_over_time'].shape[1])**(1/3)))
# %%
phi_image25 = np.max(dta25['phi_fields_over_time'][-1].reshape((num_grids25,num_grids25,num_grids25)),axis=0)
phi_image25 = np.flipud(phi_image25.T)

e_image25 = np.max(dta25['e_fields_over_time'][-1].reshape((num_grids25,num_grids25,num_grids25)),axis=0)
e_image25 = np.flipud(e_image25.T)

c_image25 = np.max(dta25['c_fields_over_time'][-1].reshape((num_grids25,num_grids25,num_grids25)),axis=0)
c_image25 = np.flipud(c_image25.T)

phi_image300 = np.max(dta300['phi_fields_over_time'][-1].reshape((num_grids300,num_grids300,num_grids300)),axis=0)
phi_image300 = np.flipud(phi_image300.T)

e_image300 = np.max(dta300['e_fields_over_time'][-1].reshape((num_grids300,num_grids300,num_grids300)),axis=0)
e_image300 = np.flipud(e_image300.T)

c_image300 = np.max(dta300['c_fields_over_time'][-1].reshape((num_grids300,num_grids300,num_grids300)),axis=0)
c_image300 = np.flipud(c_image300.T)



# %%
fig,axs=plt.subplots(2,3,figsize=(4,3))

fig.colorbar(axs[0,0].imshow(phi_image25,cmap='coolwarm'), ax=axs[0,0],location='bottom')
fig.colorbar(axs[0,1].imshow(e_image25,cmap='coolwarm'), ax=axs[0,1],location='bottom')
fig.colorbar(axs[0,2].imshow(c_image25,cmap='coolwarm'), ax=axs[0,2],location='bottom')
fig.colorbar(axs[1,0].imshow(phi_image300,cmap='coolwarm'), ax=axs[1,0],location='bottom')
fig.colorbar(axs[1,1].imshow(e_image300,cmap='coolwarm'), ax=axs[1,1],location='bottom')
fig.colorbar(axs[1,2].imshow(c_image300,cmap='coolwarm'), ax=axs[1,2],location='bottom')

for ax in axs.flatten():
    ax.axis('off')
    ax.axis('equal')
fig.savefig('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/EntangleCarrotCake5/last_frame_fields_25_300.png',dpi=300)
# fig.colorbar(axs[0].imshow(n_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[0])
            

# %%
filepath = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-HangModelos/output_dict_list_repeated.pkl'
with open(filepath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)

# %%
output_dict_list = output_dict_list_repeated[0]
ARs = []
for output_dict in output_dict_list:
    ARs.append(output_dict['AR'])
# %%
rep_vals = []
fig,ax=plt.subplots(1,1,figsize=(4,3))
for output_dict in output_dict_list:
    tt = output_dict['actual_timeline']
    fraction = output_dict['fraction_of_nodes_in_largest_cluster_over_time']    
    ax.loglog(tt,fraction)
    rep_vals.append(fraction[-1])
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-HangModelos/fraction_of_nodes_in_largest_cluster_over_time.png',dpi=300)

# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.loglog(ARs,1-np.array(rep_vals),'o-')
# detached fraction

    
# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.loglog(ARs,1-np.array(rep_vals),'o-')

# %%
output_dict_list = output_dict_list_repeated[0]
output_dict = output_dict_list[0]

def exponential(x,a,b):
    return a*np.exp(b*x)
    
smoothx = np.linspace(0,5,100)
xx = output_dict['actual_timeline']

# font size
plt.rcParams.update({'font.size': 8})

untanglement_list_wrt_AR = []

for output_dict in output_dict_list:
    lkmat_over_time = output_dict['lk_mat_over_time']
    
    lkmat0 = lkmat_over_time[0]
    original_sign = np.sign(lkmat0)
    untanglement_list = []    
    for i_ in range(len(lkmat_over_time)):
        lkmat = lkmat_over_time[i_]
        current_sign = np.sign(lkmat)
        
        naive_diff = lkmat - lkmat0
        actual_diff = naive_diff * original_sign
        untanglement = np.sum(actual_diff)/sum(np.abs(lkmat0))
        
        # np.count_nonzero(actual_diff>0)
        # np.count_nonzero(actual_diff<0)
        
        # untanglement = np.sum(actual_diff)
        # untanglement = np.sum(original_sign != current_sign)/len(lkmat)
        
        # untanglement_list.append( np.count_nonzero(np.abs(lkmat) > 0.25)/len(lkmat) )
        untanglement_list.append(untanglement)
        
    untanglement_list_wrt_AR.append(untanglement_list)
# %%
ARs = []
for output_dict in output_dict_list:
    ARs.append(output_dict['AR'])
        
# %%


fig,ax=plt.subplots(1,1,figsize=(4,3))
last_values = []
decay_constants = []
for output_dict,untanglement_list in zip(output_dict_list,untanglement_list_wrt_AR):    
    xx = output_dict['actual_timeline']
    AR = output_dict['AR']
    ax.plot(xx,untanglement_list,label=f'AR={AR}')
    
    # popt,pcov = curve_fit(exponential,xx,untanglement_list)    
    # clr = ax.get_lines()[-1].get_color()    
    # ax.plot(smoothx,exponential(smoothx,*popt),color=clr)
    
    # last_values.append(untanglement_list[-1])
    # decay_constants.append(popt[1])
    

# legend outside
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (sec)')
plt.ylabel('Untangled fraction')
# %%
decay_constants = np.array(decay_constants)
plt.plot(ARs,-decay_constants,'o-')

popt,pcov = curve_fit(power_law,ARs,-decay_constants)
smoothx = np.linspace(ARs[0],ARs[-1],100)
plt.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')

# %%
legend_count = 0
fig,ax=plt.subplots(1,1,figsize=(4,3))
avg_vv_repeated = []
for output_dict_list in output_dict_list_repeated:
    
    avg_vv = []
    for output_dict in output_dict_list:
        tt = output_dict['actual_timeline']
        vv = output_dict['centroid_velocities_over_time']
        
        vv = vv[tt > 1]
        tt = tt[tt > 1]
        
        avg_vv.append(np.mean(vv))
        
        AR = output_dict['AR']
        
        if legend_count < 18:
            ax.plot(tt,vv,label=f'AR={AR}')
            legend_count += 1
        else:
            ax.plot(tt,vv)
    avg_vv_repeated.append(avg_vv)
            
box=ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
# %%
avg_dta = np.mean(avg_vv_repeated,axis=0)
err_dta = np.std(avg_vv_repeated,axis=0)

fig,ax=plt.subplots(1,1,figsize=(6,5))
ax.errorbar(ARs,avg_dta,err_dta,fmt='o-')

popt,pcov = curve_fit(power_law,ARs,avg_dta)
smoothx = np.linspace(ARs[0],ARs[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
# %%
fig,ax=plt.subplots(1,1,figsize=(3,2))

ARs_under = ARs[ARs <= 110]
avg_dta_under = avg_dta[ARs <= 110]
err_dta_under = err_dta[ARs <= 110]

ax.errorbar(ARs_under,avg_dta_under,err_dta_under,fmt='o-')
clr = ax.get_lines()[-1].get_color()

popt,pcov = curve_fit(power_law,ARs_under,avg_dta_under)
smoothx = np.linspace(ARs_under[0],ARs_under[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_under[-1],avg_dta_under[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

ARs_over = ARs[ARs > 110]
avg_dta_over = avg_dta[ARs > 110]
err_dta_over = err_dta[ARs > 110]

ax.errorbar(ARs_over,avg_dta_over,err_dta_over,fmt='o-')
clr = ax.get_lines()[-1].get_color()

popt,pcov = curve_fit(power_law,ARs_over,avg_dta_over)
smoothx = np.linspace(ARs_over[0],ARs_over[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_over[-1],avg_dta_over[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('AR')
ax.set_ylabel('Average velocity (mm/s)')
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/EntangleCarrotCake5/avg_velocity.png',dpi=300)
# %%
ARs = np.array(ARs)
fig,ax=plt.subplots(1,1,figsize=(4,3))
for avg_vv in avg_vv_repeated:
    ax.plot(ARs,avg_vv,'o-')
    clr = ax.get_lines()[-1].get_color()
    
    popt,pcov = curve_fit(power_law,ARs,avg_vv)
    smoothx = np.linspace(ARs[0],ARs[-1],100)
    ax.plot(smoothx,power_law(smoothx,*popt),'--',color=clr,label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
    
ax.set_xscale('log')
ax.set_yscale('log')
    
    
# %%
legend_count = 0
fig,ax=plt.subplots(1,1,figsize=(4,3))
variables_repeated = []
for output_dict_list in output_dict_list_repeated:
    
    variables = []
    for output_dict in output_dict_list:
        tt = output_dict['actual_timeline']
        vv = output_dict['avg_contact_displacement_over_time']
        
        vv = vv[tt > 1]
        tt = tt[tt > 1]
        
        variables.append(np.mean(vv))
        
        AR = output_dict['AR']
        
        if legend_count < 18:
            ax.plot(tt,vv,label=f'AR={AR}')
            legend_count += 1
        else:
            ax.plot(tt,vv)
    variables_repeated.append(variables)
            
box=ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()

# %%
filepath = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-SlowExcitationModelo1/output_dict_list_repeated.pkl'
with open(filepath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
# %%


rep_vals_rep = []

fig,ax=plt.subplots(1,1,figsize=(4,3))
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
        
# %%
for rep_vals in rep_vals_rep:
    plt.loglog(ARs,np.array(rep_vals),'o-')
    
# %%
avg_dta = np.mean(rep_vals_rep,axis=0)
err_dta = np.std(rep_vals_rep,axis=0)

fig,ax=plt.subplots(1,1,figsize=(6,5))
ax.errorbar(ARs,avg_dta,err_dta,fmt='o-')

popt,pcov = curve_fit(power_law,ARs,avg_dta)
smoothx = np.linspace(ARs[0],ARs[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')




# %%
ARs = ARs[1:]
avg_dta = avg_dta[1:]
err_dta = err_dta[1:]

cutoff = 100
ARs = np.array(ARs)
ARs_under = ARs[ARs < cutoff]
avg_dta_under = avg_dta[ARs < cutoff]
err_dta_under = err_dta[ARs < cutoff]

fig,ax=plt.subplots(1,1,figsize=(3,2))
ax.errorbar(ARs_under,avg_dta_under,err_dta_under,fmt='o-')
clr = ax.get_lines()[-1].get_color()
popt,pcov = curve_fit(power_law,ARs_under,avg_dta_under)
smoothx = np.linspace(ARs_under[0],ARs_under[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_under[-1],avg_dta_under[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

ARs_over = ARs[ARs > cutoff]
avg_dta_over = avg_dta[ARs > cutoff]
err_dta_over = err_dta[ARs > cutoff]
popt,pcov = curve_fit(power_law,ARs_over,avg_dta_over)
smoothx = np.linspace(ARs_over[0],ARs_over[-1],100)
ax.errorbar(ARs_over,avg_dta_over,err_dta_over,fmt='o-')
clr = ax.get_lines()[-1].get_color()
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_over[-1],avg_dta_over[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

# vertical bar at AR = 100
ax.axvline(100,linestyle='--',color='k')

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel('AR')
plt.ylabel('Frac. largest cluster')
plt.tight_layout()
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-SlowExcitationModelo1/fraction_of_nodes_in_largest_cluster_over_time_log.png',dpi=300)
# %%

xx = np.abs(ARs_under[1:] - 100)
yy = 1-avg_dta_under[1:]

fig,ax=plt.subplots(1,1,figsize=(3,2))
ax.plot(xx,yy,'o-')
ax.set_xlabel('|AR - 100|')
ax.set_ylabel('1 - Frac. largest cluster')
# ax.axvline(0,linestyle='--',color='k')
# ax.axis([0,100,0,1])
ax.set_xscale('log')
ax.set_yscale('log')

popt,pcov = curve_fit(power_law,xx,yy)
smoothx = np.linspace(0,100,100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
ax.text(xx[-1],yy[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}')
plt.tight_layout()
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-SlowExcitationModelo1/fraction_of_nodes_in_largest_cluster_over_time_near_transition.png',dpi=300)




# %%
output_dict_list = output_dict_list_repeated[0]
# %%

representative_values = []
ARs = []
for output_dict in output_dict_list:
    ARs.append(output_dict['AR'])
    
    tt = output_dict['actual_timeline']
    vv = output_dict['centroid_velocities_over_time']
    
    vv = vv[tt > 1]
    tt = tt[tt > 1]
    
    plt.plot(tt,vv)
    
    representative_value = np.mean(vv)
    representative_values.append(representative_value)
    
# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.loglog(ARs,representative_values,'o-')

popt,pcov = curve_fit(power_law,ARs,representative_values)
smoothx = np.linspace(ARs[0],ARs[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
ax.text(ARs[-1],representative_values[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}')

plt.xlabel('AR')
plt.ylabel('Average velocity (mm/s)')
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-SlowExcitationModelo1/avg_velocity.png',dpi=300)

# %%
rep_vals = []
fig,ax=plt.subplots(1,1,figsize=(4,3))
for output_dict in output_dict_list:
    tt = output_dict['actual_timeline']
    fraction = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
    
    ax.plot(tt,fraction)
    rep_vals.append(fraction[-1])
    
# %%
fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.loglog(ARs,np.array(rep_vals),'o-')

# %%
rep_vals = []
fig,ax=plt.subplots(1,1,figsize=(4,3))
for output_dict in output_dict_list:
    tt = output_dict['actual_timeline']
    fraction = output_dict['fraction_of_nodes_in_largest_cluster_over_time']    
    ax.loglog(tt,fraction)
    rep_vals.append(fraction[-1])
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-SlowExcitationModelo1/fraction_of_nodes_in_largest_cluster_over_time.png',dpi=300)

# %%
# %%
filepath = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-Modelos/output_dict_list_repeated.pkl'
with open(filepath,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
    
# %%

rep_vals_rep = []

fig,ax=plt.subplots(1,1,figsize=(4,3))
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
        
# %%
fig=plt.figure()
ax=fig.add_subplot(111)
for rep_vals in rep_vals_rep:
    ax.loglog(ARs,np.array(rep_vals),'o-')
# vertical bar
ax.axvline(100,linestyle='--',color='k')

    
# %%
avg_dta = np.mean(rep_vals_rep,axis=0)
err_dta = np.std(rep_vals_rep,axis=0)

fig,ax=plt.subplots(1,1,figsize=(6,5))
ax.errorbar(ARs,avg_dta,err_dta,fmt='o-')

popt,pcov = curve_fit(power_law,ARs,avg_dta)
smoothx = np.linspace(ARs[0],ARs[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')

# %%
cutoff = 100
ARs = np.array(ARs)
ARs_under = ARs[ARs < cutoff]
avg_dta_under = avg_dta[ARs < cutoff]
err_dta_under = err_dta[ARs < cutoff]

fig,ax=plt.subplots(1,1,figsize=(4,3))
# font size
plt.rcParams.update({'font.size': 8})
ax.errorbar(ARs_under,avg_dta_under,err_dta_under,fmt='.')
clr = ax.get_lines()[-1].get_color()
popt,pcov = curve_fit(power_law,ARs_under,avg_dta_under)
smoothx = np.linspace(ARs_under[0],ARs_under[-1],100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_under[-1],avg_dta_under[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

ARs_over = ARs[ARs > cutoff]
avg_dta_over = avg_dta[ARs > cutoff]
err_dta_over = err_dta[ARs > cutoff]
popt,pcov = curve_fit(power_law,ARs_over,avg_dta_over)
smoothx = np.linspace(ARs_over[0],ARs_over[-1],100)
ax.errorbar(ARs_over,avg_dta_over,err_dta_over,fmt='.')
clr = ax.get_lines()[-1].get_color()
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)
ax.text(ARs_over[-1],avg_dta_over[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}',color=clr)

# vertical bar at AR = 100
ax.axvline(100,linestyle='--',color='k')

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel('AR')
plt.ylabel('Frac. largest cluster')
plt.tight_layout()
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-Modelos/fraction_of_nodes_in_largest_cluster_over_time_log.png',dpi=300)
# %%

xx = np.abs(ARs_under - 100)
yy = 1-avg_dta_under

fig,ax=plt.subplots(1,1,figsize=(3,2))
# fontsize
plt.rcParams.update({'font.size': 8})
ax.plot(xx,yy,'o-')
ax.set_xlabel('|AR - 100|')
ax.set_ylabel('1 - Frac. largest cluster')
# ax.axvline(0,linestyle='--',color='k')
# ax.axis([0,100,0,1])
ax.set_xscale('log')
ax.set_yscale('log')

popt,pcov = curve_fit(power_law,xx,yy)
smoothx = np.linspace(0,100,100)
ax.plot(smoothx,power_law(smoothx,*popt),'--',label=f'{popt[0]:.2f}x^{popt[1]:.2f}')
ax.text(xx[-1],yy[-1],f'{popt[0]:.2f}x^{popt[1]:.2f}')
plt.tight_layout()
plt.savefig(f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/Micromechanics-Modelos/fraction_of_nodes_in_largest_cluster_over_time_near_transition.png',dpi=300)

