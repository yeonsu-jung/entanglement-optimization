# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data_io import import_all_log, parse_path_string    
from analysis import get_curr_force_essentials
import re 
    
# %%

output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Jostle
protocol_id = 'EntangledCarrotCake5-fields'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N125-AR25-Scale1_20240531-222435')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N250-AR50-Scale1_20240531-222435')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N375-AR75-Scale1_20240531-222436')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N500-AR100-Scale1_20240531-222436')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N625-AR125-Scale1_20240531-222434')
num_samples = len(pathlist)
# %%
data_path_list = []
for pth in pathlist:    
    pth = Path(pth)
    # find npz file
    for files in pth.rglob('*.npz'):
        data_dict = {}
        if str(files.stem).endswith('all_fields_over_time'):            
            data_dict['file_id'] = files.parent.parent
            data_dict['file_path'] = files
            data_path_list.append(data_dict)
            break
    
    # /Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N125-AR25-Scale1_20240531-222435/data_20240602-232647/all_fields_over_time.npz


# %%
    
output_path = f'{output_root}/{protocol_id}'
if not os.path.exists(output_path):
    os.makedirs(output_path)
# %%
figure_handles = []
ax_handles = []
for _i in range(6):
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    figure_handles.append(fig)
    ax_handles.append(axs)

for dta in data_path_list:
    file_id = dta['file_id']
    file_path = dta['file_path']
    
    all_fields = np.load(file_path, allow_pickle=True)
    total_entanglement_over_time = all_fields['total_entanglement_over_time']
        
    field_list = []
    field_list.append(all_fields['n_fields_over_time'])
    field_list.append(all_fields['phi_fields_over_time'])
    field_list.append(all_fields['S_fields_over_time'])
    field_list.append(all_fields['e_fields_over_time'])
    field_list.append(all_fields['c_fields_over_time'])
    field_list.append(all_fields['f_fields_over_time'])
    # field_list.append(all_fields['Q_fields_over_time'])
    

    time_line = np.linspace(0,10,len(field_list[0]))
    mean_field = np.zeros(len(time_line))
    std_field = np.zeros(len(time_line))
    coeff_var_field = np.zeros(len(time_line))
    
    n_fields_over_time = all_fields['n_fields_over_time']
    for _ifield,field in enumerate(field_list):
        n_field = n_fields_over_time[_ifield]        
        for _i,_time in enumerate(time_line):            
            each_field = field[_i]
            mean_field[_i] = np.nanmean(each_field[n_field>0])
            std_field[_i] = np.nanstd(each_field[n_field>0])
            coeff_var_field[_i] = np.nanstd(each_field[n_field>0])/np.nanmean(each_field[n_field>0])

        local_ax = ax_handles[_ifield]
        local_ax[0].plot(time_line,mean_field)
        local_ax[1].plot(time_line,std_field)
        local_ax[2].plot(time_line,coeff_var_field)
        local_ax[0].set(xlabel='time (s)', ylabel='Mean')
        local_ax[1].set(xlabel='time (s)', ylabel='Std')
        local_ax[2].set(xlabel='time (s)', ylabel='Coeff. Var.')

titles = ['N','phi','S','e','c','f']
for _i in range(6):
    local_ax = ax_handles[_i]
    local_ax[1].set_title(titles[_i])
    
# ax_handles[0][2].legend([f'{_i}' for _i in range(5)])
# label
for _i in range(6):
    ax_handles[_i][2].legend([f'N{_i}' for _i in range(num_samples)])
    
plt.legend()
# %%
len(ax_handles)
    
# %%
axs[0,0].set_title('Total Entanglement')
axs[0,1].set_title('Mean')
axs[1,0].set_title('Std')
axs[1,1].set_title('Coeff. Var.')
# %%

for ax in axs.flat:
    ax.set(xlabel='time (s)', ylabel='a.u.')
    ax.label_outer()


# %%
phi_fields_over_time = all_fields['phi_fields_over_time']

mean_field = np.zeros(len(time_line))
for _i in range(len(time_line)):
    mean_field[_i] = np.nanmean(phi_fields_over_time[_i])
    
plt.plot(time_line,mean_field)



# for _time in [0,1,2,3,4,5,6,7,8,9,10]:
#     ax.axvline(x=_time, color='k', linestyle='--', alpha=0.5)

