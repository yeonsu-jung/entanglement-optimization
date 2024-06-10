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
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1000-AR200-Scale1_20240603-131308')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N1500-AR300-Scale1_20240603-000746')
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
num_field_types = 5
for _i in range(num_field_types):
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    figure_handles.append(fig)
    ax_handles.append(axs)

for dta in data_path_list:
    file_id = dta['file_id']
    file_path = dta['file_path']
    
    all_fields = np.load(file_path, allow_pickle=True)
    total_entanglement_over_time = all_fields['total_entanglement_over_time']
        
    field_list = []
    # field_list.append(all_fields['n_fields_over_time'])
    field_list.append(all_fields['phi_fields_over_time'])
    field_list.append(all_fields['S_fields_over_time'])
    field_list.append(all_fields['c_fields_over_time'])
    field_list.append(all_fields['e_fields_over_time'])
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
        
        if _ifield == 2 or _ifield == 4:
            local_ax[2].set_ylim([0,3])

titles = ['Volume fraction','Nematic order','Contact','Entanglement','Contact force']

for _i in range(num_field_types):
    local_ax = ax_handles[_i]
    local_ax[1].set_title(titles[_i],fontsize=16,fontweight='bold')
    
# ax_handles[0][2].legend([f'{_i}' for _i in range(5)])
# label
num_rods_items = [125,250,375,500,625,1000,1500]
AR_items = [25,50,75,100,125,200,300]
for _i in range(num_field_types):
    ax_handles[_i][2].legend([f'AR{AR_items[_i]}' for _i in range(num_samples)])
    
    
panel_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']

# Add panel labels
for _i in range(num_field_types):
    for _j in range(3):
        pannel_letter = panel_letters[_i*3+_j]
        ax_handles[_i][_j].annotate(pannel_letter, xy=(0, 1.05), xycoords='axes fraction',
                                   ha='center', va='center', fontsize=12, fontweight='bold')


plt.legend()
for _i in range(num_field_types):
    fig = figure_handles[_i]
    fig.savefig(f'{output_path}/fields_{titles[_i]}.png',dpi=300)