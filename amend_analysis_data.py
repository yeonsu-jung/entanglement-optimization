# %%
import pickle
import numpy as np

root_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/'
analysis_id = 'Micromechanics-Modelos'

output_path = f'{root_dir}/{analysis_id}'
original_file_path = f'{output_path}/output_dict_list_repeated.pkl'
with open(original_file_path,'rb') as f:
    output_dict_list_repeated = pickle.load(f)
# %%
output_dict_list_repeated[0][0].keys()
# %%
output_dict_list_repeated[0][0]['actual_timeline']
    
# %%
amend_files = []

amend_files.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300/NonIntersectingBox-N1500-AR300-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240609-105235.csv')
amend_files.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo2_FineExcitation/20240609-1056_RUN_PerturbEEModelo2_N1500_AR300/NonIntersectingBox-N1500-AR300-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240609-105615.csv')
amend_files.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo3_FineExcitation/20240609-1056_RUN_PerturbEEModelo3_N1500_AR300/NonIntersectingBox-N1500-AR300-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240609-105616.csv')
# %%
from data_io import import_all_log
time_line, node_list, contact_list = import_all_log(amend_files[0],max_rows=100000)
# %%
time_line[1:15:5]

# %%
from analysis import analyze_csv_file
skip_frames = 5
output_dict_list_for_amendment = []
for data_path in amend_files:
    output_dict = analyze_csv_file(data_path,skip_frames)
    output_dict_list_for_amendment.append(output_dict)
# %%
output_dict = analyze_csv_file('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300/NonIntersectingBox-N1500-AR300-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240609-105235.csv',skip_frames)
# %%
output_dict_list_for_amendment[0]=output_dict
# %%    
with open(f'{output_path}/output_dict_list_for_amendment.pkl','wb') as f:
    pickle.dump(output_dict_list_for_amendment,f)
# %%
from matplotlib import pyplot as plt
for output_dict in output_dict_list_for_amendment:
    tt = output_dict['actual_timeline']
    ff = output_dict['fraction_of_nodes_in_largest_cluster_over_time']
    plt.plot(tt,ff)
# %%
# amend!
for i_,output_dict_list in enumerate(output_dict_list_repeated):
    output_dict_list[-1] = output_dict_list_for_amendment[i_]
    
# %%
for i_,output_dict_list in enumerate(output_dict_list_repeated):
    tmp = output_dict_list[0]
    tt = tmp['actual_timeline']
# %%
# update original
with open(original_file_path,'wb') as f:
    pickle.dump(output_dict_list_repeated,f)

    
# %%


# pklpath = Path(f'{output_root}/Micromechanics-HangModelos/output_dict_list_repeated.pkl')
# pklpath = Path(f'{output_root}/Micromechanics-TickleModelos/output_dict_list_repeated.pkl')
# pklpath = Path(f'{output_root}/Micromechanics-TabModelos/output_dict_list_repeated.pkl')

