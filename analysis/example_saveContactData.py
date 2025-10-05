
# %%
start = time.time()
contact_dataframe_list = []

for curr_time_index in range(1,len(time_line),1):
    curr_nodes = node_list[curr_time_index].reshape((num_rods,-1,3))
    curr_force = contact_list[curr_time_index].reshape(-1,18)

    contact_info_snapshot = []
    for query_index in range(0,len(curr_force),1):
        single_contact_info = curr_force[query_index]
        contact_info = process_contact_data(single_contact_info,curr_nodes)
        contact_info_snapshot.append(contact_info)
    
    data = []
    for single_contact_info in contact_info_snapshot:
        data.append(single_contact_info)
    
    df = pd.DataFrame(data)
    contact_dataframe_list.append(df)
    
print(f'Time taken: {time.time()-start:.2f} seconds')
# %%
curr_nodes = node_list[curr_time_index].reshape((num_rods,-1,3))
curr_force_all_info = contact_list[curr_time_index].reshape(-1,18)

num_total_contacts = len(curr_force_all_info)

curr_force_essentials = np.zeros((num_total_contacts,6))
for query_index in range(num_total_contacts):
    single_contact_info = curr_force_all_info[query_index]
    contact_info = process_contact_data(single_contact_info,curr_nodes)
    
    pi = contact_info['contact_point_i']
    pj = contact_info['contact_point_j']
    cij = (pi+pj)/2
    fij = contact_info['contact_force_i']    
    
    curr_force_essentials[query_index] = np.array([cij[0],cij[1],cij[2],fij[0],fij[1],fij[2]])
# %%
fF = filamentFields.filamentFields(curr_nodes,curr_force_essentials)
# %%
fF.analyzeLocalVolume(np.array([0,0,0]), 0.2, 0.2)
# %%
fF.return_number_of_local_contacts()
fF.return_force_sum()
# %%
I = np.linalg.norm(curr_force_essentials[:,:3] - np.array([0,0,0]),axis=1) < 0.2
np.count_nonzero(I)

np.sum(np.linalg.norm(curr_force_essentials[I,3:],axis=1))

# %%


# %%
def save_dataframe_list(contact_dataframe_list, output_folder):
    for idx, df in enumerate(contact_dataframe_list):
        df.to_feather(f'{output_folder}/dataframe_{idx}.feather')
    return None
        
def load_data_frame_list(data_folder):
    import glob
    glob_return = glob.glob(f'{data_folder}/*.feather')
    # number of files
    num_files = len(glob_return)    
    data = []
    for idx in range(num_files):
        df = pd.read_feather(f'{data_folder}/dataframe_{idx}.feather')
        data.append(df)
    return data

# %%
output_folder = 'test'
start = time.time()
save_dataframe_list(contact_dataframe_list, output_folder)
print(f'Time taken: {time.time()-start:.2f} seconds')
# %%
start = time.time()
contact_dataframe_list = load_data_frame_list(output_folder)
print(f'Time taken: {time.time()-start:.2f} seconds')