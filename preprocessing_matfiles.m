xray_data_root = '/Users/yeonsu/Documents/GitHub/entanglement/data';

dir_query = fullfile(xray_data_root,sprintf('**/centerlines.mat'));
dir_return = dir(dir_query);
%%
mkdir('xray_raw_data')

for pth = dir_return'
    load(fullfile(pth.folder,pth.name))

    [~,exp_id]=fileparts(pth.folder);

    dir_for_each_data = fullfile('xray_raw_data',exp_id);
    mkdir(dir_for_each_data)
    
    out_pth = fullfile(dir_for_each_data,pth.name);
    save(out_pth,'centerlines');
    
end
