
# %%
local_edges = fF.analyze_local_volume(np.array([1000,1000,500]), R_omega, rod_diameter)
# %%
edge_dict = {}
for i,edge in enumerate(local_edges):
    label = int(edge[6])
    if label not in edge_dict:
        edge_dict[label] = []
    edge_dict[label].append(edge[:6])
    
# %%
for k,v in edge_dict.items():
    v = np.array(v)
    v = np.vstack(v)
    edge_dict[k] = v
# %%
node_dict = {}
for k,v in edge_dict.items():
    if len(v) == 1:
        v = v.reshape(-1,3)
    
    else:
        last_node = v[-1,3:]
        v = v[:,:3]
        v = np.vstack([v,last_node])
        
    node_dict[k] = v
    
# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for k,v in node_dict.items():
    ax.plot(v[:,0],v[:,1],v[:,2])
ax.axis('equal')
# %%
Q = np.zeros((3,3))
# Q += (3.0 * edge * edge.transpose() - Eigen::Matrix3d::Identity())/2.0;
for edge in local_edges:
    orientation = edge[3:6] - edge[:3]
    orientation /= np.linalg.norm(orientation)
    
    # kronecker product
    nn = orientation[:,np.newaxis]*orientation
    # orientational order
    Q += (3*nn - np.eye(3))/2
    # orientational order parameter
Q = Q/len(local_edges)
# %%
w,v=np.linalg.eig(Q)
# %%
w
v
# %%
w[np.argmax(np.abs(w))]

# %%
fF.return_orientational_order_parameter()

# %%

    
    
    

    
    
    
# %%

    