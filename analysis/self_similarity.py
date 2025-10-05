# %%
import numpy as np
from analysis import orientational_statistics
from transforms import q_to_x
from potentials import pairwise_contact_point
from jax import vmap,jit
import matplotlib.pyplot as plt
from potentials import all_pairwise_angles,all_pairwise_distances,all_pairwise_skewness,total_effective_potential
from transforms import x_to_q
from potentials import create_pairs
import re
from analysis_functions import get_N500_data,parse_pathname,compute_nematic_order,create_folder
from potentials import compute_linking_number_vectorized_with_l
def get_outer_end(q):
    packing_center = np.mean(x.reshape(-1,3),axis=0)

    v1 = np.linalg.norm(x[:,:3]-packing_center,axis=1)
    v2 = np.linalg.norm(x[:,3:6]-packing_center,axis=1)

    plt.hist(v1,bins=100)
    plt.hist(v2,bins=100)

    idx = []
    inner_end = []
    outer_end = []
    for i in range(len(v1)):
        if v1[i] > v2[i]:
            idx.append(0)
            inner_end.append(x[i,3:6])
            outer_end.append(x[i,:3])
        else:
            idx.append(1)
            inner_end.append(x[i,:3])
            outer_end.append(x[i,3:6])


    inner_end = np.array(inner_end)
    outer_end = np.array(outer_end)

    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    ax.scatter(inner_end[:,0],inner_end[:,1],inner_end[:,2])
    ax.scatter(outer_end[:,0],outer_end[:,1],outer_end[:,2])

# %%
pathlist = get_N500_data()
_,_,num_rods,random_keys = parse_pathname(pathlist[0])
common_folder = f"/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/{random_keys}"
create_folder(common_folder)

# %%
pth = pathlist[-1]
q = np.loadtxt(pth)
q_reshaped = q.reshape(-1,5).copy()
q_pairs = create_pairs(q_reshaped)

# %%
from analysis_functions import cut_


# %%
pth = pathlist[-4]
_,_,num_rods,random_keys = parse_pathname(pth)
q_300 = np.loadtxt(pth)
q_reshaped_300 = q_300.reshape(-1,5).copy()
q_pairs_300 = create_pairs(q_reshaped_300)

lk_300 = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs_300)
print(np.sum(lk_300))

# %%
def plot_x(x,ax):
    for i in range(len(x)):
        ax.plot(x[i,[0,3]],x[i,[1,4]],x[i,[2,5]])

# %%

q_500 = np.loadtxt(pathlist[-1])
q_pairs_500 = create_pairs(q_500.reshape(-1,5))
lk_500 = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs_500)

from analysis_functions import parse_pathname
from visualizations import plot_many_rods


# %%
lk_ratio = []
AR_list = []
fig,ax=plt.subplots(1,2,subplot_kw={'projection':'3d'})
# plot_many_rods(q_500.reshape(-1,5),ax=ax[0])

for pth in pathlist:
    dt_string, AR, num_rods, random_keys = parse_pathname(pth)
    cut_off_factor = AR/500
    q_i = np.loadtxt(pth)
    q_pairs_i = create_pairs(q_i.reshape(-1,5))
    lk_i = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs_i)

    plot_many_rods(q_i.reshape(-1,5),ax=ax[0])

    x_new, _ = cut_(q_500.reshape(-1,5),cut_off=cut_off_factor/2)
    q_new = x_to_q(x_new)
    plot_x(x_new,ax[1])

    new_q_pairs = create_pairs(q_new.reshape(-1,5))
    new_lk = vmap(lambda q: compute_linking_number_vectorized_with_l(q,cut_off_factor))(new_q_pairs)    
    
    AR_list.append(AR)
    # plot_many_rods(q_new.reshape(-1,5),ax,opt_dict={'color':'red'})
    lk_ratio.append(np.sum(new_lk)/np.sum(lk_i))
    plt.savefig(f'figures/self_similarity_{AR}.png')
    
lk_ratio = np.array(lk_ratio)

# %%
fig,ax=plt.subplots(figsize=(2.5,2))
plt.plot(AR_list,lk_ratio,'o-')
plt.xlabel('Aspect ratio')
plt.ylabel(r'$e_\mathrm{cut}/e_\mathrm{max}$')
plt.savefig('figures/self_similarity_e_ratio.png')


# %%


# %%


# %%

# %%
from visualizations import plot_many_rods
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
plot_many_rods(q.reshape(-1,5),ax)
# %%


# %%
def make_it_bigger(x):
    return x*1.5

# %%
lk = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs)



# %%
plt.hist(lk,density=True,bins=100)
plt.hist(new_lk,density=True,bins=100,alpha=0.5)


# %%
pth = pathlist[-1]
dt_string, AR, num_rods, random_keys = parse_pathname(pth)
print(AR)

q_500 = np.loadtxt(pth)
x_500 = q_to_x(q_500)
q_pairs_500 = create_pairs(q_500.reshape(-1,5))

# %%
pth = pathlist[4]
dt_string, AR, num_rods, random_keys = parse_pathname(pth)
print(AR)

q_100 = np.loadtxt(pth)
x_100 = q_to_x(q_100)
q_pairs_100 = create_pairs(q_100.reshape(-1,5))

# %%
lk = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs_500)
lk_100 = vmap(lambda q: compute_linking_number_vectorized_with_l(q,1))(q_pairs_100)

# %%


# %%
np.sum(new_lk*lk_100)/np.sqrt(np.sum(new_lk**2)*np.sum(lk_100**2))

# so why?


# %%

# %%
np.sum(lk*lk_100)/np.sqrt(np.sum(lk**2)*np.sum(lk_100**2))


# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for x in x_new:
    ax.plot(x[[0,3]],x[[1,4]],x[[2,5]])

# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
plot_many_rods(q_100.reshape(-1,5),ax)

# %%
np.sum(lk_100)
np.sum(new_lk)
# %%
np.sum(lk)

# %%
x_new
ll = []
for i in range(10):
    x1,x2 = x_new[i][:3], x_new[i][3:]
    ll.append(np.linalg.norm(x1-x2))

# %%
from potentials import pairwise_distance
dd = vmap(pairwise_distance)(new_q_pairs)

# %%
np.min(dd)

# %%
0.2/0.002


# %%
