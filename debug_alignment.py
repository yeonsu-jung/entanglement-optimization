from matplotlib import pyplot as plt
import filamentprocessing
from fitting import prep_svd_cylinder, fit_rod
from pathlib import Path
import pickle
from example_prune import inspect_segments
import numpy as np
import networkx as nx
from clustering import find_connected_components, explode_local_cluster
import os
from distances import lumelsky_dist_vec

class Filament:
    def __init__(self,nodes):
        self.nodes = nodes # nodes are centerline points
        
        self.certification = None
        
        
        # self.svd_cylinders = None # cylinder representation of the filament
        # self.scores = None
        # self.ij = None
        # self.graph = None
        # self.connected_components = None
        # self.cluster_size_list = None
        # self.good_clusters = None
        # self.subclusters = None
        # self.subcluster_error_list = None
        # self.subcluster_length_list = None
        
        # self.local_segments = None
        
class CollectiveFilaments:
    def __init__(self,filaments):
        self.filaments = filaments # type Filament
        
        # graph representation
        
        # partitions (clusters)
        
        # partition, assess, and merge/reject
        
        
        

def calculate_2d_align_matrix(segments):
        num_segments = len(segments)
        align_matrix = np.zeros((num_segments,num_segments))
        
        for i in range(num_segments):
            for j in range(i+1,num_segments):
                seg_i = segments[i]
                seg_j = segments[j]

                joined = np.vstack([seg_i,seg_j])
                _,_,v = np.linalg.svd(joined)
                v1 = v[0,:]
                v2 = v[1,:]

                # project to the plane
                proj_i = seg_i @ np.column_stack([v1, v2])
                proj_j = seg_j @ np.column_stack([v1, v2])
                fitted_i = fit_line(proj_i)
                fitted_j = fit_line(proj_j)

                p1 = np.array([fitted_i[0,0],fitted_i[0,1],0])
                p2 = np.array([fitted_i[-1,0],fitted_i[-1,1],0])

                q1 = np.array([fitted_j[0,0],fitted_j[0,1],0])
                q2 = np.array([fitted_j[-1,0],fitted_j[-1,1],0])

                t,u,_,_,_ = lumelsky_dist_vec(p1,p2,q1,q2)

                popt1 = p1 + t * (p2 - p1)
                popt2 = q1 + u * (q2 - q1)
                dvec = popt1 - popt2

                t_opp = np.clip(1-t,0,1)
                u_opp = np.clip(1-u,0,1)

                popt1_opp = p1 + t_opp * (p2 - p1)
                popt2_opp = q1 + u_opp * (q2 - q1)

                axis1 = popt1_opp - popt1
                axis2 = popt2_opp - popt2

                dist = np.linalg.norm(dvec)
                
                axis1 = axis1 / np.linalg.norm(axis1)
                axis2 = axis2 / np.linalg.norm(axis2)
                dvec = dvec / np.linalg.norm(dvec)
                align_score = (np.linalg.norm(np.cross(dvec,axis1)) + np.linalg.norm(np.cross(dvec,axis2)) )
                align_matrix[i,j] = align_score
            
            print(f"Segment {i} done")
                

                
        return align_matrix

    align_matrix = calculate_2d_align_matrix(segments)
    
def fit_line(seg):
    cen = np.mean(seg,axis=0)
    centered = seg-cen
    u,s,v = np.linalg.svd(centered)
    orientation = v[0]
    orientation *= np.sign(np.sum(orientation * (seg[-1, :] - seg[0, :])))
    orientation = orientation / np.linalg.norm(orientation)
    slist = np.dot((seg - cen), orientation)
    
    # s1, s2 = np.min(slist), np.max(slist)
    best_estimation = cen + np.outer(slist, orientation)
    return best_estimation


def check_align_score(i,j,segments,svd_cylinders):
    seg_i = segments[i]
    seg_j = segments[j]

    p1 = svd_cylinders[i,0:3]
    p2 = svd_cylinders[i,3:6]
    q1 = svd_cylinders[j,0:3]
    q2 = svd_cylinders[j,3:6]

    t,u,d1,d2,d12 = lumelsky_dist_vec(p1,p2,q1,q2)

    popt1 = p1 + t * (p2 - p1)
    popt2 = q1 + u * (q2 - q1)
    dvec = popt1 - popt2
    dist = np.linalg.norm(dvec)

    t_opp = np.clip(1-t,0,1)
    u_opp = np.clip(1-u,0,1)

    popt1_opp = p1 + t_opp * (p2 - p1)
    popt2_opp = q1 + u_opp * (q2 - q1)

    axis1 = popt1_opp - popt1
    axis2 = popt2_opp - popt2

    plt.close('all')
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(seg_i[:,0],seg_i[:,1],seg_i[:,2],'-',linewidth=1)
    ax.plot(seg_j[:,0],seg_j[:,1],seg_j[:,2],'-',linewidth=1)
    ax.quiver(popt2[0],popt2[1],popt2[2],dvec[0],dvec[1],dvec[2],color='r',linewidth=0.5)
    ax.quiver(popt1[0],popt1[1],popt1[2],axis1[0],axis1[1],axis1[2],color='g',linewidth=0.5)
    ax.quiver(popt2[0],popt2[1],popt2[2],axis2[0],axis2[1],axis2[2],color='g',linewidth=0.5)
    

    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    normalized_dvec = dvec / np.linalg.norm(dvec)
    align_score = (np.linalg.norm(np.cross(normalized_dvec,axis1)) + np.linalg.norm(np.cross(normalized_dvec,axis2)) )/2
    print(f'Min. distance: {dist}')
    print(f'Align score: {align_score}')
    return ax

def align_score_2d(i,j,segments):
    seg_i = segments[i]
    seg_j = segments[j]

    joined = np.vstack([seg_i,seg_j])
    u,s,v = np.linalg.svd(joined)
    v1 = v[0,:]
    v2 = v[1,:]

    # project to the plane
    proj_i = seg_i @ np.column_stack([v1, v2])
    proj_j = seg_j @ np.column_stack([v1, v2])

    fitted_i = fit_line(proj_i)
    fitted_j = fit_line(proj_j)

    p1 = np.array([fitted_i[0,0],fitted_i[0,1],0])
    p2 = np.array([fitted_i[-1,0],fitted_i[-1,1],0])

    q1 = np.array([fitted_j[0,0],fitted_j[0,1],0])
    q2 = np.array([fitted_j[-1,0],fitted_j[-1,1],0])

    t,u,_,_,_ = lumelsky_dist_vec(p1,p2,q1,q2)

    popt1 = p1 + t * (p2 - p1)
    popt2 = q1 + u * (q2 - q1)
    dvec = popt1 - popt2

    t_opp = np.clip(1-t,0,1)
    u_opp = np.clip(1-u,0,1)

    popt1_opp = p1 + t_opp * (p2 - p1)
    popt2_opp = q1 + u_opp * (q2 - q1)

    axis1 = popt1_opp - popt1
    axis2 = popt2_opp - popt2


    dist = np.linalg.norm(dvec)

    plt.close()
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.plot(proj_i[:,0],proj_i[:,1],'.-')
    ax.plot(proj_j[:,0],proj_j[:,1],'-')
    ax.plot(popt1[0],popt1[1],'b.')
    ax.plot(popt1_opp[0],popt1_opp[1],'bo')

    ax.plot(popt2[0],popt2[1],'r.')
    ax.plot(popt2_opp[0],popt2_opp[1],'ro')
    ax.quiver(popt2[0],popt2[1],dvec[0],dvec[1],color='r')

    ax.quiver(popt1[0],popt1[1],axis1[0],axis1[1],color='g')
    ax.quiver(popt2[0],popt2[1],axis2[0],axis2[1],color='g')

    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    dvec = dvec / np.linalg.norm(dvec)
    align_score = (np.linalg.norm(np.cross(dvec,axis1)) + np.linalg.norm(np.cross(dvec,axis2)) )
    
    print(f'Min. distance: {dist}')
    print(f'Align score: {align_score}')
    return ax



def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

    
def inspect_clustering(good_cl,segments):
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        joined = np.vstack([segments[i] for i in gcl])
        joined = sort_curve(joined)
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
        
    joined_segment_length_list = np.zeros(len(good_cl))
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        joined = np.vstack([segments[i] for i in gcl])
        joined = sort_curve(joined)
        joined_segment_length_list[i_gcl] = seg_len(joined)
        
    fig,ax=plt.subplots(1,1)
    ax.hist(joined_segment_length_list,bins=100)
    return joined_segment_length_list



def check_overlap(good_cl,already_cl):
    no_overlap = []    
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        if len(gcl) < 1:
            continue
        if np.isin(gcl,already_cl).any():
            tmp_nodes = gcl[np.isin(gcl,already_cl)]
            
            print(f'Cluster {i_gcl} has already been clustered')
            print(f'Nodes: {tmp_nodes}')
            
            print(f'Corresponding cluster: ', gcl)
            continue
        else:
            no_overlap.append(gcl)
            
    return no_overlap


def inspect_a_cluster(a_cluster,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50):
    good_clusters = []
    
    subclusters = explode_local_cluster(a_cluster,svd_cylinders,segments)
    # check quality
    subcluster_error_list = np.zeros(len(subclusters))
    subcluster_length_list = np.zeros(len(subclusters))
    for i_subcluster,subcluster in enumerate(subclusters):
        joined = np.vstack([segments[ a_cluster[iii] ] for iii in subcluster])
        fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
        subcluster_error_list[i_subcluster] = fit_result['err']
        # rec = fit_result['rec']
        subcluster_length_list[i_subcluster] = np.sum(np.sqrt(np.sum(np.diff(fit_result['rec'],axis=0)**2,axis=1)))
    
    certification = (subcluster_error_list < subcluster_error_threshold) & (np.abs(subcluster_length_list - subcluster_length_threshold) < subcluster_length_tolerance)
        
    for i_cert in np.where(certification)[0]:
        good_clusters.append([a_cluster[i] for i in subclusters[i_cert]])
        
    return good_clusters,subclusters







def main():
    linearity_threshold = 0.5
    radius_curvature_threshold = 500
    already_clustered = []
        
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
    segments = pickle.load(open(segments_file_path,'rb'))

    local_segments = []
    for i,segment in enumerate(segments):
        centroid = np.mean(segment,axis=0)
        
        if (np.linalg.norm(centroid - np.array([1000,1000,300])) < 250):
            local_segments.append(segment)        
    print(f'Number of segments: {len(local_segments)}')
            
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in range(len(local_segments)):
        ax.plot(local_segments[i][:,0],local_segments[i][:,1],local_segments[i][:,2],linewidth=0.5)    
    segments = local_segments
    print

    import filamentprocessing
    fp = filamentprocessing.FilamentProcessing(segments,200,1,0.99)



    ij = fp.get_svd_ij()
    scores = fp.get_svd_scores()
    ij = np.array(ij)
    scores = np.array(scores)



    svd_cylinders = fp.get_svd_cylinders()
    svd_cylinders = np.array(svd_cylinders)
    svd_cylinders.shape
    len(segments)

    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in range(100):
        p1 = svd_cylinders[i,0:3]
        p2 = svd_cylinders[i,3:6]
        r = svd_cylinders[i,6]
        
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],linewidth=0.5)

    from visualizations import plot_centerline_with_container

    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in np.random.choice(len(segments),10):
        plot_centerline_with_container(segments,svd_cylinders,i,ax)
        
    dist_score = scores[:,0]
    align_score = scores[:,1]

    mask = align_score < 0.02
    mask = (dist_score < 3) & (align_score < 0.3)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(segments)))
    graph.add_edges_from(ij[mask,:])

    connected_components = list(nx.connected_components(graph))
    connected_components = [list(x) for x in connected_components]
    cluster_size_list = [len(x) for x in connected_components]
    print(f'Number of segments: {len(segments)}')
    print(f'Number of connected components: {len(connected_components)}')
    print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

    plt.close('all')
    fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
    for cc in connected_components:
        joined = np.vstack([segments[i] for i in cc])
        joined = sort_curve(joined)
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)


    plt.close('all')
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in range(len(connected_components)):
        if len(connected_components[i]) < 2:
            joined = np.vstack([segments[i] for i in connected_components[i]])
            seg_len_sum = np.sum([seg_len(segments[i]) for i in connected_components[i]])
            if (seg_len_sum < 100) and (seg_len_sum > 30):
                ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
                ax.text(joined[0,0],joined[0,1],joined[0,2],f'{i}',fontsize=4)
            
            
    # neighbors


    neighbors = []
    for i in range(len(segments)):
        neighbor_list = list(graph.neighbors(i))    
        if len(neighbor_list) > 2:
            print(i, neighbor_list)
        
        
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})

    friends = [2841,806, 1589, 1734, 2009, 2164, 2375]
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in friends:
        ax.plot(segments[i][:,0],segments[i][:,1],segments[i][:,2],linewidth=0.5)


    # [2841,806, 1589, 1734, 2009, 2164, 2375]
    i = 287
    j = 27
    ii = connected_components[i][0]
    jj = connected_components[j][0]
    ax = check_align_score(ii,jj,segments,svd_cylinders)
    # ax.axis('equal')

    seg_len(segments[i])

    # planar alignment

    i = 1786
    j = 1787
    ax = align_score_2d(i,j,segments)


    num_segments = len(segments)

    
    
if __name__ == '__main__':
    main()