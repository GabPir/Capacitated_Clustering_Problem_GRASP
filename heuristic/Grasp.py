# -*- coding: utf-8 -*-


import numpy as np
import random
from solver.sub_solveCCP_stoch import Sub_SolverExactCCP_stoch
from solver.solveCCP_stoch import SolverExactCCP_stoch
import time
from joblib import Parallel, delayed
import multiprocessing




# GRASP ALGORITHM: 
# n_it: number of iterations of Local searches
# FLAG: 'bis' means using Local search bis 
# parallel: 'parallel' means  using the parallelized version of the heuristic
def GRASP(inst, k, n_it = 5, FLAG = '', parallel = ''):
    np.random.seed(23)
    
    # First run
    if FLAG == '':
        _, X, Y = Local_Search(inst, k)
        objf_best =feval(inst, X, Y)
    else:
        _, X, Y = Local_Search_Bis(inst, k)
        objf_best =feval(inst, X, Y)
        
    n_it -= 1  #We have already done one iteration
    
    # We search for better solutions, changing the seed so the Local search will start differently   
    for i in range(n_it):
        np.random.seed(23+i)
        
        if FLAG == '':
            if parallel == 'parallel':
                _, X, Y = Local_Search(inst, k, 'parallel')
            else:
                _, X, Y = Local_Search(inst, k)
            objf_new =feval(inst, X, Y)
            if objf_new < objf_best:   # Minimization => a lower value is better
                objf_best = objf_new
                
        else:
            if parallel == 'parallel':
                _, X, Y = Local_Search_Bis(inst, k, 'parallel')
            else:    
                _, X, Y = Local_Search_Bis(inst, k)
            objf_new =feval(inst, X, Y)
            if objf_new < objf_best:   # Minimization => a lower value is better
                objf_best = objf_new
    
    return X, Y, objf_best



     
    



def Local_Search(inst, k, parallel = ''):
    
    n_points = np.copy(inst.n_points)
    points_vector = -np.ones(n_points).astype(int)
    
    # Create the initial custer partition (randomly) 
    centroids_idx = np.array(random.sample(list(range(n_points)), k)).astype(int)
    
    # Calling the mod. kmean that perform a greedy clusterization
    points_vector  = Modified_Kmean(inst, k, centroids_idx)
    
    X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
    Y_tot = np.zeros(inst.n_points).astype(int)

    FLAG = True
    it = 0
    # STOPPING CRITIRIA: max number of iterations or FLAG=False when the centroids don't change  
    while FLAG and it < 15:
        X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
        Y_tot = np.zeros(inst.n_points).astype(int)
        new_centroids_idx = -np.ones(k).astype(int)
        
        if parallel == 'parallel':
            # Performing the parallelized version of the heuristic
            result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(Sub_problem)(points_vector, centroids_idx, inst, Y_tot, i) for i in range(k))
            for i in range(len(result)):
                # Updating the centroids (Parallel return a list of lists)
                new_centroids_idx[i] = result[i][1]  
                
                # Saving the current centroids as optimizers 
                Y_tot[new_centroids_idx[i]] = 1
        else:
            # Performing the "normal" version of the heuristic
            for i in range(k):
                _, new_centroids_idx[i] = Sub_problem(points_vector, centroids_idx, inst, Y_tot, i)
                
                Y_tot[new_centroids_idx[i]] = 1
        
        # We reallocate the points considering the new centroids     
        points_vector = Modified_Kmean(inst,k, new_centroids_idx)

        # Saving the assignment of points to the clusters
        for i in range(n_points):
            X_tot[i, points_vector[i]] = 1
        
        # If clusters do not change we stop the iterations
        if np.array_equal(centroids_idx, new_centroids_idx):
            FLAG = False
        
        centroids_idx = np.copy(new_centroids_idx)
        
        it += 1

            
    return it, X_tot, Y_tot
         



# Faster and less accurate version of Local Search
def Local_Search_Bis(inst, k, parallel = ''):
    
    n_points = np.copy(inst.n_points)    
    points_vector = -np.ones(n_points).astype(int)
    
    # Create the initial custer partition (randomly) 
    centroids_idx = np.array(random.sample(list(range(n_points)), k)).astype(int)

    # Calling the mod. kmean that perform a greedy clusterization
    points_vector  = Modified_Kmean(inst, k, centroids_idx)
    
    X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
    Y_tot = np.zeros(inst.n_points).astype(int)
    
    FLAG = True
    it = 0
    # STOPPING CRITIRIA: max number of iterations or FLAG=False when the centroids don't change  
    while FLAG and it < 15:
        X_tot = np.zeros((inst.n_points,inst.n_points)).astype(int)
        Y_tot = np.zeros(inst.n_points).astype(int)
        new_centroids_idx = -np.ones(k).astype(int)
        
        if parallel == 'parallel':
            # Performing the parallelized version of the heuristic
            result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(Sub_problem_bis)(points_vector, centroids_idx, inst, Y_tot, i) for i in range(k))
            for i in range(len(result)):
                # Updating the centroids (Parallel return a list of lists)
                new_centroids_idx[i] = result[i][1]
                # Saving the current centroids as optimizers 
                Y_tot[new_centroids_idx[i]] = 1
        else:
            # Performing the "normal" version of the heuristic
            for i in range(k):
                _, new_centroids_idx[i]  = Sub_problem(points_vector, centroids_idx, inst, Y_tot, i)
                Y_tot[new_centroids_idx[i]] = 1

        # We reallocate the points considering the new centroids   
        points_vector = Modified_Kmean(inst,k, new_centroids_idx)

        # Saving the assignment of points to the clusters
        for i in range(n_points):
            X_tot[i, points_vector[i]] = 1
        
        # If clusters do not change we stop the iterations
        if np.array_equal(centroids_idx, new_centroids_idx):
            FLAG = False
        
        centroids_idx = np.copy(new_centroids_idx)
        
        it += 1
        
            
    return it, X_tot, Y_tot
         







def Modified_Kmean(inst, k, centroids_idx):

    # Initializations
    n_points = inst.n_points
    centroids_cap = np.copy(inst.C)
    points_vector = -np.ones(n_points).astype(int) # Set all pointsas free (-1)
    points_dist = np.copy(inst.d)
    
    points_w_mean = np.zeros(n_points)
    
    # We take the mean of each row of the matrix of weights
    for i in range(n_points):
        points_w_mean[i] = np.mean(inst.w[i,:])
    
    # We have to remove the centroids capacity from themselves
    for i in range(k):
        j = int(centroids_idx[i])
        points_vector[j] = j
        centroids_cap[j] = centroids_cap[j] - points_w_mean[j] 
    
    # Allocation of points into centroids
    for i in range(k):
        c_idx = int(centroids_idx[i])
        j = 0 
        # Ordering the cluster with respect to the higher capacity
        candidates_idx = np.argsort(points_dist[c_idx,:]) 
        
        # We stop when the points are finished or when the remaining capacity is zero
        while centroids_cap[c_idx] > 0 and j < len(points_vector):
            cand_idx = candidates_idx[j]
            # If the point is free and requires less capacity than available, we associate it to the cluster 
            if points_vector[cand_idx] == -1 and points_w_mean[cand_idx] <= centroids_cap[c_idx]:
                centroids_cap[c_idx] = centroids_cap[c_idx] - points_w_mean[cand_idx]  
                points_vector[cand_idx] = c_idx
            j = j + 1
    
    # If some points remain unassigned, we allocate them to the nearest cluster (even if it's full)
    for ele in range(n_points):
        if points_vector[ele] == -1:
            elements = points_dist[np.ix_([ele], centroids_idx.astype(int))]
            centre = np.argmin(elements)  
            points_vector[ele] = int(centroids_idx[centre]) 

    return points_vector 





# Solving the sub-problems for each cluster
def Sub_problem(points_vector, centroids_idx, inst, Y_tot, i):
    
    sub_points = np.where(points_vector == centroids_idx[i])[0]
    
    # We create the sub-instance
    sub_inst = inst.sub_instance(sub_points)
    
    # We solve a sub-problem for the current cluster finding the SINGLE best centroid
    sub_solve = Sub_SolverExactCCP_stoch(sub_inst, 1)
    _, _, _, better_cen, _ = sub_solve.solve() 
    
    centroid = sub_points[better_cen]  # We have to return the original index 
    return i, centroid[0]




def Sub_problem_bis(points_vector, centroids_idx, inst, Y_tot, i):
    sub_points = np.where(points_vector == centroids_idx[i])[0]
    
    # We create the sub-instance
    sub_inst = inst.sub_instance(sub_points)
    
    # We solve a sub-problem for the current cluster finding the SINGLE best centroid
    sorted_idx = np.argsort(sub_inst.C)
    
    n = int(len(sub_points)*0.80)+1

    excluded_idx = sorted_idx[:-n]       # We consider only the most capacited 20% of points and Gurobi will choose the best one 
    sub_solve = Sub_SolverExactCCP_stoch(sub_inst, 1, excluded_idx) 
    _, _, _, better_cen, _ = sub_solve.solve()
    
    centroid = sub_points[better_cen]    # We have to return the original index 

    return i, centroid[0]




# Function for the evaluation of the results
def feval(inst, Xopt, Yopt):
    points = range(inst.n_points)
    scenarios = range(inst.n_scenarios)
            
    fun = sum(inst.d[i, j] * Xopt[i, j] for i in points for j in points) 
    fun += inst.l/inst.n_scenarios * sum(sum(max(0, (sum(inst.w[i, s] * Xopt[i,j] for i in points) - inst.C[j]*Yopt[j])) for j in points) for s in scenarios)
    return fun
    
    
    
    