# -*- coding: utf-8 -*-
import numpy as np
from Value_of_SS.VSS import ValueSS,Test_Scenarios_VSS, Test_Sigma_VSS, Test_Lambda_VSS
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
from solver.solveCCP_ev import SolverExactCCP_ev
from stability.stability import out_stability, in_stability
from heuristic.Grasp import Modified_Kmean, Local_Search, Local_Search_Bis
from heuristic.Tests import Test1, Test2, Test3, Test4, Test_Points, Test_Mean, Test_Iter


np.random.seed(23)


#******************************************************************************
# VALUE OF THE STOCHASTIC SOLUTION
#**********************************

# VSS -------------------------------------------------------------------------
N_POINTS = 100
N_SCENARIOS = 10
p=10

# Creating the instance of CCP problem
inst = InstanceCCP(N_POINTS, N_SCENARIOS)
inst.plot()

# Result of vss
VSS, _, _ = ValueSS(inst, p)
print(f"VSS: {VSS}")



# TEST with different N_SCENARIOS ---------------------------------------------
N_POINTS = 50
p = 10

# Sample mean of vss for 20 different instance
VSS_mean_list = Test_Scenarios_VSS(N_POINTS, p)
print(f"VSS: {VSS_mean_list}")



# TEST with different values of SIGMA -----------------------------------------
N_POINTS = 50
N_SCENARIOS = 10
p = 10

# Vss changing the variance by modifing the parameter sigma 
VSS_sigma_list = Test_Sigma_VSS(N_POINTS, N_SCENARIOS, p)
print(f"VSS: {VSS_sigma_list}")



# TEST with different values of LAMBDA ----------------------------------------
N_POINTS = 50
N_SCENARIOS = 10
p = 10

# Vss changing the factor lambda
VSS_lambda_list = Test_Lambda_VSS(N_POINTS, N_SCENARIOS, p)
print(f"VSS: {VSS_lambda_list}")






#******************************************************************************
# IN-SAMPLE & OUT-OF-SAMPLE STABILITY
#************************************

#Large tree dim N'<S
N_POINTS = 50
p = 10
N_large = 1000
# Creation of an instance with 1000 scenarios taken as the real distribution
inst_large = InstanceCCP(N_POINTS, N_large)
# Building of the problem class 
sol_large =  SolverExactCCP_stoch(inst_large, p)

# Testing in/out of sample stabilities
out_stability(sol_large, p)
in_stability(sol_large, p)






#******************************************************************************
# HEURISTICS
#***********


#TEST 1 - LOCAL SEARCH vs. LOCAL SEARCH BIS vs. GUROBI
N_POINTS = 150
N_SCENARIOS = 5
p_list = [10, 15, 20, 25, 30]

# Testing with different number of clusters
Test1(N_POINTS, N_SCENARIOS, p_list)



#TEST 2 - GRASP vs. GUROBI 
N_POINTS = 150
N_SCENARIOS = 5
p_list = [10, 15, 20, 25, 30, 35]
Grasp_runs = 10

# Testing with different number of cluster and 10 GRASP iterations
Test2(N_POINTS, N_SCENARIOS, p_list, Grasp_runs)   


#TEST 3 - GRASP vs. GUROBI changing the parameter lamda
N_POINTS = 150
N_SCENARIOS = 5
p = 19       # To avoid the critical behaviour of Gurobi at 20-24 clusters
lambda_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
Grasp_runs = 10

# Testing with different values of lambda
Test3(N_POINTS, N_SCENARIOS, p, lambda_list, Grasp_runs) 


'''
#TEST 4 - GRASP vs. GUROBI changing the number of scenarios
N_POINTS = 150
scenario_list = [3, 6, 9, 12, 15, 18, 20]
p = 19       # To avoid the critical behaviour of Gurobi at 20-24 clusters
Grasp_runs = 10

# Testing with different number of scanrios
Test4(N_POINTS, scenario_list, p, Grasp_runs) 
'''



# PARALLELIZATION
#****************

#TEST AVERAGE - GRASP vs. GUROBI for different n. of clusters - AVERAGE + PARALLELIZATION
N_POINTS = 150
N_SCENARIOS = 5
p_list = [5, 10, 15, 20, 25, 30]
Grasp_runs = 10

# Testing with different number of clusters and repeating the test 5 times
Test_Mean(N_POINTS, N_SCENARIOS, p_list, Grasp_runs)    


#TEST ITER - GRASP vs. GUROBI for different number Grasp's runs - AVERAGE + PARALLELIZATION
N_POINTS = 150
N_SCENARIOS = 5
p = 19   # To avoid the critical behaviour of Gurobi around 20-24 clusters
Grasp_runs = [1, 20, 40, 60]

# Testing with different number of GRASP iterations and repeating the test 5 times
Test_Iter(N_POINTS, N_SCENARIOS, p, Grasp_runs)


'''
#TEST INCREASING NUMBER OF POINTS - GRASP vs. GRASP bis vs. Gurobi
Grasp_it = 30
n_scenarios = 5
n_points_list = [200, 250, 300]
n_clusters_list = [24, 30 , 36]

# Testing with different number of clusters and points
Test_Points(Grasp_it, n_points_list, n_clusters_list, n_scenarios)
'''
