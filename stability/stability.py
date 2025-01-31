import scipy.stats as st
import numpy as np
from instances.instanceCCP import InstanceCCP
from solver.solveCCP_stoch import SolverExactCCP_stoch
import matplotlib.pyplot as plt
import random


def out_stability(sol_test, p):

    N_large = sol_test.inst.n_scenarios
    
    # We initialize the parameters
    N = 30    # max number of scenarios that we will consider
    M = 5     # the number of times we will repeat the test
    step = 5  # the step of increasing the number of scenarios from 0 to 100
    k=0
    conf_int = []
    means = []
    x1 = np.zeros(0)
    
    
    n_rows = len(range(5, N+5, step))   # We consider different number of rows of the matrix of weights, i.e. number of scenarios (< 1000)
    results = np.zeros((n_rows, M))     # We will put results along the rows

    idx = list(range(N_large))   # Indexing the 1000 scearios => we will randomly select some among them
    
    for n in range(5, N+5, step):
        
        for m in range(M):
            
            # Simulate M scenario trees di dimensins n => the scenarios are taken randomly from the large tree
            inst = InstanceCCP(instance = sol_test.inst)
            idx_sample = np.sort(np.array(random.sample(idx, n)))

            # We modify the matrix of the weights considering the sampled scenarios
            inst.sampled(idx_sample)
            
            # We solve the problem with the sampled scenarios
            sol = SolverExactCCP_stoch(inst, p)
            Y, X, _, _, _ = sol.solve()
            
            # We evaluate the results in the original large tree
            results[k, m] = sol_test.evaluate_f(X, Y)

        print("---")
        
        # We consdier the t-Student confidence interval
        conf_int.append((st.t.interval(0.95, M-1, loc=np.mean(results[k]), scale=st.sem(results[k]))))
        means.append(np.mean(results[k]))
 
        k=k+1
        a = np.repeat(n,M)
        x1 = np.append(x1,a)
        
    print("Results out_stability: ", results)
    print("Means: ", means)
    print("Confidence interval out_stability: ", conf_int)
    
    #PLOT
    y1 = np.matrix.flatten(results)
    plt.scatter(x1, y1, s = 1.3, c = 'black', edgecolor = 'black')
    plt.title("Out of sample")
    plt.show()
        
    return




def in_stability(sol_test, p):
    N_large = sol_test.inst.n_scenarios

    # We initialize the parameters    
    N = 30    # max number of scenarios that we will consider
    M = 5     # the number of times we will repeat the test
    step = 5  # the step of increasing the number of scenarios from 0 to 100
    k=0
    conf_int = []
    means = []
    x1 = np.zeros(0)
    
    n_rows = len(range(5, N+5, step))    # We consider different number of rows of the matrix of weights, i.e. number of scenarios (< 1000)
    results = np.zeros((n_rows, M))      # We will put results along the rows

    idx = list(range(N_large))           # Indexing the 1000 scearios => we will randomly select some among them
    
    for n in range(5, N+5, step):
        for m in range(M):
            
            #Simulate M scenario trees di dimensins n => the scenarios are taken randomly from the large tree
            inst = InstanceCCP(instance = sol_test.inst)
            idx_sample = np.sort(np.array(random.sample(idx, n)))
            
            # We modify the matrix of the weights considering the sampled scenarios
            inst.sampled(idx_sample)
            
            # We solve the problem with the sampled scenarios
            sol = SolverExactCCP_stoch(inst, p)
            Y, X, _, _, _ = sol.solve()
            
            # We evaluate the results in the small tree
            results[k, m] = sol.evaluate_f(X, Y)

        print("---")
        
        #We use the t-Student confidence interval
        conf_int.append((st.t.interval(0.95, M-1, loc=np.mean(results[k]), scale=st.sem(results[k]))))
        means.append(np.mean(results[k]))

        k=k+1
        a = np.repeat(n,M)
        x1 = np.append(x1,a)
        
    print("Results in_stability: ", results)
    print("Means: ", means)
    print("Confidence interval in_stability: ", conf_int)
    
    #PLOT
    y1 = np.matrix.flatten(results)
    plt.scatter(x1, y1, s = 1.3, c = 'black', edgecolor = 'black')
    plt.title("In-sample stability")
    plt.xlabel('Number of scenarios')
    plt.ylabel('Values of the objective functions)')
    plt.show()
        
    return