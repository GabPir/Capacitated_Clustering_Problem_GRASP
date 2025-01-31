# -*- coding: utf-8 -*-


import time
import gurobipy as grb
import numpy as np



# Creating the class of the two stage stochastic subproblems for the Local search
# NOTE: the following programming considers also the non-negative part of the 
# stochastic term in the obj function to return a better centroid candidate.  

class Sub_SolverExactCCP_stoch:
    
    # The version with excluded points creates an instance without them
    def __init__(self, inst, p, excluded = -np.ones(1)):
        self.inst = inst
        self.model = grb.Model('ccp')
        self.n_points = self.inst.n_points
        self.n_scenarios = self.inst.n_scenarios


        points = range(self.n_points)
        scenarios = range(self.n_scenarios)

        # Y = 1 if cluster is the center
        self.Y = self.model.addVars(
            self.n_points,
            vtype=grb.GRB.BINARY,
            name='Y'
        )
        
        self.X = self.model.addVars(
            self.n_points, self.n_points,
            vtype=grb.GRB.BINARY,
            name='X'
        )
        
        # We don't have the lower bound anymore
        self.Z = self.model.addVars(            
            self.n_points, self.n_scenarios,
            vtype=grb.GRB.CONTINUOUS,       
            #lb=0.0,                   
            name='Z'                        
        )

        # set objective function
        expr = sum(
            self.inst.d[i, j] * self.X[i, j] for i in points for j in points
        )
        expr += self.inst.l / self.n_scenarios * sum( self.Z[j, s] for j in points for s in scenarios)      

        self.model.setObjective(
            expr,
            grb.GRB.MINIMIZE         
        )

        # add constraints
        self.model.addConstrs(
            ( grb.quicksum(self.X[i,j] for j in points) == 1 for i in points),
            name="x_assigned"
        )
        self.model.addConstr(
            grb.quicksum(self.Y[i] for i in points) == p,
            name="p_upbound"
        )
        self.model.addConstrs(
            (self.X[i,j] <= self.Y[j] for i in points for j in points),
            name="linkXY"                          
        )
        self.model.addConstrs(    #We want to "pay" if the demand is not satisfied
            (self.Z[j, s] >= grb.quicksum( self.inst.w[i, s] * self.X[i,j] for i in points ) - self.inst.C[j]*self.Y[j] for j in points for s in scenarios), 

            name="linkZYX"       
        )


        if not(np.array_equal(excluded, -np.ones(1))):
            #We semplify excluding the worst candidates
            for i in range(len(excluded)):
                j = int(excluded[i])   
                self.n_points
                self.model.addConstr(self.Y[j] == 0)
                
        self.model.update()



    # Solving the sub-stochastic problem using Gurobi
    def solve(self, lp_name=None, gap=None, time_limit=None, verbose=False):
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        if lp_name:
            self.model.write(f"./logs/{lp_name}_sub_stoch.lp")
        self.model.setParam('LogFile', './logs/gurobi_sub_stoch.log')

        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start
        #print(f"computational time SubSolve Gurobi: {comp_time} s")

        
        sub_sol = []   
        if self.model.status == grb.GRB.Status.OPTIMAL:
            for j in range(self.n_points):
                if self.Y[j].X > 0.5:
                    sub_sol.append(j)

        all_vars = self.model.getVars()
        values = self.model.getAttr("X", all_vars)
        obj_value_stoch = self.model.getObjective().getValue()

        arr = np.array(values)
        Yminimizer = arr[0:self.n_points]
        low = self.n_points
        up = self.n_points*self.n_points + self.n_points
        Xminimizer = np.reshape(arr[low:up], (self.n_points,self.n_points))

        return Yminimizer, Xminimizer, obj_value_stoch, sub_sol, comp_time



