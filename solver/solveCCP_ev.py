# -*- coding: utf-8 -*-
import time
import gurobipy as grb
import numpy as np

# Creating the class of the EV programming
class SolverExactCCP_ev:
    def __init__(self, inst, p):
        self.inst = inst
        self.model = grb.Model('ccp')
        self.n_points = self.inst.n_points
        self.n_scenarios = self.inst.n_scenarios
        
        points = range(self.n_points)
         
        self.w_mean = np.mean(self.inst.w, axis = 1)

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
        self.Z = self.model.addVars(       
            self.n_points,
            vtype=grb.GRB.CONTINUOUS,       
            lb=0.0,                         
            name='Z'                        
        )

        # set objective function
        expr = sum(
            self.inst.d[i, j] * self.X[i, j] for i in points for j in points
        )
        expr += self.inst.l * sum(self.Z[j] for j in points)         

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
            grb.quicksum(self.Y[i] for i in points) <= p,
            name="p_upbound"
        )
        self.model.addConstrs(
            (self.X[i,j] <= self.Y[j] for i in points for j in points),
            name="linkXY"                         
        )
        self.model.addConstrs(    #We want to "pay" if the demand is not satisfied
            (self.Z[j] >= grb.quicksum( self.w_mean[i] * self.X[i,j] for i in points ) - self.inst.C[j]*self.Y[j] for j in points),
            name="linkZYX"       
        )
        
        
        
        
        self.model.update()



    # Solving the EV deterministic problem using Gurobi
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
            self.model.write(f"./logs/{lp_name}_ev.lp")
        self.model.setParam('LogFile', './logs/gurobi_ev.log')

        start = time.time()
        self.model.optimize()
        end = time.time()
        comp_time = end - start
        print(f"computational time: {comp_time} s")

        
        sol = []   
        if self.model.status == grb.GRB.Status.OPTIMAL:
            for j in range(self.n_points):
                if self.Y[j].X > 0.5:
                    sol.append(j)
                    
        all_vars = self.model.getVars()
        values = self.model.getAttr("X", all_vars)
        obj_value_mean = self.model.getObjective().getValue()
        
        arr = np.array(values)
        Yminimizer = arr[0:self.n_points]
        low = self.n_points
        up = self.n_points*self.n_points + self.n_points
        Xminimizer = np.reshape(arr[low:up], (self.n_points,self.n_points))

        return Yminimizer, Xminimizer, obj_value_mean, sol, comp_time
    
    
    
    
    # Given the optimizers X, Y return the mean values of obj. function w.r.t. different scenatios
    def evaluate_meanf(self, Xopt, Yopt):
    
        points = range(self.n_points)
        scenarios = range(self.n_scenarios)
        
        fun = np.zeros(self.n_scenarios)
        
        # Sample mean
        for s in scenarios:
            fun[s] = sum(self.inst.d[i, j] * Xopt[i, j] for i in points for j in points) 
            fun[s] += self.inst.l * sum(max(0, (sum(self.inst.w[i, s] * Xopt[i,j] for i in points) - self.inst.C[j]*Yopt[j])) for j in points)
       
        obj_f_mean = np.mean(fun)
        
        return obj_f_mean
            
    
        
        