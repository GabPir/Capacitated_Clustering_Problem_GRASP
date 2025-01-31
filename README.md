# Capacitated Clustering Problem: EV Optimization Problem and Grasp Heuristic


## Project 2 of "Numerical methods for Large Scale Problems 22/23"

In this project, we aim to tackle the CCP - Capacitated Clustering Problem, where the goal is to find an optimal clustering given a capacity constraint on the centroids and a stochastic weight for each individual cluster element (e.g., demand for a service, supply of a service in a spatial context).
The objective function to be minimized considers both the distance between points and their assigned cluster and the capacity constraint (the constraint is relaxed by introducing a penalty even when there is unused capacity).
<br>
![immagine](https://github.com/user-attachments/assets/6f75f155-4019-4895-94d7-47da8b1a598b)
<br>
The problem is a two-stage problem, where in the first stage, the centroids are selected, and in the second stage, the assignments to these centroids (clustering) are made.Between the first and second stage, the stochastic weight factors w_i (capacity cost) of individual points are observed.The algorithm operates by simulating a finite number of scenarios, using a discrete expected value approach.
<br>
<br>
For optimization, the Python package Gurobi© (https://www.gurobi.com/) was used.
### Pt1: Expected Value Optimization
Uncertainty is addressed by considering the expected value formulation, where the stochastic weights are replaced by the expected value of the weights across different scenarios:
<br>
![immagine](https://github.com/user-attachments/assets/ce0ccdbd-1953-46e4-923a-d6ffcfc0c03c)

<br> The results obtained between the two models are:

![immagine](https://github.com/user-attachments/assets/beb12d66-d988-4b6d-b333-7dbc4867aa12)
![immagine](https://github.com/user-attachments/assets/7f162958-7d6e-4e7d-83e1-bba9f4d0f081)

### Pt2: Heuristic GRASP method
Larger-scale problems often require heuristic approaches. A Greedy Randomized Adaptive Search Procedure based on k-means was used to cluster the data and solve the optimization problem in the subproblems, with the goal of determining the best medoid in each subcluster (local search phase). A version with lower computational cost involves selecting the new centroids only from the top 20% of the most promising points.
<br>
The pseudocode is:
<br>
![immagine](https://github.com/user-attachments/assets/1e8af8cb-53dd-4140-9672-607d6fe5d276)
<br>
Our algorithm meets expectations, and despite a slight decrease in overall cost, there is a significant increase in computational costs:
![immagine](https://github.com/user-attachments/assets/5b741eca-71d4-4840-a839-ec4da874ee9c)
<br>
Furthermore, we evaluated how the computational cost and performance varied by increasing the weight of the stochastic factor in the second stage (capacity mismatch in the clusters) in the optimization problem, achieving excellent results from the use of the heuristic:
<br>
![immagine](https://github.com/user-attachments/assets/7eb67cb3-1fe0-4bc0-bb36-e11056fd302a)
<br>
In conclusion, the number of clusters allows for the parallel execution of the optimization subproblems, thereby improving the performance of the GRASP algorithm:
<br>
![immagine](https://github.com/user-attachments/assets/60b6bda9-7327-4ff2-8005-db7dc5bdb87e)
<br>
<br>

