#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This module contains the function required to implement a Conjugate-Gradient solution procedure of the discretised
#equations
#Refer to the notes on the log-book for the mathematical details of the process


# In[ ]:


#The functions included in the module are:

#global_vector_residual(): given the matrix of coefficients, a solution vector and a matric of positions, this functions
#returns the global vector residual of the system of equations, this function is useful also in the implementation of
#multi-grid methods
#CG_solver(): this function takes the matrix of coefficients, the matrix of positions, a guess solution and returns the
#solution to the system within the specified tolerance

#for a parallelised solution method we want to separate the ietartions, so the following two function were defined:
#CG_0th_iter(): it performs the initiator step of the CG solution procedure
#CG_one_iter(): it performs one iteration of the CG solution procedure

#Below details are provided regarding the inputs and outputs of the individual functions


# In[ ]:


import numpy as np


# In[ ]:


#VARIABLES
#coefficients and position: matrix of coefficients and of positions
#source: source vector of the problem
#potential: guess solution
#active_cells: total number of cells over wich the for-loops in the function will run
#start: the cells from which the function starts iterating

#RETURNS
#R: the global vector residual of the system

def global_vector_residual(coefficients, position, source, potential, active_cells, start):
    
    R = np.zeros(len(potential))
    
    for n in range(active_cells):
        
        S = position[n, 0]
        W = position[n, 1]
        P = position[n, 2]
        E = position[n, 3]
        N = position[n, 4]
        
        a_S = coefficients[n, 0]
        a_W = coefficients[n, 1]
        a_P = coefficients[n, 2]
        a_E = coefficients[n, 3]
        a_N = coefficients[n, 4]
        
        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]
        
        b = source[n]
        
        r = (b - a_S*V_S - a_W*V_W - a_P*V_P - a_E*V_E - a_N*V_N)
        
        location = start + n
        R[location] = r
        
    return R


# In[ ]:


#VARIABLES
#coefficients and position: matrix of coefficients and of positions
#source: source vector of the problem
#potential: initial guess
#active_cells: total number of cells over wich the for-loops in the function will run
#start: the cells from which the function starts iterating
#tolerance: the global residual at which the iterations stop

#RETURNS
#potential: solution within the required tolerance level

def CG_solver(coefficients, position, source, potential, active_cells, start, tolerance):
    
    residual_0 = global_vector_residual(coefficients, position, source, potential, active_cells, start)
    R_tot = np.sqrt(np.dot(residual_0, residual_0))
    print(R_tot)
    
    search_0 = np.copy(residual_0)
    
    ###########################################################################################################################
    #compute alpha_1
    
    numerator = np.dot(residual_0, residual_0)
    
    #denominator
    
    #1st we compute A*search_0
    
    A_search_0 = np.zeros(active_cells)
    
    for n in range(active_cells):
        
        S = position[n, 0]
        W = position[n, 1]
        P = position[n, 2]
        E = position[n, 3]
        N = position[n, 4]
        
        a_S = coefficients[n, 0]
        a_W = coefficients[n, 1]
        a_P = coefficients[n, 2]
        a_E = coefficients[n, 3]
        a_N = coefficients[n, 4]
        
        s_S = search_0[S]
        s_W = search_0[W]
        s_P = search_0[P]
        s_E = search_0[E]
        s_N = search_0[N]
        
        location = start + n
        A_search_0[location] = (a_S*s_S + a_W*s_W + a_P*s_P + a_E*s_E + a_N*s_N)
        
    #2nd we copute A_search_0 dot search_0
    
    denominator = np.dot(A_search_0, search_0)
    
    #finally
    
    alpha_1 = numerator/denominator
    ############################################################################################################################
    
    #update the potential
    potential = potential + alpha_1*search_0
    
    #compute new global vector residual
    residual_1 = global_vector_residual(coefficients, position, source, potential, active_cells, start)
    R_tot = np.dot(residual_1, residual_1)
    print(R_tot)
    
    #compute beta factor
    beta_1 = np.dot(residual_1, residual_1)/np.dot(residual_0, residual_0)
    
    #update serarch direction
    search_1 = residual_1 + beta_1*search_0
    
    #general iteration
    residual_n = np.copy(residual_1)
    search_n = np.copy(search_1)
    
    iteration = 0
    while R_tot > tolerance:
        
        ###########################################################################################################################
        #compute alpha_n

        numerator = np.dot(residual_n, residual_n)

        #denominator

        #1st we compute A*search_n

        A_search_n = np.zeros(active_cells)

        for n in range(active_cells):

            S = position[n, 0]
            W = position[n, 1]
            P = position[n, 2]
            E = position[n, 3]
            N = position[n, 4]

            a_S = coefficients[n, 0]
            a_W = coefficients[n, 1]
            a_P = coefficients[n, 2]
            a_E = coefficients[n, 3]
            a_N = coefficients[n, 4]

            s_S = search_n[S]
            s_W = search_n[W]
            s_P = search_n[P]
            s_E = search_n[E]
            s_N = search_n[N]

            location = start + n
            A_search_n[location] = (a_S*s_S + a_W*s_W + a_P*s_P + a_E*s_E + a_N*s_N)

        #2nd we copute A_search_0 dot search_0

        denominator = np.dot(A_search_n, search_n)

        #finally

        alpha_n_1 = numerator/denominator
        ############################################################################################################################
        
        #update potential
        potential = potential + alpha_n_1*search_n
        
        #old residual
        old_residual = np.copy(residual_n)
        
        #calculate new residual
        residual_n = global_vector_residual(coefficients, position, source, potential, active_cells, start)
        
        #new global residual
        R_tot = np.sqrt(np.dot(residual_n, residual_n))
        
        #beta factor
        beta_n_1 = np.dot(residual_n, residual_n)/np.dot(old_residual, old_residual)
        
        #update search vector
        search_n = residual_n + beta_n_1*search_n
        
        #check convergence
        print(iteration, R_tot)
        iteration = iteration + 1
        
    return potential


# In[ ]:


#VARIABLES
#coefficients and position: matrix of coefficients and of positions
#source: source vector of the problem
#potential: initial guess
#active_cells: total number of cells over wich the for-loops in the function will run
#start: the cells from which the function starts iterating

#RETURNS
#potential: updated solution
#search_1: search direction
#residual_1: global vector residual

def CG_0th_iter(coefficients, position, source, potential, active_cells, start):
    
    residual_0 = global_vector_residual(coefficients, position, source, potential, active_cells, start)
    R_tot = np.sqrt(np.dot(residual_0, residual_0))
    print(R_tot)
    
    search_0 = np.copy(residual_0)
    
    ###########################################################################################################################
    #compute alpha_1
    
    numerator = np.dot(residual_0, residual_0)
    
    #denominator
    
    #1st we compute A*search_n
    
    A_search_0 = np.zeros(active_cells)
    
    for n in range(active_cells):
        
        S = position[n, 0]
        W = position[n, 1]
        P = position[n, 2]
        E = position[n, 3]
        N = position[n, 4]
        
        a_S = coefficients[n, 0]
        a_W = coefficients[n, 1]
        a_P = coefficients[n, 2]
        a_E = coefficients[n, 3]
        a_N = coefficients[n, 4]
        
        s_S = search_0[S]
        s_W = search_0[W]
        s_P = search_0[P]
        s_E = search_0[E]
        s_N = search_0[N]
        
        location = start + n
        A_search_0[location] = (a_S*s_S + a_W*s_W + a_P*s_P + a_E*s_E + a_N*s_N)
        
    #2nd we copute A_search_0 dot search_0
    
    denominator = np.dot(A_search_0, search_0)
    
    #finally
    
    alpha_1 = numerator/denominator
    ############################################################################################################################
    
    #update the potential
    potential = potential + alpha_1*search_0
    
    #compute new global vector residual
    residual_1 = global_vector_residual(coefficients, position, source, potential, active_cells, start)
    R_tot = np.sqrt(np.dot(residual_1, residual_1))
    print(R_tot)
    
    #compute beta factor
    beta_1 = np.dot(residual_1, residual_1)/np.dot(residual_0, residual_0)
    
    #update serarch direction
    search_1 = residual_1 + beta_1*search_0
    
    return potential, search_1, residual_1


# In[ ]:


#VARIABLES
#coefficients and position: matrix of coefficients and of positions
#source: source vector of the problem
#potential: solution at iteration n-1
#active_cells: total number of cells over wich the for-loops in the function will run
#start: the cells from which the function starts iterating
#search_n: search direction at iteration n-1
#residual_n: global vector residual at iteration n-1

#RETURNS
#potential: updated solution
#search_1: updated search direction
#residual_1: updated global vector residual
#R_tot: global residual

def CG_one_iter(coefficients, position, source, potential, active_cells, start, search_n, residual_n):
        
    ###########################################################################################################################
    #compute alpha_n

    numerator = np.dot(residual_n, residual_n)

    #denominator

    #1st we compute A*search_n

    A_search_n = np.zeros(active_cells)

    for n in range(active_cells):

        S = position[n, 0]
        W = position[n, 1]
        P = position[n, 2]
        E = position[n, 3]
        N = position[n, 4]

        a_S = coefficients[n, 0]
        a_W = coefficients[n, 1]
        a_P = coefficients[n, 2]
        a_E = coefficients[n, 3]
        a_N = coefficients[n, 4]

        s_S = search_n[S]
        s_W = search_n[W]
        s_P = search_n[P]
        s_E = search_n[E]
        s_N = search_n[N]

        location = start + n
        A_search_n[location] = (a_S*s_S + a_W*s_W + a_P*s_P + a_E*s_E + a_N*s_N)

    #2nd we copute A_search_0 dot search_0

    denominator = np.dot(A_search_n, search_n)

    #finally

    alpha_n_1 = numerator/denominator
    ############################################################################################################################
        
    #update potential
    potential = potential + alpha_n_1*search_n

    #old residual
    old_residual = np.copy(residual_n)

    #calculate new residual
    residual_n = global_vector_residual(coefficients, position, source, potential, active_cells, start)

    #new global residual
    R_tot = np.sqrt(np.dot(residual_n, residual_n))

    #beta factor
    beta_n_1 = np.dot(residual_n, residual_n)/np.dot(old_residual, old_residual)

    #update search vector
    search_n = residual_n + beta_n_1*search_n
    
    return potential, search_n, residual_n, R_tot

