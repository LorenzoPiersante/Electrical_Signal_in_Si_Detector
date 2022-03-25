#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#See the log-book notes for the specific form of the data structures implemented


# In[ ]:


#The module contains the following functions:

#BASIC GAUSS SEIDEL ON A 5-DIAGONAL MATRIX
#global_residual(): this function computes the global residual of the system of equations provided the matrix of coefficients,
#the matrix of positions, and the guessed solution
#SOR_solver(): this function solves a system of linear equations using the SOR GS method within the required tolerance level
#SOR_one_iter(): this function performs one iteration of the SOR GS method, it is needed for code parallelisation
#SOR_one_iter_opp(): this function performs one iteration of the SOR GS method in the reversed cell order, it is useful to
#propagate the BC in both directions
#SOR_one_iter_empty_block(): this function performs one iteration of the SOR GS method in a solution domain where there are
#no inactive cells
#global_residual_5diagonals_optimised(): this function calculates the global residual for a system that involves a matrix of
#coefficients with 5 diagonals. The function operates only on a group of "active" cells in order to speed up the execution
#of the program

#These two fucntions are the oprimised version of the above functions for transient FVM problems:
#SOR_solver_5diagonals(): this function performs the SOR GS iteration for a system of 5 diagonals by running only over a
#group of active cells (see the 7-diagonal matrix functions for more details). The function returns the updated vector and
#a tuple containig the start cell and the end cell of the active group of cells.
#SOR_solver_5diagonals_one_iter(): this function performs one iteration of the previous method, it is useful in parallelised
#versions of the code

#GAUSS SEIDEL ON A 7-DIAGONAL MATRIX - OPTIMISED FOR TRANIENT PROBLEM OPERATION
#The following functions are optimised for the transient problem of drift-diffusion of charge carriers across the
#detector

#global_residual_7diagonals(): this function calculates the global residual for a system that involves a matrix of
#coefficients with 7 diagonals
#global_residual_7diagonals_optimised(): this function calculates the global residual for a system that involves a matrix of
#coefficients with 7 diagonals by iterating only over a limited group of cells, these cells are identified as the "active"
#cells of the system, in the drift-diffusion problem, these are the cells with concentration above a certain treshhold
#find_start_end(): this function identifies the location of the active cells in the solution vector according to
#some minimum concentration requirements. The iteration region is taken to be a certain number of cells away from the ones
#that fullfil the minimum requirement, the distance is determined by physical considerations
#SOR_solver_7diagonals(): this function works like the SOR_solver() function, but it acts on a matrix with 7 diagonals
#and it is specific for accplication to the FVM procedure that uses QUICK differencing, the solver applies deferred
#correction by default in order to imporve the stability properties of the QUICK matrix. The function is optimised for the
#solution of the drift-diffusion problem, in fact it returns the solution vector, but also a smaller vector that contains
#the non-zero components of the solution vector, this file is smaller and it will occupy less memory sapce when saved. The
#function also returns a start_end tuple which specifies the start and end cells of the smaller vector that will be saved.
#Knowing this information it is then possible to reconstruct the full concentration distribution (the other values are 0).
#SOR_solver_7diagonals_2(): it works like the previous function but it does not return the smaller vector, it returrns only
#the tuple with the location of the start and end cells of the region of the system that will be saved. The creation of
#the file for storage is done subsequently
#SOR_solver_7diagonals_one_iter_2(): this function performs an iteration of the above method, it is useful in parallelised
#versions of the code

#WEGHTING POTENTIAL CALCULATION
#In the context of the fully parallelised weighting potential calculation, these functions are needed to update the
#transition regions of a given block
#update_trans_regions(): this work on blocks with an electrode
#update_trans_regions_t(): this work on blocks without an electrode, that is blocks above the series of strips

#All the functions assume the usual cell numbering convention: bottom-top, left-right


# In[ ]:


#This module contains all the Gauus-Seidel Successive-Over-Relaxation based functions for the solution of the discretised
#system of equations
#The basic method is adapted to the different types of discretisation method used in the simulations


# In[2]:


import numpy as np


# In[3]:


#VARIABLES
#coefficients: matrix of coefficients
#position: matrix of positions
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method

#RETURNS
#R: global residual

def global_residual(coefficients, position, source, potential, active_cells):
    
    R = 0
    
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
        
        r_2 = (b - a_S*V_S - a_W*V_W - a_P*V_P - a_E*V_E - a_N*V_N)**2
        
        R = R + r_2
        
    R = np.sqrt(R)
        
    return R


# In[4]:


#VARIABLES
#coefficients: matrix of coefficients
#position: matrix of positions
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#over_relax: over-relaxation parameter used in SOR
#tolerance: global residual requiremnts for stopping the iterative method

#RETURNS
#potential: solution within specified tolerance

def SOR_solver(coefficients, position, source, potential, over_relax, tolerance, active_cells):
    
    
    iteration = 0
    R = 1
    
    while R > tolerance:
        
        #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
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
            V_E = potential[E]
            V_N = potential[N]

            b = source[n]
            
            #this is to check if a cell is an electrode
            if a_P == 0:
                continue
            else:
                potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
                #values are updated as we iterate over the domain (Gauss-Seidel approach)
        
        R = global_residual(coefficients, position, source, potential, active_cells)
        
        print(iteration, R)
        
        iteration = iteration + 1
        
    return potential


# In[5]:


#VARIABLES
#coefficients: matrix of coefficients
#position: matrix of positions
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: updated solution

def SOR_one_iter(coefficients, position, source, potential, over_relax, active_cells):
    
    #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
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
        V_E = potential[E]
        V_N = potential[N]

        b = source[n]

        if a_P == 0:
            continue
        else:
            potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
        
    return potential


# In[6]:


#VARIABLES
#coefficients: matrix of coefficients
#position: matrix of positions
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: updated solution

def SOR_one_iter_opp(coefficients, position, source, potential, over_relax, active_cells):
    
    #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
    for n in reversed(range(active_cells)):

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
        V_E = potential[E]
        V_N = potential[N]

        b = source[n]

        if a_P == 0:
            continue
        else:
            potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
        
    return potential


# In[8]:


#VARIABLES
#coefficients: matrix of coefficients
#position: matrix of positions
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#over_relax: over*relaxation parameter used in SOR

#RETURNS
#potential: updated solution

def SOR_one_iter_empty_block(coefficients, position, source, potential, over_relax, active_cells):
    
    #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
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
        V_E = potential[E]
        V_N = potential[N]

        b = source[n]
        
        potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
        
    return potential


# In[11]:


#VARIABLES
#coefficients: matrix of coefficients - 7-diagonal
#position: matrix of positions - 7-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method

#RETURNS
#R: global residual

def global_residual_7diagonals(coefficients, position, source, potential, active_cells):
    
    R = 0
    
    for n in range(active_cells):
        
        SS = int(position[n, 0])
        S = int(position[n, 1])
        W = int(position[n, 2])
        P = int(position[n, 3])
        E = int(position[n, 4])
        N = int(position[n, 5])
        NN = int(position[n, 6])
        
        a_SS = coefficients[n, 0]
        a_S = coefficients[n, 1]
        a_W = coefficients[n, 2]
        a_P = coefficients[n, 3]
        a_E = coefficients[n, 4]
        a_N = coefficients[n, 5]
        a_NN = coefficients[n, 6]
        
        V_SS = potential[SS]
        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]
        V_NN = potential[NN]
        
        b = source[n]
        
        r_2 = (b - a_S*V_S - a_W*V_W - a_P*V_P - a_E*V_E - a_N*V_N - a_SS*V_SS - a_NN*V_NN)**2
        
        R = R + r_2
    
    R = np.sqrt(R)
        
    return R


# In[12]:


#VARIABLES
#coefficients: matrix of coefficients - 7-diagonal
#position: matrix of positions - 7-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#start_end: tuple of start and end cell for the active region over which the function will act

#RETURNS
#R: global residual

def global_residual_7diagonals_optimised(coefficients, position, source, potential, active_cells, start_end):
    
    start_cell, end_cell = start_end
    
    R = 0
    
    #we want the loop to go from start cell to end cell!
    for n in range(start_cell, end_cell):
        
        SS = int(position[n, 0])
        S = int(position[n, 1])
        W = int(position[n, 2])
        P = int(position[n, 3])
        E = int(position[n, 4])
        N = int(position[n, 5])
        NN = int(position[n, 6])

        a_SS = coefficients[n, 0]
        a_S = coefficients[n, 1]
        a_W = coefficients[n, 2]
        a_P = coefficients[n, 3]
        a_E = coefficients[n, 4]
        a_N = coefficients[n, 5]
        a_NN = coefficients[n, 6]
        
        V_SS = potential[SS]
        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]
        V_NN = potential[NN]
        
        b = source[n]
        
        r_2 = (b - a_S*V_S - a_W*V_W - a_P*V_P - a_E*V_E - a_N*V_N - a_SS*V_SS - a_NN*V_NN)**2
        
        R = R + r_2
    
    R = np.sqrt(R)
        
    return R


# In[13]:


#VARIABLES
#coefficients: matrix of coefficients - 7-diagonal - QUICK METHOD IN FVM
#coefficients_UD: UD part of the matrix
#coefficeints_correction: 2nd order part of the matrix
#position: matrix of positions - 7-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#tolerance: global residual requirement
#minimum_conc: concentration treshold to identify the active region
#start_end_guess: tuple, start and end cell guess for finding the new active region
#over_relax: over-relaxation parameter used in SOR


#RETURNS
#potential: solution within required tolerance 
#start_end: start and end of the active region
#save_file: reduced vector for storage, this vector has appended as last two entries the start and end cell of its
#concentration values

def SOR_solver_7diagonals(coefficients, coefficients_UD, coefficeints_correction, position, source, potential, over_relax, tolerance, minimum_conc, start_end_guess, active_cells):
    
    start_guess, end_guess = start_end_guess
    
    #find the first concentration cell that meets the condition
    for n in range(start_guess, active_cells):
        if potential[n] < minimum_conc:
            continue
        else:
            start_index = n
            break

    #find the last concentration cell that meets the condition
    for n in reversed(range(end_guess)):
        if potential[n] < minimum_conc:
            continue
        else:
            end_index = n
            break

    start_cell = int(start_index -  5*100*100) #start from ... microns below this cell
    end_cell = int(end_index +  5*100*100) #start from ... microns above this cell

    if start_cell < 0:
        start_cell = 0
    if end_cell > active_cells:
        end_cell = active_cells

    #tuple to keep track of the start and end of the calculation domain
    start_end = (start_cell, end_cell)
    
    iteration = 0
    R = 1
    
    while R > tolerance:
        
        R_old = R
        potential_old = np.copy(potential)
        
        #we are going to save these cells only as a file in order to save memory
        save_file = []
        
        #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
        #we want the loop to go from start cell to end cell!
        for n in range(start_cell, end_cell):
            
            SS = int(position[n, 0])
            S = int(position[n, 1])
            W = int(position[n, 2])
            P = int(position[n, 3])
            E = int(position[n, 4])
            N = int(position[n, 5])
            NN = int(position[n, 6])

            #UD part coefficients
            a_S = coefficients_UD[n, 1]
            a_W = coefficients_UD[n, 2]
            a_P = coefficients_UD[n, 3]
            a_E = coefficients_UD[n, 4]
            a_N = coefficients_UD[n, 5]
            
            #deferred correction coefficents
            a_SS_c = coefficeints_correction[n, 0]
            a_S_c = coefficeints_correction[n, 1]
            a_W_c = coefficeints_correction[n, 2]
            a_P_c = coefficeints_correction[n, 3]
            a_E_c = coefficeints_correction[n, 4]
            a_N_c = coefficeints_correction[n, 5]
            a_NN_c = coefficeints_correction[n, 6]
            
            #all concentrations of old iteration
            V_SS_c = potential_old[SS]
            V_S_c = potential_old[S]
            V_W_c = potential_old[W]
            V_P_c = potential_old[P]
            V_E_c = potential_old[E]
            V_N_c = potential_old[N]
            V_NN_c = potential_old[NN]
            
            #all concentrations of new iteration
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            #source with deferred correction
            b = source[n] - a_SS_c*V_SS_c - a_S_c*V_S_c - a_W_c*V_W_c - a_P_c*V_P_c - a_E_c*V_E_c - a_N_c*V_N_c - a_NN_c*V_NN_c
            
            #this is to check if a cell is an electrode
            if a_P == 0:
                save_file.append(0)
                continue
            else:
                potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
                #values are updated as we iterate over the domain (Gauss-Seidel approach)
                save_file.append(potential[P])
                
                
        R = global_residual_7diagonals_optimised(coefficients, position, source, potential, active_cells, start_end)
        
        print(iteration, R)
        
        if R_old == R:
            break
        if iteration > 40:
            break
        
        iteration = iteration + 1
        
    return potential, start_end, save_file


# In[14]:


#VARIABLES
#coefficients: matrix of coefficients - 7-diagonal - QUICK METHOD IN FVM
#coefficients_UD: UD part of the matrix
#coefficeints_correction: 2nd order part of the matrix
#position: matrix of positions - 7-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#tolerance: global residual requirement
#minimum_conc: concentration treshold to identify the active region
#start_end_guess: tuple, start and end cell guess for finding the new active region
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: solution within required tolerance 
#start_end: start and end of the active region

def SOR_solver_7diagonals_2(coefficients, coefficients_UD, coefficeints_correction, position, source, potential, over_relax, tolerance, minimum_conc, start_end_guess, active_cells):
    
    start_guess, end_guess = start_end_guess
    
    #find the first concentration cell that meets the condition
    for n in range(start_guess, active_cells):
        if potential[n] < minimum_conc:
            continue
        else:
            start_index = n
            break

    #find the last concentration cell that meets the condition
    for n in reversed(range(end_guess)):
        if potential[n] < minimum_conc:
            continue
        else:
            end_index = n
            break

    start_cell = int(start_index -  5*100*100) #start from ... microns below this cell
    end_cell = int(end_index +  5*100*100) #start from ... microns above this cell

    if start_cell < 0:
        start_cell = 0
    if end_cell > active_cells:
        end_cell = active_cells

    #tuple to keep track of the start and end of the calculation domain
    start_end = (start_cell, end_cell)
        
    iteration = 0
    R = 1
    
    while R > tolerance:
        
        R_old = R
        potential_old = np.copy(potential)
        
        #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
        #we want the loop to go from start cell to end cell!
        for n in range(start_cell, end_cell):
            
            SS = int(position[n, 0])
            S = int(position[n, 1])
            W = int(position[n, 2])
            P = int(position[n, 3])
            E = int(position[n, 4])
            N = int(position[n, 5])
            NN = int(position[n, 6])

            #UD part coefficients
            a_S = coefficients_UD[n, 1]
            a_W = coefficients_UD[n, 2]
            a_P = coefficients_UD[n, 3]
            a_E = coefficients_UD[n, 4]
            a_N = coefficients_UD[n, 5]
            
            #deferred correction coefficents
            a_SS_c = coefficeints_correction[n, 0]
            a_S_c = coefficeints_correction[n, 1]
            a_W_c = coefficeints_correction[n, 2]
            a_P_c = coefficeints_correction[n, 3]
            a_E_c = coefficeints_correction[n, 4]
            a_N_c = coefficeints_correction[n, 5]
            a_NN_c = coefficeints_correction[n, 6]
            
            #all concentrations of old iteration
            V_SS_c = potential_old[SS]
            V_S_c = potential_old[S]
            V_W_c = potential_old[W]
            V_P_c = potential_old[P]
            V_E_c = potential_old[E]
            V_N_c = potential_old[N]
            V_NN_c = potential_old[NN]
            
            #all concentrations of new iteration
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            #source with deferred correction
            b = source[n] - a_SS_c*V_SS_c - a_S_c*V_S_c - a_W_c*V_W_c - a_P_c*V_P_c - a_E_c*V_E_c - a_N_c*V_N_c - a_NN_c*V_NN_c
            
            #this is to check if a cell is an electrode
            if a_P == 0:
                continue
            else:
                potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
                #values are updated as we iterate over the domain (Gauss-Seidel approach)
                
        R = global_residual_7diagonals_optimised(coefficients, position, source, potential, active_cells, start_end)
        
        print(iteration, R)
        
        if R_old == R:
            break
        if iteration > 40:
            break
        
        iteration = iteration + 1
        
    return potential, start_end


# In[15]:


#VARIABLES
#potential: concentration distribution at a given time-step
#start_end_guess: tuple, start and end cell guess for finding the new active region
#active_cells: cells that take part in the iterative solution method
#minimum_conc: concentration treshold to identify the active region

#RETURNS
#start_end: tuple with start and end cell of the active region

def find_start_end(potential, start_end_guess, active_cells, minimum_conc):
    
    start_guess, end_guess = start_end_guess
    
    #find the first concentration cell that meets the condition
    for n in range(start_guess, active_cells):
        if potential[n] < minimum_conc:
            continue
        else:
            start_index = n
            break

    #find the last concentration cell that meets the condition
    for n in reversed(range(end_guess)):
        if potential[n] < minimum_conc:
            continue
        else:
            end_index = n
            break

    start_cell = int(start_index -  5*100*100) #start from ... microns below this cell
    end_cell = int(end_index +  5*100*100) #start from ... microns above this cell

    if start_cell < 0:
        start_cell = 0
    if end_cell > active_cells:
        end_cell = active_cells

    #tuple to keep track of the start and end of the calculation domain
    start_end = (start_cell, end_cell)
        
    return start_end


# In[16]:


#VARIABLES
#coefficients: matrix of coefficients - 7-diagonal - QUICK METHOD IN FVM
#coefficients_UD: UD part of the matrix
#coefficeints_correction: 2nd order part of the matrix
#position: matrix of positions - 7-diagonal
#source: source vector
#potential: guessed solution
#potential_old: guessed solution, needed for implementation of deferred correction
#active_cells: cells that take part in the iterative solution method
#tolerance: global residual requirement
#minimum_conc: concentration treshold to identify the active region
#start_end_guess: tuple, start and end cell guess for finding the new active region
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: solution within required tolerance 
#R: global residual

def SOR_solver_7diagonals_one_iter_2(coefficients, coefficients_UD, coefficeints_correction, position, source, potential, potential_old, over_relax, start_end, active_cells):
    
    start_cell, end_cell = start_end
    
    #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
    #we want the loop to go from start cell to end cell!
    for n in range(start_cell, end_cell):

        SS = int(position[n, 0])
        S = int(position[n, 1])
        W = int(position[n, 2])
        P = int(position[n, 3])
        E = int(position[n, 4])
        N = int(position[n, 5])
        NN = int(position[n, 6])

        #UD part coefficients
        a_S = coefficients_UD[n, 1]
        a_W = coefficients_UD[n, 2]
        a_P = coefficients_UD[n, 3]
        a_E = coefficients_UD[n, 4]
        a_N = coefficients_UD[n, 5]

        #deferred correction coefficents
        a_SS_c = coefficeints_correction[n, 0]
        a_S_c = coefficeints_correction[n, 1]
        a_W_c = coefficeints_correction[n, 2]
        a_P_c = coefficeints_correction[n, 3]
        a_E_c = coefficeints_correction[n, 4]
        a_N_c = coefficeints_correction[n, 5]
        a_NN_c = coefficeints_correction[n, 6]

        #all concentrations of old iteration
        V_SS_c = potential_old[SS]
        V_S_c = potential_old[S]
        V_W_c = potential_old[W]
        V_P_c = potential_old[P]
        V_E_c = potential_old[E]
        V_N_c = potential_old[N]
        V_NN_c = potential_old[NN]

        #all concentrations of new iteration
        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]

        #source with deferred correction
        b = source[n] - a_SS_c*V_SS_c - a_S_c*V_S_c - a_W_c*V_W_c - a_P_c*V_P_c - a_E_c*V_E_c - a_N_c*V_N_c - a_NN_c*V_NN_c

        #this is to check if a cell is an electrode
        if a_P == 0:
            continue
        else:
            potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
            #values are updated as we iterate over the domain (Gauss-Seidel approach)

    R = global_residual_7diagonals_optimised(coefficients, position, source, potential, active_cells, start_end)
    
    return potential, R


# In[18]:


#VARIABLES
#coefficients: matrix of coefficients - 5-diagonal
#position: matrix of positions - 5-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#start_end: tuple of start and end cell for the active region over which the function will act

#RETURNS
#R: global residual

def global_residual_5diagonals_optimised(coefficients, position, source, potential, active_cells, start_end):
    
    start_cell, end_cell = start_end
    
    R = 0
    
    #we want the loop to go from start cell to end cell!
    for n in range(start_cell, end_cell):
        
        S = int(position[n, 0])
        W = int(position[n, 1])
        P = int(position[n, 2])
        E = int(position[n, 3])
        N = int(position[n, 4])

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
        
        r_2 = (b - a_S*V_S - a_W*V_W - a_P*V_P - a_E*V_E - a_N*V_N)**2
        
        R = R + r_2
    
    R = np.sqrt(R)
        
    return R


# In[19]:


#VARIABLES
#coefficients: matrix of coefficients - 5-diagonal - QUICK METHOD IN FVM
#position: matrix of positions - 5-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#tolerance: global residual requirement
#minimum_conc: concentration treshold to identify the active region
#start_end_guess: tuple, start and end cell guess for finding the new active region
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: solution within required tolerance 
#start_end: start and end of the active region

def SOR_solver_5diagonals(coefficients_UD, position, source, potential, over_relax, tolerance, minimum_conc, start_end_guess, active_cells):
    
    N_y = 3000
    N_x = 1000
    
    start_guess, end_guess = start_end_guess
    
    #find the first concentration cell that meets the condition
    for n in range(start_guess, active_cells):
        if potential[n] < minimum_conc:
            continue
        else:
            start_index = n
            break

    #find the last concentration cell that meets the condition
    for n in reversed(range(end_guess)):
        if potential[n] < minimum_conc:
            continue
        else:
            end_index = n
            break

    start_cell = int(start_index -  5*100*100) #start from ... microns below this cell
    end_cell = int(end_index +  5*100*100) #start from ... microns above this cell

    if start_cell < 0:
        start_cell = 0
    if end_cell > active_cells:
        end_cell = active_cells

    #tuple to keep track of the start and end of the calculation domain
    start_end = (start_cell, end_cell)
    
    iteration = 0
    R = 1
    
    while R > tolerance:
        
        R_old = R
        
        #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
        #we want the loop to go from start cell to end cell!
        for n in range(start_cell, end_cell):
            
            S = int(position[n, 0])
            W = int(position[n, 1])
            P = int(position[n, 2])
            E = int(position[n, 3])
            N = int(position[n, 4])
            
            a_S = coefficients_UD[n, 0]
            a_W = coefficients_UD[n, 1]
            a_P = coefficients_UD[n, 2]
            a_E = coefficients_UD[n, 3]
            a_N = coefficients_UD[n, 4]
            
            
            #all concentrations of new iteration
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            #source with deferred correction
            b = source[n]
            
            #this is to check if a cell is an electrode
            if a_P == 0:
                continue
            else:
                potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
                
                
        R = global_residual_5diagonals_optimised(coefficients_UD, position, source, potential, active_cells, start_end)
        
        print(iteration, R)
        
        if R_old == R:
            break
        if iteration > 40:
            break
        
        iteration = iteration + 1
        
    return potential, start_end


# In[ ]:


#VARIABLES
#coefficients: matrix of coefficients - 5-diagonal - QUICK METHOD IN FVM
#position: matrix of positions - 5-diagonal
#source: source vector
#potential: guessed solution
#active_cells: cells that take part in the iterative solution method
#minimum_conc: concentration treshold to identify the active region
#start_end_guess: tuple, start and end cell guess for finding the new active region
#over_relax: over-relaxation parameter used in SOR

#RETURNS
#potential: solution within required tolerance 
#R: global residual

def SOR_solver_5diagonals_one_iter(coefficients_UD, position, source, potential, over_relax, start_end, active_cells):
    
    
    start_cell, end_cell = start_end

    #Gauss/Seidel-type iteration, the values are updated immediately as they are calculated
    #we want the loop to go from start cell to end cell!
    for n in range(start_cell, end_cell):

        S = int(position[n, 0])
        W = int(position[n, 1])
        P = int(position[n, 2])
        E = int(position[n, 3])
        N = int(position[n, 4])

        a_S = coefficients_UD[n, 0]
        a_W = coefficients_UD[n, 1]
        a_P = coefficients_UD[n, 2]
        a_E = coefficients_UD[n, 3]
        a_N = coefficients_UD[n, 4]


        #all concentrations of new iteration
        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]

        #source with deferred correction
        b = source[n]

        #this is to check if a cell is an electrode
        if a_P == 0:
            continue
        else:
            potential[P] = (1-over_relax)*potential[P] + (over_relax/a_P)*(-a_S*V_S -a_W*V_W -a_E*V_E -a_N*V_N + b)
                
                
    R = global_residual_5diagonals_optimised(coefficients_UD, position, source, potential, active_cells, start_end)
        
    return potential, R


# In[9]:


#UPDATE TRANSITION REGIONS of bottom blocks for weighting field computation
#B_1 is either top or bottom

def update_trans_regions(potential, B_1, B_left, B_right, N_x, N_y):
    
    B_1 = potential[(N_y-1)*N_x : N_y*N_x]

    for i in range(N_y-1):
        location = int((i+1)*N_x)
        B_left[i] = potential[location]

    for i in range(N_y-1):
        location = int((i+2)*N_x -1)
        B_right[i] = potential[location]
        
    return (B_1, B_left, B_right)


# In[10]:


#UPDATE TRANSITION REGIONS of top blocks for weighting field computation
#B_1 is either top or bottom

def update_trans_regions_t(potential, B_1, B_left, B_right, N_x, N_y):
    
    B_1 = potential[0:N_x]

    for i in range(N_y-1):
        location = int(i*N_x)
        B_left[i] = potential[location]

    for i in range(N_y-1):
        location = int((i+1)*N_x -1)
        B_right[i] = potential[location]
        
    return (B_1, B_left, B_right)

