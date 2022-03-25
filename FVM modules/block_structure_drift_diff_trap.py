#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This module contains function that create the matrices of coefficients and position in a situation of drift-diffusion with
#trapping occurring

#The functions are built considering a single block (i.e. a single strip), so that the boundary conditions on the left and right
#side are set to no-flux

#The functions are identical to the one for drift-diffusion with the addition of a term in the central coefficient


# In[ ]:


#ONE BLOCK MOTION - QUICK BASED
#drift_diff_trap_block(): creates matrix of positions and coefficients, applies no flux boundary conditions on both RHS and LHS
#boundaries
#drift_diff_trap_block_UD_part(): corresponding UD coefficients
#THE SOURCE CREATORS DEFINED IN THE block_charge_drift MODULE WILL CREATE THE APPROPRIATE SOURCES
#NOTE THAT THE DIFFUSIVE TERM WILL ALL BE COLLECTED IN THE UD PART OF THE QUICK METHOD, THIS MEANS THAT THE CORRECTION PART
#DOES NOR REQUIRE ANY MODIFICATION

#ONE BLOCK MOTION - UD BASED
#drift_diff_trap_block_UD(): creates matrix of positions and coefficients, applies no flux boundary conditions on both the RHS
#and LHS
#THE SOURCE CREATORS DEFINED IN THE block_charge_drift MODULE WILL CREATE THE APPROPRIATE SOURCES


# In[ ]:


#MATRIX CREATOR VARIABLES

#height and pitch: dimensions of the region of the Si detector we are interested in
#height is the thickness of the detector, pitch is the separation between the centres of neighbouring strips
#the functions below locate the electrode/strip at the centre of the solution domain and takes the total width of the domain
#to be the pitch (i.e. the strip is located at pitch/2)

#potential: it is a flattened version of the potential map, i.e. the 2D map is turned in a 1D array according to the usual
#numbering convention

#mobility: mobility of electrons or holes
#time_step: size of time advancement specified in seconds
#time_const: characteristic trapping time expressed in seconds

#charge_carrier: +1 for hole, -1 for electron, it determines the direction of the drift

#method: "c" to create a matrix based on CN time differencing, "e" for Euler-impicit time differencing

#The variables are the same for all functions, yet they are processed in different ways and the putput differs as explained in
#the brief description


# In[ ]:


import numpy as np

#the time const must be expressed in seconds
#mobility must be expressed in m^2 V^-1 s^-1


# In[ ]:


def drift_diff_trap_block(height, pitch, potential, mobility, time_const, charge_carrier, time_step, method):
    
    mesh_size = 0.1

    #diffusion factor D/mu, V at 300K
    diff_fact = 0.02584
    D = diff_fact
        
    trap_fact = ((mesh_size*10**(-6))**2)/(mobility*time_const)
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #first we define the matrix of positions, usual numbering convention
    
    positions = []
    
    #bottom-left corner
    positions.append([0, 0, 0, 0, 1, N_x, 2*N_x])
    
    for j in range(1, N_x-1):
        location = j
        positions.append([0, 0, location-1, location, location + 1, location + N_x, location + 2*N_x])
        
    #bottom-right corner
    
    positions.append([0, 0, N_x - 2, N_x - 1, 0, 2*N_x - 1, 3*N_x - 1])
    
    #first row
    #left corner
    positions.append([0, 0, 0, N_x, N_x + 1, 2*N_x, 3*N_x])
    
    for j in range(1, N_x-1):
        location = N_x + j
        positions.append([0, location - N_x, location-1, location, location + 1, location + N_x, location + 2*N_x])
        
    #right corner
    positions.append([0, N_x-1, 2*N_x - 2, 2*N_x - 1, 0, 3*N_x - 1, 4*N_x - 1])
    
    #bulk
    for i in range(2, N_y-6):
        positions.append([(i-2)*N_x, (i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x, (i+2)*N_x])
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - 2*N_x, location - N_x, location-1, location, location + 1, location + N_x, location + 2*N_x])
        positions.append([(i-1)*N_x - 1, i*N_x -1, (i+1)*N_x -2, (i+1)*N_x -1, 0, (i+2)*N_x - 1, (i+3)*N_x - 1])
    
    #penultimate row
    positions.append([(N_y-8)*N_x, (N_y-7)*N_x, 0, (N_y-6)*N_x, (N_y-6)*N_x + 1, (N_y-5)*N_x, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-6)*N_x + j
        positions.append([location - 2*N_x, location - N_x, location -1, location, location + 1, location + N_x, 0])
        
    positions.append([(N_y-7)*N_x - 1, (N_y-6)*N_x - 1, (N_y-5)*N_x - 2, (N_y-5)*N_x - 1, 0, (N_y-4)*N_x - 1, 0])
    
    #top-left corner
    positions.append([(N_y-7)*N_x, (N_y-6)*N_x, 0, (N_y-5)*N_x, (N_y-5)*N_x + 1, 0, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-5)*N_x + j
        positions.append([location - 2*N_x, location - N_x, location -1, location, location + 1, 0, 0])
    
    #top-right corner
    positions.append([(N_y-6)*N_x - 1, (N_y-5)*N_x - 1, (N_y-4)*N_x - 2, (N_y-4)*N_x - 1, 0, 0, 0])
    
    for i in range(N_y-4, N_y):
        for j in range(N_x):
            positions.append([0, 0, 0, 0, 0, 0, 0])
    
    positions = np.array(positions)
    
    
    #use the matix of positions to loop over the potential and create the matrix of coefficients
    coefficients = []
    last_cell = int(N_x-1)
    
    #mesh-size is in micrometers!!!
    if method == "e":
        mob_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    elif method == "c":
        mob_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    #first row set to 0 concentration, these are inactive cells
    for j in range(3*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    #For the bulk we use QUICK
    for i in range(3, N_y-10):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 1]
            W = positions[location, 2]
            P = positions[location, 3]
            E = positions[location, 4]
            N = positions[location, 5]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = - D
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = a_e - D
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = - D
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = 0
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

            
            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = 0
                        a_W = a_w - D
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = 0
                        a_W = - D
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = 0
                        a_W = - D
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
            
            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = - D
                        a_W = a_w - D
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = a_e - D
                        a_W = - D
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e*E_w < 0:
                        a_NN = 0
                        a_N = (3/8)*a_n - D
                        a_S = a_s - a_s/4 - a_n/8 - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_w/2 + a_e/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = - D
                        a_W = a_w - D
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e*E_w < 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8 - D
                        a_S = (3/8)*a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_w/2 + a_e/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
    
    #The last two rows contain the inactive p+ implant
    #Fill coefficients up to here taking advantage of the fact that I know where E-field is pointing
    for i in range(N_y-10, N_y-5):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 1]
            W = positions[location, 2]
            P = positions[location, 3]
            E = positions[location, 4]
            N = positions[location, 5]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j in range(int((2/5)*N_x), int((3/5)*N_x)):
                coefficients.append([0, 0, 0, 0, 0, 0, 0])
            else:
                if j == 0:
                    E_w = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = - D
                            a_W = 0
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = a_e - D
                            a_W = 0
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = - D
                            a_W = 0
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = a_e - D
                            a_W = 0
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = 0
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])


                elif j == last_cell:
                    E_e = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = 0
                            a_W = a_w - D
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = 0
                            a_W = - D
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = 0
                            a_W = - D
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

                else:
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = - D
                            a_W = a_w - D
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = a_e - D
                            a_W = - D
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e*E_w < 0:
                            a_NN = 0
                            a_N = (3/8)*a_n - D
                            a_S = a_s - a_s/4 - a_n/8 - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_SS = -a_s/8
                            a_P = a_n + (3/8)*a_s - a_n/4 + a_w/2 + a_e/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = - D
                            a_W = a_w - D
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e <= 0 and E_w <= 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = a_e - D
                            a_W = - D
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        elif E_e*E_w < 0:
                            a_NN = -a_n/8
                            a_N = a_n - a_n/4 - a_s/8 - D
                            a_S = (3/8)*a_s - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_SS = 0
                            a_P = a_s + (3/8)*a_n - a_s/4 + a_w/2 + a_e/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e - D
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e*E_w < 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                            
    for j in range(N_x):
            
        location = (N_y-5)*N_x + j

        S = positions[location, 1]
        W = positions[location, 2]
        P = positions[location, 3]
        E = positions[location, 4]
        N = positions[location, 5]

        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]

        a_s = (V_P - V_S)*charge_carrier
        a_w = (V_P - V_W)*charge_carrier
        a_e = (V_P - V_E)*charge_carrier
        a_n = (V_P - V_N)*charge_carrier

        E_s = -(V_P - V_S)*charge_carrier
        E_n = -(V_N - V_P)*charge_carrier
        E_e = -(V_E - V_P)*charge_carrier
        E_w = -(V_P - V_W)*charge_carrier

        if j in range(int((2/5)*N_x), int((3/5)*N_x)):
            coefficients.append([0, 0, 0, 0, 0, 0, 0])
        else:
            E_n = 0
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = - D
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = a_e - D
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = - D
                        a_W = 0
                        a_SS = 0
                        a_P = a_s - a_s/4 + a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_SS = 0
                        a_P = a_s - a_s/4 + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])

            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = 0
                        a_W = a_w - D
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = 0
                        a_W = - D
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s - a_s/4 + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = 0
                        a_W = - D
                        a_SS = 0
                        a_P = a_s - a_s/4 + a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])

            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E =  - D
                        a_W = a_w  - D
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = a_e - D
                        a_W  = - D
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e*E_w < 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s - a_s/4 - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s + a_w/2 + a_e/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E =  - D
                        a_W = a_w - D
                        a_SS = 0
                        a_P = a_s - a_s/4 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_SS = 0
                        a_P = a_s - a_s/4 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e*E_w < 0:
                        a_NN = 0
                        a_N = -a_s/8
                        a_S = (3/8)*a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_SS = 0
                        a_P = a_s - a_s/4 + a_w/2 + a_e/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                        
    for i in range(N_y-4, N_y):
        for j in range(N_x):
            coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    coefficients = np.array(coefficients)
                        
    return coefficients, positions


# In[ ]:


def drift_diff_trap_block_UD_part(height, pitch, positions, potential, mobility, time_const, charge_carrier, time_step, method):
    
    mesh_size = 0.1
    
    trap_fact = ((mesh_size*10**(-6))**2)/(mobility*time_const)
    
    #diffusion factor D/mu, V at 300K
    diff_fact = 0.02584
    D = diff_fact
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #use the matix of positions to loop over the potential and create the matrix of coefficients
    coefficients = []
    last_cell = int(N_x-1)
    
    #mesh-size is in micrometers!!!
    if method == "e":
        mob_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    elif method == "c":
        mob_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    #first row set to 0 concentration, these are inactive cells
    for j in range(3*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
        
    #In the first two rows we use UD
    for i in range(3, N_y-10):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 1]
            W = positions[location, 2]
            P = positions[location, 3]
            E = positions[location, 4]
            N = positions[location, 5]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = - D
                        a_W = 0
                        a_P = a_n + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_n + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = - D
                        a_W = 0
                        a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = 0
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
            
            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_n + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = 0
                        a_W = - D
                        a_P = a_n + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        
            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_n + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_n + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_n + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_s + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_s + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
    
    #the last two rows contain the inactive p+ implant
    for i in range(N_y-10, N_y-5):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 1]
            W = positions[location, 2]
            P = positions[location, 3]
            E = positions[location, 4]
            N = positions[location, 5]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j in range(int((2/5)*N_x), int((3/5)*N_x)):
                coefficients.append([0, 0, 0, 0, 0, 0, 0])
            else:
                if j == 0:
                    E_w = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = - D
                            a_W = 0
                            a_P = a_n + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_n + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = - D
                            a_W = 0
                            a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_s + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = 0
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

                elif j == last_cell:
                    E_e = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_n + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = 0
                            a_W = - D
                            a_P = a_n + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S  = - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = 0
                            a_W = - D
                            a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

                else:
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_n + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e - D
                            a_W = - D
                            a_P = a_n + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e*E_w < 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_n + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_s + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e - D
                            a_W = - D
                            a_P = a_s + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e*E_w < 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_s + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2- D
                            a_E = a_e- D
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                        elif E_e*E_w < 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                            
                            
    for j in range(N_x):
        
        location = (N_y-5)*N_x + j
            
        S = positions[location, 1]
        W = positions[location, 2]
        P = positions[location, 3]
        E = positions[location, 4]
        N = positions[location, 5]

        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]

        a_s = (V_P - V_S)*charge_carrier
        a_w = (V_P - V_W)*charge_carrier
        a_e = (V_P - V_E)*charge_carrier
        a_n = (V_P - V_N)*charge_carrier

        E_s = -(V_P - V_S)*charge_carrier
        E_n = -(V_N - V_P)*charge_carrier
        E_e = -(V_E - V_P)*charge_carrier
        E_w = -(V_P - V_W)*charge_carrier

        if j in range(int((2/5)*N_x), int((3/5)*N_x)):
            coefficients.append([0, 0, 0, 0, 0, 0, 0])
        else:
            E_n = 0
    
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = - D
                        a_W = 0
                        a_P = a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = - D
                        a_E = - D
                        a_W = 0
                        a_P = a_s + a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = 0
                        a_W = - D
                        a_P = a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0 
                        a_S = - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])

            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_e/2 + a_w/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                    elif E_e*E_w < 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_s + a_e/2 + a_w/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([0, a_S, a_W, a_P, a_E, a_N, 0])
                             
    for i in range(N_y-4, N_y):
        for j in range(N_x):
            coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    coefficients = np.array(coefficients)
    
    return coefficients


# In[ ]:


def drift_diff_trap_block_UD(height, pitch, potential, mobility, time_const, charge_carrier, time_step, method):
    
    mesh_size = 0.1
    
    trap_fact = ((mesh_size*10**(-6))**2)/(mobility*time_const)
    
    #diffusion factor D/mu, V at 300K
    diff_fact = 0.02584
    D = diff_fact
        
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #first we define the matrix of positions, usual numbering convention
    
    positions = []
    
    #bottom-left corner
    positions.append([0, 0, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        location = j
        positions.append([0, location-1, location, location + 1, location + N_x])
        
    #bottom-right corner
    
    positions.append([0, N_x - 2, N_x - 1, 0, 2*N_x - 1])
    
    #bulk
    for i in range(1, N_y-5):
        positions.append([(i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x])
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location-1, location, location + 1, location + N_x])
        positions.append([i*N_x -1, (i+1)*N_x -2, (i+1)*N_x -1, 0, (i+2)*N_x - 1])
    
    #last row
    positions.append([(N_y-6)*N_x, 0, (N_y-5)*N_x, (N_y-5)*N_x + 1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-5)*N_x + j
        positions.append([location-N_x, location -1, location, location + 1, 0])
        
    positions.append([(N_y-5)*N_x-1, (N_y-4)*N_x-2, (N_y-4)*N_x-1, 0, 0])
    
    for i in range(N_y-4, N_y):
        for j in range(N_x):
            positions.append([0, 0, 0, 0, 0])
            
    positions = np.array(positions)
    
    #use the matix of positions to loop over the potential and create the matrix of coefficients
    coefficients = []
    last_cell = int(N_x-1)
    
    #mesh-size is in micrometers!!!
    if method == "e":
        mob_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    elif method == "c":
        mob_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
        
     #first row set to 0 concentration, these are inactive cells
    for j in range(3*N_x):
        coefficients.append([0, 0, 0, 0, 0])
        
    #In the first two rows we use UD
    for i in range(3, N_y-10):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 0]
            W = positions[location, 1]
            P = positions[location, 2]
            E = positions[location, 3]
            N = positions[location, 4]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = - D
                        a_W = 0
                        a_P = a_n + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_n + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = - D
                        a_W = 0
                        a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = 0
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
            
            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_n + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = 0
                        a_W = - D
                        a_P = a_n + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        
            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_n + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_n + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e*E_w < 0:
                        a_N = - D
                        a_S = a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_n + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_s + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e*E_w < 0:
                        a_N = a_n - D
                        a_S = - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_s + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n*E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e*E_w < 0:
                        a_N = a_n/2 - D
                        a_S = a_s/2 - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
    
    #the last two rows contain the inactive p+ implant
    for i in range(N_y-10, N_y-5):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = positions[location, 0]
            W = positions[location, 1]
            P = positions[location, 2]
            E = positions[location, 3]
            N = positions[location, 4]
            
            V_S = potential[S]
            V_W = potential[W]
            V_P = potential[P]
            V_E = potential[E]
            V_N = potential[N]
            
            a_s = (V_P - V_S)*charge_carrier
            a_w = (V_P - V_W)*charge_carrier
            a_e = (V_P - V_E)*charge_carrier
            a_n = (V_P - V_N)*charge_carrier
            
            E_s = -(V_P - V_S)*charge_carrier
            E_n = -(V_N - V_P)*charge_carrier
            E_e = -(V_E - V_P)*charge_carrier
            E_w = -(V_P - V_W)*charge_carrier
            
            if j in range(int((2/5)*N_x), int((3/5)*N_x)):
                coefficients.append([0, 0, 0, 0, 0])
            else:
                if j == 0:
                    E_w = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = - D
                            a_W = 0
                            a_P = a_n + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_n + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = - D
                            a_W = 0
                            a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_s + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = 0
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e - D
                            a_W = 0
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])

                elif j == last_cell:
                    E_e = 0
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_n + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = 0
                            a_W = - D
                            a_P = a_n + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S  = - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = 0
                            a_W = - D
                            a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = a_w - D
                            a_P = a_s/2 + a_n/2 + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = 0
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 3*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])

                else:
                    if E_n >= 0 and E_s >= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_n + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e - D
                            a_W = - D
                            a_P = a_n + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e*E_w < 0:
                            a_N = - D
                            a_S = a_s - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_n + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n <= 0 and E_s <= 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_s + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e - D
                            a_W = - D
                            a_P = a_s + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e*E_w < 0:
                            a_N = a_n - D
                            a_S = - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_s + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_n*E_s < 0:
                        if E_e >= 0 and E_w >= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = - D
                            a_W = a_w - D
                            a_P = a_n/2 + a_s/2 + a_e + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e <= 0 and E_w <= 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2- D
                            a_E = a_e- D
                            a_W = - D
                            a_P = a_s/2 + a_n/2 + a_w + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        elif E_e*E_w < 0:
                            a_N = a_n/2 - D
                            a_S = a_s/2 - D
                            a_E = a_e/2 - D
                            a_W = a_w/2 - D
                            a_P = a_n/2 + a_s/2 + a_e/2 + a_w/2 + mob_fact + 4*D + trap_fact
                            coefficients.append([a_S, a_W, a_P, a_E, a_N])
                            
                            
    for j in range(N_x):
        
        location = (N_y-5)*N_x + j
            
        S = positions[location, 0]
        W = positions[location, 1]
        P = positions[location, 2]
        E = positions[location, 3]
        N = positions[location, 4]

        V_S = potential[S]
        V_W = potential[W]
        V_P = potential[P]
        V_E = potential[E]
        V_N = potential[N]

        a_s = (V_P - V_S)*charge_carrier
        a_w = (V_P - V_W)*charge_carrier
        a_e = (V_P - V_E)*charge_carrier
        a_n = (V_P - V_N)*charge_carrier

        E_s = -(V_P - V_S)*charge_carrier
        E_n = -(V_N - V_P)*charge_carrier
        E_e = -(V_E - V_P)*charge_carrier
        E_w = -(V_P - V_W)*charge_carrier

        if j in range(int((2/5)*N_x), int((3/5)*N_x)):
            coefficients.append([0, 0, 0, 0, 0])
        else:
            E_n = 0
    
            if j == 0:
                E_w = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = - D
                        a_W = 0
                        a_P = a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = - D
                        a_E = - D
                        a_W = 0
                        a_P = a_s + a_e + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e - D
                        a_W = 0
                        a_P = a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])

            elif j == last_cell:
                E_e = 0
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = 0
                        a_W = - D
                        a_P = a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0 
                        a_S = - D
                        a_E = 0
                        a_W = a_w - D
                        a_P = a_s + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = 0
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 2*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])

            else:
                if E_n >= 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e*E_w < 0:
                        a_N = 0
                        a_S = a_s - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_e/2 + a_w/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                elif E_n <= 0 and E_s <= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = - D
                        a_E = - D
                        a_W = a_w - D
                        a_P = a_s + a_e + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e - D
                        a_W = - D
                        a_P = a_s + a_w + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e*E_w < 0:
                        a_N = 0
                        a_S = - D
                        a_E = a_e/2 - D
                        a_W = a_w/2 - D
                        a_P = a_s + a_e/2 + a_w/2 + mob_fact + 3*D + trap_fact
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                             
    for i in range(N_y-4, N_y):
        for j in range(N_x):
            coefficients.append([0, 0, 0, 0, 0])
    
    coefficients = np.array(coefficients)
    
    return coefficients, positions

