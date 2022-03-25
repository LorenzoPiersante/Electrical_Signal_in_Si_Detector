#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This function creates the matrix of coefficients and the matrix of positions for the diffusion problem in the presence of 
#trapping

#The featues of the solution domain are the same as usual

#For the mathematical detail required to explain the from of the equation refer to the logbook


# In[ ]:


#Here as usual the function sets the entire p+ implant to 0 concentration


# In[ ]:


#The function uses a central differencing scheme for the diffusion problem, while the time differencing sheme can be selected


# In[ ]:


#MATRIX CREATOR VARIABLES

#height and pitch: dimensions of the region of the Si detector we are interested in
#height is the thickness of the detector, pitch is the separation between the centres of neighbouring strips
#the functions below locate the electrode/strip at the centre of the solution domain and takes the total width of the domain
#to be the pitch (i.e. the strip is located at pitch/2)

#mobility: mobility of electrons or holes
#time_step: size of time advancement specified in seconds
#time_const: this is the effective time constant that describes trapping

#charge_carrier: +1 for hole, -1 for electron, it determines the direction of the drift

#method: "c" to create a matrix based on CN time differencing, "e" for Euler-impicit time differencing


# In[ ]:


import numpy as np

#the time const must be expressed in seconds
#mobility must be expressed in m^2 V^-1 s^-1

def block_structure_diff_trap(height, pitch, mobility, time_const, time_step, method):
    
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
    positions.append([0, 0, 0, 0, 1, N_x, 0])
    
    for j in range(1, N_x-1):
        location = j
        positions.append([0, 0, location-1, location, location + 1, location + N_x, 0])
        
    #bottom-right corner
    
    positions.append([0, 0, N_x - 2, N_x - 1, 0, 2*N_x - 1, 0])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([0, (i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x, 0])
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([0, location - N_x, location-1, location, location + 1, location + N_x, 0])
        positions.append([0, i*N_x -1, (i+1)*N_x -2, (i+1)*N_x -1, 0, (i+2)*N_x - 1, 0])
    
    #last row
    positions.append([0, (N_y-2)*N_x, 0, (N_y-1)*N_x, (N_y-1)*N_x + 1, 0, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([0, location-N_x, location -1, location, location + 1, 0, 0])
        
    positions.append([0, (N_y-1)*N_x-1, N_y*N_x-2, N_y*N_x-1, 0, 0, 0])
            
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
    for j in range(N_x):
        coefficients.append([0, 0, 0, 0, 0])
    
    #bulk cells
    for i in range(1, N_y-10):
        for j in range(N_x):
            
            if j == 0:
                coefficients.append([0, -D, 0, 3*D + trap_fact + mob_fact, -D, -D, 0])
            
            elif j == last_cell:
                coefficients.append([0, -D, -D, 3*D + trap_fact + mob_fact, 0, -D, 0])
                
            else:
                coefficients.append([0, -D, -D, 4*D + trap_fact + mob_fact, -D, -D, 0])

    
    #define the inactive implant
    for i in range(N_y-10, N_y-1):
        for j in range(N_x):
            
            if j in range(int((2/5)*N_x), int((3/5)*N_x)):
                coefficients.append([0, 0, 0, 0, 0, 0, 0])
            
            else:
                if j == 0:
                    coefficients.append([0, -D, 0, 3*D + trap_fact + mob_fact, -D, -D, 0])
            
                elif j == last_cell:
                    coefficients.append([0, -D, -D, 3*D + trap_fact + mob_fact, 0, -D, 0])
                
                else:
                    coefficients.append([0, -D, -D, 4*D + trap_fact + mob_fact, -D, -D, 0])

    #no flux on top
    for j in range(N_x):
        
        if j in range(int((2/5)*N_x), int((3/5)*N_x)):
            coefficients.append([0, 0, 0, 0, 0, 0, 0])
        
        else:   
            if j == 0:
                coefficients.append([0, -D, 0, 2*D + trap_fact + mob_fact, -D, 0, 0])

            elif j == last_cell:
                coefficients.append([0, -D, -D, 2*D + trap_fact + mob_fact, 0, 0, 0])

            else:
                coefficients.append([0, -D, -D, 3*D + trap_fact + mob_fact, -D, 0, 0])
        
    
    coefficients = np.array(coefficients)
    
    return coefficients, positions

