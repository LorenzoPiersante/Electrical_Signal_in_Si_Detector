#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


#The form of the equations is sorted out in the paper notes
#This is important because this will determine the form of the coeffiecnts and the parameters that will enter the sourcces

#For these information you must check the theory on the log-book


# In[ ]:


#This module contains some general function needed for both the drift and diffusion problem in the Si detector
#It also contains some simple implementations of the FVM procedure that were used for validation

#SOURCES

#drift_source(): this function creates the source for the transient problem according to a UD spatial differencing scheme
#and a CN time differencing scheme
#drift_source_Euler_implicit(): this function creates the source for the transient problem according to a UD spatial differencing scheme
#and a Euler implicit time differencing scheme
#QUICK_drift_source_CN(): this function creates the source for the transient problem according to a QUICK spatial differencing scheme
#and a CN time differencing scheme
#QUICK_drift_source_Euler_impl(): this function creates the source for the transient problem according to a QUICK spatial differencing scheme
#and a Euler implicit time differencing scheme

#QUICK_drift_source_CN_optimised(): this function is identical to the homonimous function with the difference that is possible
#to specify the cells over which the update will occur using the start_end variable
#drift_source_CN_optimised(): SAME

#MATRIX CREATORS

#5-diagonals
#drift_parallel_plate(): this function creates the matrix of positions and the matirx of coefficients for a parallel plate
#capacitor set-up. It uses UD in the x-dir and LUD in the y-dir

#7-diagonals
#QUICK_drift_parallel_plate(): this function creates the matrix of positions and the matirx of coefficients for a parallel plate
#capacitor set-up. It uses UD in the x-dir and QUICK in the y-dir
#QUICK_drift_parallel_plate_UD_part(): this function creates a matrix of coefficients composed of the UD coefficients of the QUICK
#differencing scheme, the mattrix is essentially 5-diagonlas, but for ease of implementation it is kept 7-diagonals setting the
#SS and NN entries to 0
#QUICK_drift_parallel_plate_correction_part(): this function creates a matrix of coefficients containing the 2nd order parts of
#QUICK coefficients, they are used in iterative solution methods as deferred correction terms in the source

#IMPORTANT NOTE ON THE BOUNDARIES
#due to the periodicity of the electric field, there will be no horizontal component of the electric field at the domain
#boundary. This implies that if only drift (convective motion) is considered, the solution domains of two neighbouring strips
#effectively don't communicate. Therefore, no flux BC on the left and right boundaries can be assumed, which means that the
#matrix coefficients will completely determine the time evolution of the system.
#The source does not require additional term coming from transition cells.


# In[ ]:


#SOURCE VARIABLES

#height and pitch: these are dimension data of the region of interest of the Si detector, the default mesh size is 0.1
#coefficients: matrix of coefficients
#positions: matrix of positions, it specifies the neighbouring cells for each cell and encodes the problem geometry
#concentration: chsarge distribution at the previous time step
#mobility: mobility of electrons or holes
#time_step: size of time advancement specified in seconds

#The above are the general variables, depending on the specific features of the method only some of them or some exra variables
#might be required

#SPACE DIFFERENCING FEATURES

#UD based functions involve 5-diagonal matrices
#QUICK based methods involve 7-diagonal matrices

#TIME DIFFERENCING FEATURES

#Euler-implicit based methods don't use information about the matrix of coefficeints, so the only required system inputs are the
#concentration at the previous time step, the time step size and the mobility of the species

#CN based methods require information about the coeffients since time-advancement is determined through a mid-point rule


# In[2]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def drift_source(height, pitch, coefficients, positions, concentration, mobility, time_step):
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = []
    
    for i in range(N_y):
        for j in range(N_x):
            
            location = i*N_x + j
            
            S = int(positions[location, 0])
            W = int(positions[location, 1])
            P = int(positions[location, 2])
            E = int(positions[location, 3])
            N = int(positions[location, 4])
            
            a_S = coefficients[location, 0]
            a_W = coefficients[location, 1]
            a_P = coefficients[location, 2]
            a_E = coefficients[location, 3]
            a_N = coefficients[location, 4]
            
            c_S = concentration[S]
            c_W = concentration[W]
            c_P = concentration[P]
            c_E = concentration[E]
            c_N = concentration[N]
            
            source_term = -a_N*c_N - a_S*c_S - a_P*c_P - a_E*c_E - a_W*c_W + 2*mobility_fact*c_P
            
            source.append(source_term)
            
    return source


# In[3]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def drift_source_Euler_implicit(height, pitch, concentration, mobility, time_step):
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = []
    
    for i in range(N_y):
        for j in range(N_x):
            
            location = i*N_x + j
            
            c_P = concentration[location]
            
            source_term = mobility_fact*c_P
            
            source.append(source_term)
            
    return source


# In[5]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def QUICK_drift_source_CN(height, pitch, coefficients, positions, concentration, mobility, time_step):
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = []
     
    for i in range(N_y):
        for j in range(N_x):
            
            location = i*N_x + j
        
            SS = int(positions[location, 0])
            S = int(positions[location, 1])
            W = int(positions[location, 2])
            P = int(positions[location, 3])
            E = int(positions[location, 4])
            N = int(positions[location, 5])
            NN = int(positions[location, 6])

            a_SS = coefficients[location, 0]
            a_S = coefficients[location, 1]
            a_W = coefficients[location, 2]
            a_P = coefficients[location, 3]
            a_E = coefficients[location, 4]
            a_N = coefficients[location, 5]
            a_NN = coefficients[location, 6]

            c_SS = concentration[SS]
            c_S = concentration[S]
            c_W = concentration[W]
            c_P = concentration[P]
            c_E = concentration[E]
            c_N = concentration[N]
            c_NN = concentration[NN]

            source_term = -a_N*c_N - a_S*c_S - a_P*c_P - a_E*c_E - a_W*c_W - a_SS*c_SS - a_NN*c_NN + 2*mobility_fact*c_P

            source.append(source_term)
            
    return source


# In[7]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s


def QUICK_drift_source_Euler_impl(height, pitch, concentration, mobility, time_step):
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = []
    
    for i in range(N_y):
        for j in range(N_x):
            
            location = i*N_x + j
            
            c_P = concentration[location]
            
            source_term = mobility_fact*c_P
            
            source.append(source_term)
            
    return source


# In[6]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def QUICK_drift_source_CN_optimised(height, pitch, coefficients, positions, start_end, concentration, mobility, time_step):
    
    start_cell, end_cell = start_end
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = np.zeros(N_x*N_y)
     
    for location in range(start_cell, end_cell):
        
        SS = int(positions[location, 0])
        S = int(positions[location, 1])
        W = int(positions[location, 2])
        P = int(positions[location, 3])
        E = int(positions[location, 4])
        N = int(positions[location, 5])
        NN = int(positions[location, 6])

        a_SS = coefficients[location, 0]
        a_S = coefficients[location, 1]
        a_W = coefficients[location, 2]
        a_P = coefficients[location, 3]
        a_E = coefficients[location, 4]
        a_N = coefficients[location, 5]
        a_NN = coefficients[location, 6]

        c_SS = concentration[SS]
        c_S = concentration[S]
        c_W = concentration[W]
        c_P = concentration[P]
        c_E = concentration[E]
        c_N = concentration[N]
        c_NN = concentration[NN]

        source_term = -a_N*c_N - a_S*c_S - a_P*c_P - a_E*c_E - a_W*c_W - a_SS*c_SS - a_NN*c_NN + 2*mobility_fact*c_P

        source[location] = source_term
            
    return source


# In[ ]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def drift_source_CN_optimised(height, pitch, coefficients, positions, start_end, concentration, mobility, time_step):
    
    start_cell, end_cell = start_end
    
    mesh_size = 0.1
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    mobility_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    source = np.zeros(N_x*N_y)
     
    for location in range(start_cell, end_cell):
        
        S = int(positions[location, 0])
        W = int(positions[location, 1])
        P = int(positions[location, 2])
        E = int(positions[location, 3])
        N = int(positions[location, 4])

        a_S = coefficients[location, 0]
        a_W = coefficients[location, 1]
        a_P = coefficients[location, 2]
        a_E = coefficients[location, 3]
        a_N = coefficients[location, 4]
        
        c_S = concentration[S]
        c_W = concentration[W]
        c_P = concentration[P]
        c_E = concentration[E]
        c_N = concentration[N]
        
        source_term = -a_N*c_N - a_S*c_S - a_P*c_P - a_E*c_E - a_W*c_W + 2*mobility_fact*c_P
        
        source[location] = source_term
            
    return source


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

#charge_carrier: +1 for hole, -1 for electron, it determines the direction of the drift

#method: "c" to create a matrix based on CN time differencing, "e" for Euler-impicit time differencing

#The variables are the same for all functions, yet they are processed in different ways and the putput differs as explained in
#the brief description


# In[4]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def drift_parallel_plate(height, pitch, potential, mobility, charge_carrier, time_step, method):
    
    #The position part is the same
    mesh_size = 0.1
    
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
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x])
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location-1, location, location + 1, location + N_x])
        positions.append([i*N_x -1, (i+1)*N_x -2, (i+1)*N_x -1, 0, (i+2)*N_x - 1])
        
    #top-left corner
    positions.append([(N_y-2)*N_x, 0, (N_y-1)*N_x, (N_y-1)*N_x + 1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location -1, location, location + 1, 0])
    
    #top-right corner
    positions.append([(N_y-1)*N_x - 1, N_y*N_x - 2, N_y*N_x - 1, 0, 0])
    
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
    
    #create all cells except for the top row
    for i in range(1, N_y-1):
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
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = 0
                        a_W = 0
                        a_P = a_n + a_e + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = a_e
                        a_W = 0
                        a_P = a_n + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_P = a_s + a_e + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w <= 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = a_e
                        a_W = 0
                        a_P = a_s + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
            
            elif j == last_cell:
                E_e = 0
                if E_n > 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = 0
                        a_W = a_w
                        a_P = a_n + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w < 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = 0
                        a_W = 0
                        a_P = a_n + a_w + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                if E_n <= 0 and E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = 0
                        a_W = a_w
                        a_P = a_s + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w < 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_P = a_s + a_w + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
            
            else:
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = 0
                        a_W = a_w
                        a_P = a_n + a_e + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w < 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = a_e
                        a_W = 0
                        a_P = a_n + a_w + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e == 0 and E_w == 0:
                        a_N = 0
                        a_S = a_s - a_n/2
                        a_E = 0
                        a_W = 0
                        a_P = a_n + mob_fact + a_n/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = 0
                        a_W = a_w
                        a_P = a_s + a_e + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e <= 0 and E_w < 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = a_e
                        a_W = 0
                        a_P = a_s + a_w + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                    elif E_e == 0 and E_w == 0:
                        a_N = a_n - a_s/2
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_P = a_s + mob_fact + a_s/2
                        coefficients.append([a_S, a_W, a_P, a_E, a_N])
                        
    #top row of inactive cells
    for j in range(N_x):
        coefficients.append([0, 0, 0, 0, 0])
        
    coefficients = np.array(coefficients)
    

    return coefficients, positions


# In[9]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def QUICK_drift_parallel_plate(height, pitch, potential, mobility, charge_carrier, time_step, method):
    
    #The position part is the same
    mesh_size = 0.1
    
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
    
    positions.append([0, 0, N_x - 2, N_x - 1, 0, 2*N_x - 1, 2*N_x - 1])
    
    #first row
    #left corner
    positions.append([0, 0, 0, N_x, N_x + 1, 2*N_x, 3*N_x])
    
    for j in range(1, N_x-1):
        location = N_x + j
        positions.append([0, location - N_x, location-1, location, location + 1, location + N_x, location + 2*N_x])
        
    #right corner
    positions.append([0, N_x-1, 2*N_x - 2, 2*N_x - 1, 0, 3*N_x - 1, 4*N_x - 1])
    
    
    #bulk
    for i in range(2, N_y-2):
        positions.append([(i-2)*N_x, (i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x, (i+2)*N_x])
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - 2*N_x, location - N_x, location-1, location, location + 1, location + N_x, location + 2*N_x])
        positions.append([(i-1)*N_x - 1, i*N_x -1, (i+1)*N_x -2, (i+1)*N_x -1, 0, (i+2)*N_x - 1, (i+3)*N_x - 1])
    
    #penultimate row
    positions.append([(N_y-4)*N_x, (N_y-3)*N_x, 0, (N_y-2)*N_x, (N_y-2)*N_x + 1, (N_y-1)*N_x, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-2)*N_x + j
        positions.append([location - 2*N_x, location - N_x, location -1, location, location + 1, location + N_x, 0])
        
    positions.append([(N_y-3)*N_x - 1, (N_y-2)*N_x - 1, (N_y-1)*N_x - 2, (N_y-1)*N_x - 1, 0, N_y*N_x - 1, 0])
    
    #top-left corner
    positions.append([(N_y-3)*N_x, (N_y-2)*N_x, 0, (N_y-1)*N_x, (N_y-1)*N_x + 1, 0, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - 2*N_x, location - N_x, location -1, location, location + 1, 0, 0])
    
    #top-right corner
    positions.append([(N_y-2)*N_x - 1, (N_y-1)*N_x - 1, N_y*N_x - 2, N_y*N_x - 1, 0, 0, 0])
    
    positions = np.array(positions)
    
    #use the matix of positions to loop over the potential and create the matrix of coefficients
    coefficients = []
    last_cell = int(N_x-1)
    
    #mesh-size is in micrometers!!!
    if method == "e":
        mob_fact = ((mesh_size*10**(-6))**2)/(mobility*time_step)
    elif method == "c":
        mob_fact = (2*(mesh_size*10**(-6))**2)/(mobility*time_step)
    
    #set first two rows to 0 coeff, these are inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    #bulk
    for i in range(2, N_y-2):
        for j in range(N_x):
            
            location = i*N_x + j
            
            #N.B. now the 0th and last column are SS and NN, so you must pick the new right positions!
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
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = a_e
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            elif j == last_cell:
                E_e = 0
                if E_n > 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = 0
                        a_W = a_w
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = a_w
                        a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            else:
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = 0
                        a_W = a_w
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = a_e
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = a_s - a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = a_n + (3/8)*a_s - a_n/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = a_w
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = -a_n/8
                        a_N = a_n - a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + (3/8)*a_n - a_s/4 + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
    
    #top two rows of inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
        
    coefficients = np.array(coefficients)
    

    return coefficients, positions


# In[10]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def QUICK_drift_parallel_plate_UD_part(height, pitch, positions, potential, mobility, charge_carrier, time_step, method):
    
    #The position part is the same
    mesh_size = 0.1
    
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
    
    #set first two rows to 0 coeff, these are inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    #bulk
    for i in range(2, N_y-2):
        for j in range(N_x):
            
            location = i*N_x + j
            
            #N.B. now the 0th and last column are SS and NN, so you must pick the new right positions!
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
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_n + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_n + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = a_n
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = a_n 
                        a_S = 0
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            elif j == last_cell:
                E_e = 0
                if E_n > 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s 
                        a_E = 0
                        a_W = a_w
                        a_SS = 0
                        a_P = a_n + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_n + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = a_n 
                        a_S = 0
                        a_E = 0
                        a_W = a_w
                        a_P = a_s + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = a_n 
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            else:
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = 0
                        a_W = a_w
                        a_SS = 0
                        a_P = a_n + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_n + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = 0
                        a_N = 0
                        a_S = a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_n + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = a_n 
                        a_S = 0
                        a_E = 0
                        a_W = a_w
                        a_SS = 0
                        a_P = a_s + a_e + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = a_n 
                        a_S = 0
                        a_E = a_e
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + a_w + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = 0
                        a_N = a_n
                        a_S = 0
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = a_s + mob_fact
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
    
    #top two rows of inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
        
    coefficients = np.array(coefficients)
    

    return coefficients


# In[11]:


#mobility must be expressed in m^2 s^-1 V^-1
#time step in s

def QUICK_drift_parallel_plate_correction_part(height, pitch, positions, potential, mobility, charge_carrier, time_step, method):
    
    #The position part is the same
    mesh_size = 0.1
    
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
    
    #set first two rows to 0 coeff, these are inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
    
    #bulk
    for i in range(2, N_y-2):
        for j in range(N_x):
            
            location = i*N_x + j
            
            #N.B. now the 0th and last column are SS and NN, so you must pick the new right positions!
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
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w <= 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            elif j == last_cell:
                E_e = 0
                if E_n > 0 and E_s >= 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e >= 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
            
            else:
                if E_n > 0 and E_s >= 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = 0
                        a_N = (3/8)*a_n
                        a_S = -a_s/4 - a_n/8
                        a_E = 0
                        a_W = 0
                        a_SS = -a_s/8
                        a_P = (3/8)*a_s - a_n/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                if E_n <= 0 and E_s < 0:
                    if E_e > 0 and E_w >= 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e <= 0 and E_w < 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
                    elif E_e == 0 and E_w == 0:
                        a_NN = -a_n/8
                        a_N = -a_n/4 - a_s/8
                        a_S = (3/8)*a_s
                        a_E = 0
                        a_W = 0
                        a_SS = 0
                        a_P = (3/8)*a_n - a_s/4
                        coefficients.append([a_SS, a_S, a_W, a_P, a_E, a_N, a_NN])
    
    #top two rows of inactive cells
    for j in range(2*N_x):
        coefficients.append([0, 0, 0, 0, 0, 0, 0])
        
    coefficients = np.array(coefficients)
    

    return coefficients

