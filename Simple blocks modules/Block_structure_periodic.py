#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#For the mathematical details check the log-book

#This module includes the following functions:

#BLOCK CREATORS
#The following function create the matrices of positions and coefficeints for various set-ups

#EMPTY BLOCK
#block_without_electrode_periodic(): this function creates the matrix of position and the matrix of coefficients for the
#empty space above and below the Si detector, it assumes V = 0 BC at infinity
#block_without_electrode_periodic_von_neumann_top(): this function creates the matrix of position and the matrix of coefficients for the
#empty space above and below the Si detector, it assumes E_normal = 0 BC at infinity

#BLOCK WITH DETECTOR
#these function assume that the Si detector occupies the bottom part of the solution domain and that some air is present
#above the Si slab, this means that the bottom boundary of the domain corresponds to the bottom electrode of the detector,
#while the top boundary of the solution domain corresponds to the boundary condition at infinity provided that this is taken
#enough far away
#block_with_electrode_periodic(): this function assumes a dirichelet boundary condition on bottom, V = 0 BC at infinity and
#it does not account for the permitivity discontinuity between air and Si, the strip is identically set to 0 potential by
#inactivating its cells
#block_with_electrode_periodic_modified(): this function is identical to the previous one, but it applies the permitivity
#discontinuity condition between Si and Air
#von_neumann_bottom_top_block_periodic(): this function is analogous to the the previous one but it applies E_normal = 0 BC
#at the bottom and top boundaries of the solution domain

#The previous block creators assumed a p+ implant thickness of 1 micron, and a strip thickness of 0.5 electrodes

#MULTIGRID METHOD
#The functions below assume a p+ implant thickness of 1 micron, and a strip thickness of 1 micron

#These two functions were developed so that they could be used together in a multi-grid solution procedure for the potential
#This approach required to slightly modify the thickness of the strip in order to simplify the definition of the grid
#von_neumann_bottom_block_periodic(): this function works analogously to the previous ones, but it applies E_normal = 0 BC
#only on the bottom electrode
#von_neumann_bottom_block_periodic_Coarse(): this function performs the same operations as the function above but on a mesh
#that is double the size

#von_neumann_bottom_block_periodic_inactivated() and von_neumann_bottom_block_periodic_Coarse_inactivated(): these two
#functions work like the prevous two ones but they enforce cell inactivation of the implant region, this way it is possible
#to set the p+ implant to the required potential

#SOURCE CREATOR
#update_bottom_top_source(): in a multi-block parallelised approach this function is used to update the transition
#cells between a block that is "sandwiched" between two other blocks (typically empty blocks)


# In[ ]:


#BLOCK CREATORS VARIABLES
#pitch, height, mesh_size: physical dimensions of the block
#pitch is the horizontal dimension, we assume that the strip (axis of symmetry) is placed at the centre of the set-up

#default mesh size for block with detector = 0.5
#default mesh size for block without detector = 1 (this is to speed up the process of iterating over such cells)


# In[ ]:


#SOURCE CREATOR VARIABLES
#b: old source
#B_B, B_T: transition cells on top and bottom of the block of interest

#The function return the updated version of the source


# In[ ]:


#This code implements the usual numbering convention for the cells, bottom-top, left-right

#The strip electrode is defined to be located within 2/5 and 3/5 of the horizontal direction

#The location of the Si-Air interface can be adjusted by changing the value of a fraction, the fraction will depend
#also on the total height of the domain

#range((2/5)*N_x, (3/5)*N_x) defines the location of the electrode in the x direction
#fraction*N_y defines the location in the y-direction
#conditional statements are used to establish the type of cells according to their fullfillment of any such conditions


# In[9]:


def block_without_electrode_periodic(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0]
    bottom = [0, 1, -4, 1, 1]
    
    #bottom layer
    for j in range(N_x):
        coefficients.append(bottom)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


def block_without_electrode_periodic_von_neumann_top(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -3, 1, 0] #neumann E_normal = 0 condition
    bottom = [0, 1, -4, 1, 1]
    
    #bottom layer
    for j in range(N_x):
        coefficients.append(bottom)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


def block_with_electrode_periodic(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size) #Number of cells per side
    N_y = int(height/mesh_size)
    
    #Define the electrode location
    start_electrode = int((4/5)*N_y*N_x + (2/5)*N_x)
    end_electrode = int((4/5)*N_y*N_x + (3/5)*N_x)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0] #Dirichet BC
    bottom = [0, 1, -4, 1, 1] #Dirichet BC
    
    #bottom layer
    for j in range(N_x):
        coefficients.append(bottom)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
    
    #define the electrode
    for i in range(start_electrode, end_electrode):
        coefficients[i] = [0, 0, 0, 0, 0]
    #due to the definition of the SOR solver these cells will not be updated (they will be left at the electrode potential)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


def block_with_electrode_periodic_modified(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size) #numer of cells per side
    N_y = int(height/mesh_size)
    
    #Define the electrode location
    start_electrode = int((3/4)*N_y*N_x + (2/5)*N_x)
    end_electrode = int((3/4)*N_y*N_x + (3/5)*N_x)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0] #Dirichet BC
    bottom = [0, 1, -4, 1, 1] #Dirichet BC
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2] #transition between Si and Air
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1] #transition between Air and Si
    
    #bottom layer
    for j in range(N_x):
        coefficients.append(bottom)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((3/4)*N_y - 1): #adjust the fraction to define the location of the Si-Air interface in terms of cell number
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                else:
                    coefficients.append(bulk)
            
            elif i == int((3/4)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
    
    #define the electrode
    for i in range(start_electrode, end_electrode):
        coefficients[i] = [0, 0, 0, 0, 0]
    #due to the definition of the SOR solver these cells will not be updated (they will be left at the electrode potential)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


def von_neumann_bottom_top_block_periodic(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -3, 1, 0] #E_normal = 0 BC
    bottom = [0, 1, -3, 1, 1] #E_normal = 0 BC
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2] #transition between Si and Air
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1] #transition between Air and Si
    
    for j in range(N_x):
        coefficients.append(bottom)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((1/3)*N_y - 1):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                else:
                    coefficients.append(bulk)
            
            elif i == int((1/3)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
            
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[2]:


#FINE MESH
#This version inactivates only the strip cells

#Fine mesh = 0.5

def von_neumann_bottom_block_periodic(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0]
    bottom_dir = [0, 1, -3, 1, 1]
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2]
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1]
    
    for j in range(N_x):
        coefficients.append(bottom_dir)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((1/2)*N_y - 1):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([1, 1, -3, 1, 0])
                else:
                    coefficients.append(bulk)
            
            elif i == int((1/2)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])
                    
            elif i == int((1/2)*N_y + 1) and j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0]) #here the electrode is of thickness 1, so a further group of cells
                                                         #must be deactivated

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[1]:


#COARSE MESH
#This version inactivates only the strip cells

#Coarse mesh = 1

def von_neumann_bottom_block_periodic_Coarse(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0]
    bottom_dir = [0, 1, -3, 1, 1]
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2]
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1]
    
    for j in range(N_x):
        coefficients.append(bottom_dir)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((1/2)*N_y - 1):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([1, 1, -3, 1, 0])
                else:
                    coefficients.append(bulk)
            
            elif i == int((1/2)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


#FINE MESH
#This version inactivates both the strip cells and the implant cells

#Fine mesh = 0.5

def von_neumann_bottom_block_periodic_inactivated(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0]
    bottom_dir = [0, 1, -3, 1, 1]
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2]
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1]
    
    for j in range(N_x):
        coefficients.append(bottom_dir)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((1/2)*N_y - 2) and j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0]) #here the p+ implant is deactivated
            
            elif i == int((1/2)*N_y - 1):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0]) #p+
            
            elif i == int((1/2)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0]) #electrode
                    
            elif i == int((1/2)*N_y + 1) and j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0]) #electrode

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[ ]:


#COARSE MESH
#This version inactivates both the strip cells and the implant cells

#Coarse mesh = 1

def von_neumann_bottom_block_periodic_Coarse_inactivated(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    coefficients = []
    
    rel_perm = 11.7
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    top = [1, 1, -4, 1, 0]
    bottom_dir = [0, 1, -3, 1, 1]
    trans_Si_air = [1, 1, -(3 + (1 + 1/rel_perm)/2), 1, (1 + 1/rel_perm)/2]
    trans_air_Si = [(1 + rel_perm)/2, 1, -(3 + (1 + rel_perm)/2), 1, 1]
    
    for j in range(N_x):
        coefficients.append(bottom_dir)
        
    #bulk
    for i in range(1, N_y-1):
        for j in range(N_x):
            
            if i == int((1/2)*N_y - 1):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_Si_air)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_Si_air)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])
                else:
                    coefficients.append(bulk)
            
            elif i == int((1/2)*N_y):
                if j in range(0, int((2/5)*N_x)):
                    coefficients.append(trans_air_Si)
                elif j in range(int((3/5)*N_x), N_x):
                    coefficients.append(trans_air_Si)
                elif j in range(int((2/5)*N_x), int((3/5)*N_x)):
                    coefficients.append([0, 0, 0, 0, 0])

            else:
                coefficients.append(bulk)
            
    #top layer
    for j in range(N_x):
        coefficients.append(top)
        
    #MATRIX OF POSITIONS
    
    positions = []
    
    #bottom layer
    positions.append([0, N_x-1, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
    
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x-1])
    
    #bulk
    for i in range(1, N_y-1):
        positions.append([(i-1)*N_x, (i+1)*N_x-1, i*N_x, i*N_x+1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            location = i*N_x + j
            positions.append([location - N_x, location - 1, location, location + 1, location + N_x])
        
        positions.append([i*N_x-1, (i+1)*N_x-2, (i+1)*N_x-1, i*N_x, (i+2)*N_x-1])
        
    #top layer
    positions.append([(N_y-2)*N_x, N_y*N_x-1, (N_y-1)*N_x, (N_y-1)*N_x+1, 0])
    
    for j in range(1, N_x-1):
        location = (N_y-1)*N_x + j
        positions.append([location - N_x, location - 1, location, location + 1, 0])
    
    positions.append([(N_y-1)*N_x - 1, N_y*N_x-2, N_y*N_x-1, (N_y-1)*N_x, 0])
    
    return coefficients, positions


# In[20]:


#UPDATE THE SOURCE TERM
#THE ENTRIES THAT MUST BE UPDATED ARE THE ONES FOR TOP AND BOTTOM CELLS
#The possible cases that are contempleted by this function are the transition between blocks of the same size,
#and the transition betweeen blocks which have mesh size one half of the other
#the typical situation is 0.5 and 1

#This function allows to update the source term only at the boundary locations, the source is in fact fixed by the charge
#density of the Si detector

def update_bottom_top_source(b, pitch, height, mesh_size, B_B, B_T):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    N_cells = int(N_x*N_y)
    
    #up
    
    if len(B_B) == N_x:
        for i in range(N_x):
            b[i] = -B_B[i]
    elif len(B_B) == int(N_x/2):
        for i in range(int(N_x/2)):
            location = int(2*i)
            b[location] = -B_B[i]
            b[location + 1] = -B_B[i]
    elif len(B_B) == int(2*N_x):
        for i in range(0, 2*N_x, 2):
            location = int(i/2)
            b[location] = -(B_B[i]+B_B[i+1])/2
    
    if len(B_T) == N_x:
        for i, j in zip(range(N_cells - N_x, N_cells), range(N_x)):
            b[i] = -B_T[j]
    elif len(B_T) == int(N_x/2):
        vec = []
        for k in range(int(N_x/2)):
            vec.append(k)
            vec.append(k)
        for i, j in zip(range(N_cells - N_x, N_cells), vec):
            b[i] = -B_T[j]
    elif len(B_T) == int(2*N_x):
        for i, j in zip(range(N_cells - N_x, N_cells), range(0, 2*N_x, 2)):
            b[i] = -(B_T[j]+B_T[j+1])/2
            
    return b

