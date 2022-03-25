#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CHECK THE DOCUMENTATION FILE FOR MORE INFORMATION
#16/01/22

#This module includes 4 functions:
#block_with_segment(): creates the matrix of coefficients and the matrix of positions for the block with an electrode
#block_without_segment(): creates the matrix of coefficients and the matrix of positions for an empty block
#block_with_segment_source(): creates the source term of the system of linear equations for the block with electrode
#block_without_segment_source(): creates the source term of the system of linear equations for the empty block

#these are not fully flexible functions, they are designed to be optimised for a specific mesh size
#WITH ELECTRODE: mesh size = 0.5 microns
#WITHOUT ELECTRODE: mesh size = 1 microns

#the data type of the output is a list type
#the matrix of coefficients and positions must them by converted to np.array() in order to be used as arrays with indeces
#[i, j]


# In[ ]:


#GENERAL VARIABLES

#MATRIX CREATORS
#pitch, height, mesh_size: the physical dimensions of the block, i.e. detector + some air above + grid size

#SOURCE CREATORS
#pitch, height, mesh_size
#B_L, B_R, B_B, B_T: arrays of transition cells between two blocks, we have the transition cells between two adjacent blocks in
#the horizontal direction (B_L and B_R), and between two blocks in the vertical direction (B_B or B_T depending on the
#]type of block)


# In[32]:


#BLOCK WITH ELECTRODE
#assumed electrode thickness: 0.5 microns
#we are defining the electrode to be 1/5 of the pitch

#the height of the domain must be 3 times the height of the electrode
#the function is designed so that it positions an electrode at 1/3 of the height of the domain and at the centre of
#the domain width

#THIS FUNCTION DEFINES THE MATRIX OF COEFFICIENTS AND THE MATRIX OF LOCATIONS FOR THE CELLS IN THE BLOCK
#The function takes the pitch (thickness of the block, pitch of the microstrip detector), height of the domain, mesh size
#typical values 100, 900 (so that electrode is at 300), 0.5

def block_with_segment(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #define the electrode first and last cell (since thickness is 0.5 like the mesh only one row must be removed)
    start_electrode = int((N_x*N_y)/3 + N_x*(2/5))
    end_electrode = int((N_x*N_y)/3 + N_x*(3/5))
    
    #system has V = 0 boundary condition at the bottom
    #L and R the BC is given by the voltage in the other blocks
    #T the BC is given by the voltage in the top block
    
    #DEFINE THE MATRIX OF COEFFICIENTS - coefficients of the system of linear equations
    
    coefficients = []
    
    #type of cells
    left_B = [1, 0, -4, 1, 1]
    right_B = [1, 1, -4, 0, 1]
    bulk = [1, 1, -4, 1, 1]
    top_left_corner = [1, 0, -4, 1, 0]
    top_right_corner = [1, 1, -4, 0, 0]
    top_B = [1, 1, -4, 1, 0]
    
    #define most of the block except for the top layer
    #TOP IS BOUNDARY WITH OTHER BLOCK
    for i in range(N_y-1):
        
        coefficients.append(left_B)
        
        for j in range(N_x-2):
            coefficients.append(bulk)
            
        coefficients.append(right_B)
        
    #define the top layer
    coefficients.append(top_left_corner)
    
    for j in range(N_x-2):
        coefficients.append(top_B)
            
    coefficients.append(top_right_corner)
    
    #define the electrode by setting the coefficients corresponding to it to 0
    #i.e. the coefficients attached to the cells that make up the electrode
    
    for i in range(start_electrode, end_electrode):
        coefficients[i] = [0, 0, 0, 0, 0]
    #since the source is set to 0 at these locations the equation is satisfied automatically
    
    #eliminate bottom layer that is not iterated over since it is fixed to 0
    del coefficients[0 : N_x]
    
    #NOW THE MARIX OF COEFFICIENTS IS DEFINED
    
    #DEFINE THE MATRIX OF POSITIONS - the positions of the cells surrounding a given cell
    
    positions = []
    
    #define most of the positions except for the top layer
    for i in range(N_y-1):
        
        positions.append([(i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x])
        
        for j in range(1, N_x - 1):
            positions.append([(i-1)*N_x + j, i*N_x - 1 + j, i*N_x + j, i*N_x + 1 + j, (i+1)*N_x + j])
            
        positions.append([i*N_x - 1 , (i+1)*N_x - 2, (i+1)*N_x - 1, 0, (i+2)*N_x - 1])
        
    #define the top layer
    positions.append([(N_y-2)*N_x, 0, (N_y-1)*N_x, (N_y-1)*N_x + 1, 0])
    
    for j in range(1, N_x - 1):
        positions.append([(N_y-2)*N_x + j, (N_y-1)*N_x - 1 + j, (N_y-1)*N_x + j, (N_y-1)*N_x + 1 + j, 0])
            
    positions.append([(N_y-1)*N_x - 1 , N_y*N_x - 2, N_y*N_x - 1, 0, 0])
    
    #eliminate bottom layer
    del positions[0 : N_x]
    
    #NOW THE MATRIX OF POSITIONS IS DEFINED
    
    #DEFINE THE SOURCE VECTOR
    
    return coefficients, positions


# In[15]:


#BLOCK WITH ELECTRODE
#THIS FUNCTION TAKES THE TRANSITION CELLS FOR THE LEFT, RIGHT AND TOP BLOCKS AND PRODUCES THE SOURCE VECTOR FOR THE GIVEN
#BLOCK
#Pitch, height and mesh size are needed to produce vectors of the right size

#ATTENTION
#B_R AND B_L WILL BE OF SIZE 0.5 MICRONS IN SIDE, VERTICAL SLICES
#B_T WILL BE OF SIZE 1 MICRONS IN SIDE, HORIZONTAL SLICE

#B_R AND B_L WILL BE OF SIZE N_y - 1 (exclude the bottom that is 0 by set-up), must be very careful in its definition
#B_B WILL BE OF SIZE N_x/2 (since the mesh size is double for the top electrode)

def block_with_segment_source(pitch, height, mesh_size, B_L, B_R, B_B):

    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    source = []
    
    #the source will have the number of active cells in the block
    for i in range(N_y - 1):
        source.append(-B_L[i])
        
        for j in range(N_x - 2):
            source.append(0)
            
        source.append(-B_R[i])
    #in this loop we have added the sources for left and right
    
    #source for top
    #B_T has half the cells  
    position_B_B = []
    for n in range(0, int(N_x/2)):
        position_B_B.append(n)
        position_B_B.append(n)
    
    #add the contribution from the top
    for j, k in zip(range(N_x), position_B_B):
        location = (N_y-2)*N_x + j
        source[location] = source[location] - B_B[k]
    
    return source


# In[17]:


#BLOCK WITHOUT ELECTRODE

#THIS FUNCTION DEFINES THE MATRIX OF COEFFICIENTS AND THE MATRIX OF LOCATIONS FOR THE CELLS IN THE BLOCK
#The function takes the pitch (thickness of the block, pitch of the microstrip detector), height of the domain, mesh size
#typical values 100, 900, 1

def block_without_segment(pitch, height, mesh_size):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    #DEFINE THE MATRIX OF COEFFICIENTS
    
    #types of cells
    bulk = [1, 1, -4, 1, 1]
    left_B = [1, 0, -4, 1, 1]
    right_B = [1, 1, -4, 0, 1]
    bottom_B = [0, 1, -4, 1, 1]
    bottom_left_corner = [0, 0, -4, 1, 1]
    bottom_right_corner = [0, 1, -4, 0, 1]
    
    coefficients = []
    
    #bottom row
    coefficients.append(bottom_left_corner)
    
    for j in range(N_x-2):
        coefficients.append(bottom_B)
        
    coefficients.append(bottom_right_corner)
    
    #most of the block except for the top row
    #the top row is set to 0 so it does not need matrix elements
    for i in range(1, N_y-1):
        coefficients.append(left_B)
        
        for j in range(N_x-2):
            coefficients.append(bulk)
            
        coefficients.append(right_B)
        
    #DEFINE THE MATRIX OF POSITIONS
    
    positions = []
    
    #bottom row
    positions.append([0, 0, 0, 1, N_x])
    
    for j in range(1, N_x-1):
        positions.append([0, j-1, j, j+1, j + N_x])
        
    positions.append([0, N_x-2, N_x-1, 0, 2*N_x -1])
    
    #rest of the block
    for i in range(1, N_y-1):
        
        positions.append([(i-1)*N_x, 0, i*N_x, i*N_x + 1, (i+1)*N_x])
        
        for j in range(1, N_x-1):
            positions.append([(i-1)*N_x + j, i*N_x - 1 + j, i*N_x + j, i*N_x + 1 + j, (i+1)*N_x + j])
        
        positions.append([i*N_x - 1 , (i+1)*N_x - 2, (i+1)*N_x - 1, 0, (i+2)*N_x - 1])
        
    return coefficients, positions


# In[20]:


#BLOCK WITHOUT ELECTRODE

#THIS FUNCTION TAKES THE TRANSITION CELLS FROM L, R AND B (transition with block with electrode) AND DEFINES
#THE SOURCE VECTOR
#Pitch, height and mesh size are needed to produce vectors of the right size

#B_L and B_R should have length N_y-1 and include all boundary cells except for the top cell which is set to 0
#B_B is of size 2*N_x since the mesh size in the bottom block is 0.5 microns

def block_without_segment_source(pitch, height, mesh_size, B_L, B_R, B_T):
    
    N_x = int(pitch/mesh_size)
    N_y = int(height/mesh_size)
    
    source = []
    
    for i in range(N_y-1):
        source.append(-B_L[i])
        
        for j in range(1, N_x-1):
            source.append(0)
        
        source.append(-B_R[i])
        
    #add the values from the bottom
    for j , k in zip(range(N_x), range(0, int(2*N_x), 2)):
        source[j] = source[j] - (B_T[k]+B_T[k+1])/2 #take the average of the two cells in the bottom
                                                    #the transition occurs between 0.5 and 1
    
    return source

