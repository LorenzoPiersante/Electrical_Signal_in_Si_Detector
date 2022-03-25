#!/usr/bin/env python
# coding: utf-8

# In[3]:


#weighting_field_microstrip_parallel_update(): takes in a dictionary data type containing the potential of all the 22
#blocks and applies the SOR algorithm in parallel to all the blocks to updated the solution until the residual requirement
#is met

#the function utilizes the functions contained in the module "Block_structure" to create the coefficients matrices and the
#positions matrices needed in the calculation of the potential
#the function uses the module "Solvers_residual" for implementing the solution method

#VARIABLES
#pitch, height: pitch and height of a block, block with and without electrode have the same dimesion but different meshes
#block with electrode has mesh 0.5, block without has mesh 1
#tolerance, over_relax: residual requiremnet and over-relaxation parameter
#potential_dict: dictionary containing the potentials of all 11 blocks with electrodes
#potential_dict_t: dictionary containing the potentials of all 11 blocks without electrodes (sit on top of the previous ones)

#The function returns the updated blocks in the form of a dictionary


# In[ ]:


#These short functions update the transition regions between blocks after each iteration

#This function updates the top transition cells and the right transition cells of a block with electrode
def update_B_block_left(potential, B_1, B_right, N_x, N_y):
    B_1 = potential[(N_y-1)*N_x : N_y*N_x]
    
    for i in range(N_y-1):
        location = int((i+2)*N_x -1)
        B_right[i] = potential[location]
    
    return (B_1, B_right)

#This function updates the top transition cells and the left transition cells of a block with electrode
def update_B_block_right(potential, B_1, B_left, N_x, N_y):
    B_1 = potential[(N_y-1)*N_x : N_y*N_x]
    
    for i in range(N_y-1):
        location = int((i+1)*N_x)
        B_left[i] = potential[location]
    
    return (B_1, B_left)

#This function updates the top transition cells and the right transition cells of a block without electrode
def update_B_block_left_t(potential, B_1, B_right, N_x, N_y):
    B_1 = potential[0:N_x]
    
    for i in range(N_y-1):
        location = int((i+1)*N_x -1)
        B_right[i] = potential[location]
    
    return (B_1, B_right)

#This function updates the top transition cells and the left transition cells of a block without electrode
def update_B_block_right_t(potential, B_1, B_left, N_x, N_y):
    B_1 = potential[0:N_x]
    
    for i in range(N_y-1):
        location = int(i*N_x)
        B_left[i] = potential[location]
        
    return (B_1, B_left)


# In[1]:


import numpy as np
from Block_structure import *
from Solvers_residual import *
import multiprocessing as mp

def weighting_field_microstrip_parallel_update(pitch, height, tolerance, over_relax, potential_dict, potential_dict_t):
    
    #DEFINE DIMENSIONS OF DOMAIN
    #######################################################################################################################
    #mesh size of block with electrode
    mesh_size_1 = 0.5
    N_x_1 = int(pitch/mesh_size_1)
    N_y_1 = int(height/mesh_size_1)
    active_cells_1 = int(N_x_1*(N_y_1 - 1))

    #thickness of the electrode is 0.5
    start_electrode = int((N_x_1*N_y_1)/3 + N_x_1*(2/5))
    end_electrode = int((N_x_1*N_y_1)/3 + N_x_1*(3/5))

    #mesh size of block without electrode
    mesh_size_2 = 1
    N_x_2 = int(pitch/mesh_size_2)
    N_y_2 = int(height/mesh_size_2)
    active_cells_2 = int(N_x_2*(N_y_2-1))
    
    #DEFINE TRANSITION VECTORS BETWEEN BLOCKS
    #######################################################################################################################
    #Define central block left and right boundaries
    B_central_left = np.zeros(N_y_1 - 1)
    B_central_right = np.zeros(N_y_1 - 1)
    #Define central_t block left and right boundaries
    B_central_t_left = np.zeros(N_y_2 - 1)
    B_central_t_right = np.zeros(N_y_2 - 1)
    #Define transition between central bottom and top - ATTENTION THAT THE TOP BOUNDARY OF CENTRAL IS THE BOTTOM OF CENTRAL_t
    B_central_top = np.zeros(N_y_1)
    B_central_t_bottom = np.zeros(N_y_2)

    #Define left_1 block left and right boundaries
    B_left_1_left = np.zeros(N_y_1 - 1)
    B_left_1_right = np.zeros(N_y_1 - 1)
    #Define left_1_t block left and right boundaries
    B_left_1_t_left = np.zeros(N_y_2 - 1)
    B_left_1_t_right = np.zeros(N_y_2 - 1)
    #Define transition between left_1 bottom and top
    B_left_1_top = np.zeros(N_y_1)
    B_left_1_t_bottom = np.zeros(N_y_2)

    #Define left_2 block left and right boundaries
    B_left_2_left = np.zeros(N_y_1 - 1)
    B_left_2_right = np.zeros(N_y_1 - 1)
    #Define left_2_t block left and right boundaries
    B_left_2_t_left = np.zeros(N_y_2 - 1)
    B_left_2_t_right = np.zeros(N_y_2 - 1)
    #Define transition between left_2 bottom and top
    B_left_2_top = np.zeros(N_y_1)
    B_left_2_t_bottom = np.zeros(N_y_2)

    #Define left_3 block left and right boundaries
    B_left_3_left = np.zeros(N_y_1 - 1)
    B_left_3_right = np.zeros(N_y_1 - 1)
    #Define left_3_t block left and right boundaries
    B_left_3_t_left = np.zeros(N_y_2 - 1)
    B_left_3_t_right = np.zeros(N_y_2 - 1)
    #Define transition between left_3 bottom and top
    B_left_3_top = np.zeros(N_y_1)
    B_left_3_t_bottom = np.zeros(N_y_2)

    #Define left_4 block left and right boundaries
    B_left_4_left = np.zeros(N_y_1 - 1)
    B_left_4_right = np.zeros(N_y_1 - 1)
    #Define left_4_t block left and right boundaries
    B_left_4_t_left = np.zeros(N_y_2 - 1)
    B_left_4_t_right = np.zeros(N_y_2 - 1)
    #Define transition between left_4 bottom and top
    B_left_4_top = np.zeros(N_y_1)
    B_left_4_t_bottom = np.zeros(N_y_2)

    #Define right_1 block left and right boundaries
    B_right_1_left = np.zeros(N_y_1 - 1)
    B_right_1_right = np.zeros(N_y_1 - 1)
    #Define right_1_t block left and right boundaries
    B_right_1_t_left = np.zeros(N_y_2 - 1)
    B_right_1_t_right = np.zeros(N_y_2 - 1)
    #Define transition between right_1 bottom and top
    B_right_1_top = np.zeros(N_y_1)
    B_right_1_t_bottom = np.zeros(N_y_2)

    #Define right_2 block left and right boundaries
    B_right_2_left = np.zeros(N_y_1 - 1)
    B_right_2_right = np.zeros(N_y_1 - 1)
    #Define right_2_t block left and right boundaries
    B_right_2_t_left = np.zeros(N_y_2 - 1)
    B_right_2_t_right = np.zeros(N_y_2 - 1)
    #Define transition between right_2 bottom and top
    B_right_2_top = np.zeros(N_y_1)
    B_right_2_t_bottom = np.zeros(N_y_2)

    #Define right_3 block left and right boundaries
    B_right_3_left = np.zeros(N_y_1 - 1)
    B_right_3_right = np.zeros(N_y_1 - 1)
    #Define right_3_t block left and right boundaries
    B_right_3_t_left = np.zeros(N_y_2 - 1)
    B_right_3_t_right = np.zeros(N_y_2 - 1)
    #Define transition between right_3 bottom and top
    B_right_3_top = np.zeros(N_y_1)
    B_right_3_t_bottom = np.zeros(N_y_2)

    #Define right_4 block left and right boundaries
    B_right_4_left = np.zeros(N_y_1 - 1)
    B_right_4_right = np.zeros(N_y_1 - 1)
    #Define right_4_t block left and right boundaries
    B_right_4_t_left = np.zeros(N_y_2 - 1)
    B_right_4_t_right = np.zeros(N_y_2 - 1)
    #Define transition between right_4 bottom and top
    B_right_4_top = np.zeros(N_y_1)
    B_right_4_t_bottom = np.zeros(N_y_2)

    #Define B_left block left and right boundaries
    B_B_left_left = np.zeros(N_y_1 - 1) #never updated!
    B_B_left_right = np.zeros(N_y_1 - 1)
    #Define B_left_t block left and right boundaries
    B_B_left_t_left = np.zeros(N_y_2 - 1) #never updated!
    B_B_left_t_right = np.zeros(N_y_2 - 1)
    #Define transition between B_left bottom and top
    B_B_left_top = np.zeros(N_y_1)
    B_B_left_t_bottom = np.zeros(N_y_2)

    #Define B_right block left and right boundaries
    B_B_right_left = np.zeros(N_y_1 - 1)
    B_B_right_right = np.zeros(N_y_1 - 1) #never updated!
    #Define B_left_t block left and right boundaries
    B_B_right_t_left = np.zeros(N_y_2 - 1)
    B_B_right_t_right = np.zeros(N_y_2 - 1) #never updated!
    #Define transition between B_left bottom and top
    B_B_right_top = np.zeros(N_y_1)
    B_B_right_t_bottom = np.zeros(N_y_2)
    
    #DEFINE COEFFICIENT AND POSITION MATRICES
    #######################################################################################################################
    #Central
    coeff_central, position_central = block_with_segment(pitch, height, mesh_size_1)
    coeff_central_t, position_central_t = block_without_segment(pitch, height, mesh_size_2)

    #convert them to numpy arrays - for SOR algorithm
    coeff_central = np.array(coeff_central)
    coeff_central_t = np.array(coeff_central_t)
    position_central = np.array(position_central)
    position_central_t = np.array(position_central_t)

    #define a dictionary for the different blocks
    #match a block to the required coefficient matrix or position matrix

    coeff_dict_left = { 1 : coeff_central, 2 : coeff_central, 3 : coeff_central, 4 : coeff_central, 5 : coeff_central}
    #left_1, left_2, left_3, left_4, B_left
    coeff_dict_left_t = {1 : coeff_central_t, 2 : coeff_central_t, 3 : coeff_central_t, 4 : coeff_central_t, 5 : coeff_central_t}
    #left_1_t, left_2_t, left_3_t, left_4_t, B_left_t

    coeff_dict_right = { 1 : coeff_central, 2 : coeff_central, 3 : coeff_central, 4 : coeff_central, 5 : coeff_central}
    #right_1, right_2, right_3, right_4, B_right
    coeff_dict_right_t = {1 : coeff_central_t, 2 : coeff_central_t, 3 : coeff_central_t, 4 : coeff_central_t, 5 : coeff_central_t}
    #right_1_t, right_2_t, right_3_t, right_4_t, B_right_t

    position_dict_left = {1 : position_central, 2 : position_central, 3 : position_central, 4 : position_central, 5 : position_central}
    #left_1, left_2, left_3, left_4, B_left
    position_dict_left_t = {1 : position_central_t, 2 : position_central_t, 3 : position_central_t, 4 : position_central_t, 5 : position_central_t}
    #left_1_t, left_2_t, left_3_t, left_4_t, B_left_t

    position_dict_right = { 1 : position_central, 2 : position_central, 3 : position_central, 4 : position_central, 5 : position_central}
    #right_1, right_2, right_3, right_4, B_right
    position_dict_right_t = {1 : position_central_t, 2 : position_central_t, 3 : position_central_t, 4 : position_central_t, 5 : position_central_t}
    #right_1_t, right_2_t, right_3_t, right_4_t, B_right_t
    
    #EXTRACT THE DIFFERENT POTENTIALS FROM THE DICTIONARIES
    #######################################################################################################################
    potential_B_left = potential_dict[0]
    potential_left_4 = potential_dict[1]
    potential_left_3 = potential_dict[2]
    potential_left_2 = potential_dict[3]
    potential_left_1 = potential_dict[4]
    potential_central = potential_dict[5]
    potential_right_1 = potential_dict[6]
    potential_right_2 = potential_dict[7]
    potential_right_3 = potential_dict[8]
    potential_right_4 = potential_dict[9]
    potential_B_right = potential_dict[10]
    
    potential_B_left_t = potential_dict_t[0]
    potential_left_4_t = potential_dict_t[1]
    potential_left_3_t = potential_dict_t[2]
    potential_left_2_t = potential_dict_t[3]
    potential_left_1_t = potential_dict_t[4]
    potential_central_t = potential_dict_t[5]
    potential_right_1_t = potential_dict_t[6]
    potential_right_2_t = potential_dict_t[7]
    potential_right_3_t = potential_dict_t[8]
    potential_right_4_t = potential_dict_t[9]
    potential_B_right_t = potential_dict_t[10]
    
    #UPDATE ALL TRANSITION REGIONS
    #######################################################################################################################
    #central
    B_central_top = potential_central[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_central_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_central_left[i] = potential_central[location]
    #B_central_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_central_right[i] = potential_central[location]
    #central_t
    B_central_t_bottom = potential_central_t[0:N_x_2]
    #B_central_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_central_t_left[i] = potential_central_t[location]
    #B_central_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_central_t_right[i] = potential_central_t[location]
    #left_1
    B_left_1_top = potential_left_1[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_left_1_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_left_1_left[i] = potential_left_1[location]
    #B_left_1_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_left_1_right[i] = potential_left_1[location]
    #right_1
    B_right_1_top = potential_right_1[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_right_1_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_right_1_left[i] = potential_right_1[location]
    #B_right_1_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_right_1_right[i] = potential_right_1[location]
    #left_1_t
    B_left_1_t_bottom = potential_left_1_t[0:N_x_2]
    #B_left_1_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_left_1_t_left[i] = potential_left_1_t[location]
    #B_left_1_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_left_1_t_right[i] = potential_left_1_t[location]
    #right_1_t
    B_right_1_t_bottom = potential_right_1_t[0:N_x_2]
    #B_right_1_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_right_1_t_left[i] = potential_right_1_t[location]
    #B_right_1_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_right_1_t_right[i] = potential_right_1_t[location]
    #left_2
    B_left_2_top = potential_left_2[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_left_2_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_left_2_left[i] = potential_left_2[location]
    #B_left_2_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_left_2_right[i] = potential_left_2[location]
    #right_2
    B_right_2_top = potential_right_2[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_right_2_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_right_2_left[i] = potential_right_2[location]
    #B_right_2_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_right_2_right[i] = potential_right_2[location]
    #left_2_t
    B_left_2_t_bottom = potential_left_2_t[0:N_x_2]
    #B_left_2_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_left_2_t_left[i] = potential_left_2_t[location]
    #B_left_1_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_left_2_t_right[i] = potential_left_2_t[location]
    #right_2_t
    B_right_2_t_bottom = potential_right_2_t[0:N_x_2]
    #B_right_2_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_right_2_t_left[i] = potential_right_2_t[location]
    #B_right_2_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_right_2_t_right[i] = potential_right_2_t[location]
    #left_3
    B_left_3_top = potential_left_3[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_left_3_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_left_3_left[i] = potential_left_3[location]
    #B_left_3_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_left_3_right[i] = potential_left_3[location]
    #right_3
    B_right_3_top = potential_right_3[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_right_3_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_right_3_left[i] = potential_right_3[location]
    #B_right_3_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_right_3_right[i] = potential_right_3[location]
    #left_3_t
    B_left_3_t_bottom = potential_left_3_t[0:N_x_2]
    #B_left_3_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_left_3_t_left[i] = potential_left_3_t[location]
    #B_left_3_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_left_3_t_right[i] = potential_left_3_t[location]
    #right_3_t
    B_right_3_t_bottom = potential_right_3_t[0:N_x_2]
    #B_right_3_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_right_3_t_left[i] = potential_right_3_t[location]
    #B_right_3_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_right_3_t_right[i] = potential_right_3_t[location]
    #left_4
    B_left_4_top = potential_left_4[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_left_4_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_left_4_left[i] = potential_left_4[location]
    #B_left_4_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_left_4_right[i] = potential_left_4[location]
    #right_4
    B_right_4_top = potential_right_4[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_right_4_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_right_4_left[i] = potential_right_4[location]
    #B_right_4_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_right_4_right[i] = potential_right_4[location]
    #left_4_t
    B_left_4_t_bottom = potential_left_4_t[0:N_x_2]
    #B_left_4_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_left_4_t_left[i] = potential_left_4_t[location]
    #B_left_4_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_left_4_t_right[i] = potential_left_4_t[location]
    #right_4_t
    B_right_4_t_bottom = potential_right_4_t[0:N_x_2]
    #B_right_4_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_right_4_t_left[i] = potential_right_4_t[location]
    #B_right_4_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_right_4_t_right[i] = potential_right_4_t[location]
    #B_left
    B_B_left_top = potential_B_left[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_B_left_right
    for i in range(N_y_1-1):
        location = int((i+2)*N_x_1 -1)
        B_B_left_right[i] = potential_B_left[location]
    #B_left_t
    B_B_left_t_bottom = potential_B_left_t[0:N_x_2]
    #B_B_left_t_right
    for i in range(N_y_2-1):
        location = int((i+1)*N_x_2 -1)
        B_B_left_t_right[i] = potential_B_left_t[location]
    #B_right
    B_B_right_top = potential_B_right[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
    #B_B_right_left
    for i in range(N_y_1-1):
        location = int((i+1)*N_x_1)
        B_B_right_left[i] = potential_B_right[location]
    #B_right_t
    B_B_right_t_bottom = potential_B_right_t[0:N_x_2]
    #B_B_right_t_left
    for i in range(N_y_2-1):
        location = int(i*N_x_2)
        B_B_right_t_left[i] = potential_B_right_t[location]
    
    #ITERATION WITH SOR ALGORITHM - here go the tolerance and over-relax parameters
    #######################################################################################################################
    R_tot = 1
    iteration = 0

    #source of cetral block
    b_central = block_with_segment_source(pitch, height, mesh_size_1, B_left_1_right , B_right_1_left, B_central_t_bottom)
    #left block right cells will be source for the left cells of central and so on for the other blocks

    while R_tot > tolerance:

        #CENTRAL BLOCKS
        #######################################################################################################################
        #central block
        potential_central = SOR_one_iter(coeff_central, position_central, b_central, potential_central, over_relax, active_cells_1)

        #update transition regions
        B_central_top = potential_central[(N_y_1-1)*N_x_1 : N_y_1*N_x_1]
        #B_central_left
        for i in range(N_y_1-1):
            location = int((i+1)*N_x_1)
            B_central_left[i] = potential_central[location]
        #B_central_right
        for i in range(N_y_1-1):
            location = int((i+2)*N_x_1 -1)
            B_central_right[i] = potential_central[location]

        #central_t block
        b_central_t = block_without_segment_source(pitch, height, mesh_size_2, B_left_1_t_right, B_right_1_t_left, B_central_top)

        potential_central_t = SOR_one_iter(coeff_central_t, position_central_t, b_central_t, potential_central_t, over_relax, active_cells_2)

        #update transition regions
        B_central_t_bottom = potential_central_t[0:N_x_2]
        #B_central_t_left
        for i in range(N_y_2-1):
            location = int(i*N_x_2)
            B_central_t_left[i] = potential_central_t[location]
        #B_central_t_right
        for i in range(N_y_2-1):
            location = int((i+1)*N_x_2 -1)
            B_central_t_right[i] = potential_central_t[location]

        #1-BLOCKS
        #######################################################################################################################
        #bottom blocks
        #update sources
        pool = mp.Pool(2)
        
        out_b_left_1 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_left_2_right, B_central_left, B_left_1_t_bottom))
        out_b_right_1 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_central_right, B_right_2_left, B_right_1_t_bottom))
        
        b_left_1 = out_b_left_1.get()
        b_right_1 = out_b_right_1.get()
        
        #SOR iteration
        
        out_potential_left_1 = pool.apply_async(SOR_one_iter, args=(coeff_dict_left[1], position_dict_left[1], b_left_1, potential_left_1, over_relax, active_cells_1))
        out_potential_right_1 = pool.apply_async(SOR_one_iter, args=(coeff_dict_right[1], position_dict_right[1], b_right_1, potential_right_1, over_relax, active_cells_1))
        
        potential_left_1 = out_potential_left_1.get()
        potential_right_1 = out_potential_right_1.get()
        
        
        #update transition regions
        
        out_trans_left_1 = pool.apply_async(update_trans_regions, args=(potential_left_1, B_left_1_top, B_left_1_left, B_left_1_right, N_x_1, N_y_1))
        out_trans_right_1 = pool.apply_async(update_trans_regions, args=(potential_right_1, B_right_1_top, B_right_1_left, B_right_1_right, N_x_1, N_y_1))
        
        trans_left_1 = out_trans_left_1.get()
        trans_right_1 = out_trans_right_1.get()
        
        pool.close()
        
        #unpack tuples of the transition regions
        B_left_1_top, B_left_1_left, B_left_1_right = trans_left_1
        B_right_1_top, B_right_1_left, B_right_1_right = trans_right_1
        
        #top blocks
        #update the source terms
        pool = mp.Pool(2)
        
        out_b_left_1_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_left_2_t_right, B_central_t_left, B_left_1_top))
        out_b_right_1_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_central_t_right, B_right_2_t_left, B_right_1_top))
        
        b_left_1_t = out_b_left_1_t.get()
        b_right_1_t = out_b_right_1_t.get()
        
        #SOR iteration
        
        out_potential_left_1_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_left_t[1], position_dict_left_t[1], b_left_1_t, potential_left_1_t, over_relax, active_cells_2))
        out_potential_right_1_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_right_t[1], position_dict_right_t[1], b_right_1_t, potential_right_1_t, over_relax, active_cells_2))
        
        potential_left_1_t = out_potential_left_1_t.get()
        potential_right_1_t = out_potential_right_1_t.get()
        
        #update transition regions
        
        out_trans_left_1_t = pool.apply_async(update_trans_regions_t, args=(potential_left_1_t, B_left_1_t_bottom, B_left_1_t_left, B_left_1_t_right, N_x_2, N_y_2))
        out_trans_right_1_t = pool.apply_async(update_trans_regions_t, args=(potential_right_1_t, B_right_1_t_bottom, B_right_1_t_left, B_right_1_t_right, N_x_2, N_y_2))
        
        trans_left_1_t = out_trans_left_1_t.get()
        trans_right_1_t = out_trans_right_1_t.get()
        
        pool.close()
        
        #unpack tuples of the transition regions
        B_left_1_t_bottom, B_left_1_t_left, B_left_1_t_right = trans_left_1_t
        B_right_1_t_bottom, B_right_1_t_left, B_right_1_t_right = trans_right_1_t

        #2-BLOCKS
        #######################################################################################################################
        #bottom blocks
        #update sources
        pool = mp.Pool(2)
        
        out_b_left_2 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_left_3_right, B_left_1_left, B_left_2_t_bottom))
        out_b_right_2 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_right_1_right, B_right_3_left, B_right_2_t_bottom))
        
        b_left_2 = out_b_left_2.get()
        b_right_2 = out_b_right_2.get()
        
        #SOR iteration
        
        out_potential_left_2 = pool.apply_async(SOR_one_iter, args=(coeff_dict_left[2], position_dict_left[2], b_left_2, potential_left_2, over_relax, active_cells_1))
        out_potential_right_2 = pool.apply_async(SOR_one_iter, args=(coeff_dict_right[2], position_dict_right[2], b_right_2, potential_right_2, over_relax, active_cells_1))
        
        potential_left_2 = out_potential_left_2.get()
        potential_right_2 = out_potential_right_2.get()
        
        #update transition regions
        
        out_trans_left_2 = pool.apply_async(update_trans_regions, args=(potential_left_2, B_left_2_top, B_left_2_left, B_left_2_right, N_x_1, N_y_1))
        out_trans_right_2 = pool.apply_async(update_trans_regions, args=(potential_right_2, B_right_2_top, B_right_2_left, B_right_2_right, N_x_1, N_y_1))
        
        trans_left_2 = out_trans_left_2.get()
        trans_right_2 = out_trans_right_2.get()
        
        pool.close()
        
        #unpack tuples
        B_left_2_top, B_left_2_left, B_left_2_right = trans_left_2
        B_right_2_top, B_right_2_left, B_right_2_right = trans_right_2
        
        #top blocks
        #update source terms
        pool = mp.Pool(2)
        
        out_b_left_2_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_left_3_t_right, B_left_1_t_left, B_left_2_top))
        out_b_right_2_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_right_1_t_right, B_right_3_t_left, B_right_2_top))
        
        b_left_2_t = out_b_left_2_t.get()
        b_right_2_t = out_b_right_2_t.get()
        
        #SOR iteration
        
        out_potential_left_2_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_left_t[2], position_dict_left_t[2], b_left_2_t, potential_left_2_t, over_relax, active_cells_2))
        out_potential_right_2_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_right_t[2], position_dict_right_t[2], b_right_2_t, potential_right_2_t, over_relax, active_cells_2))
        
        potential_left_2_t = out_potential_left_2_t.get()
        potential_right_2_t = out_potential_right_2_t.get()
        
        #update transition regions
        
        out_trans_left_2_t = pool.apply_async(update_trans_regions_t, args=(potential_left_2_t, B_left_2_t_bottom, B_left_2_t_left, B_left_2_t_right, N_x_2, N_y_2))
        out_trans_right_2_t = pool.apply_async(update_trans_regions_t, args=(potential_right_2_t, B_right_2_t_bottom, B_right_2_t_left, B_right_2_t_right, N_x_2, N_y_2))
        
        trans_left_2_t = out_trans_left_2_t.get()
        trans_right_2_t = out_trans_right_2_t.get()
        
        pool.close()
        
        #unpack tuples
        B_left_2_t_bottom, B_left_2_t_left, B_left_2_t_right = trans_left_2_t
        B_right_2_t_bottom, B_right_2_t_left, B_right_2_t_right = trans_right_2_t

        #3-BLOCKS
        #######################################################################################################################
        #bottom blocks
        #update source terms
        pool = mp.Pool(2)
        
        out_b_left_3 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_left_4_right, B_left_2_left, B_left_3_t_bottom))
        out_b_right_3 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_right_2_right, B_right_4_left, B_right_3_t_bottom))
        
        b_left_3 = out_b_left_3.get()
        b_right_3 = out_b_right_3.get()
        
        #SOR iteration
        
        out_potential_left_3 = pool.apply_async(SOR_one_iter, args=(coeff_dict_left[3], position_dict_left[3], b_left_3, potential_left_3, over_relax, active_cells_1))
        out_potential_right_3 = pool.apply_async(SOR_one_iter, args=(coeff_dict_right[3], position_dict_right[3], b_right_3, potential_right_3, over_relax, active_cells_1))
        
        potential_left_3 = out_potential_left_3.get()
        potential_right_3 = out_potential_right_3.get()

        #update transition regions
        
        out_trans_left_3 = pool.apply_async(update_trans_regions, args=(potential_left_3, B_left_3_top, B_left_3_left, B_left_3_right, N_x_1, N_y_1))
        out_trans_right_3 = pool.apply_async(update_trans_regions, args=(potential_right_3, B_right_3_top, B_right_3_left, B_right_3_right, N_x_1, N_y_1))
        
        trans_left_3 = out_trans_left_3.get()
        trans_right_3 = out_trans_right_3.get()
        
        pool.close()
        
        #unpack the tuples
        B_left_3_top, B_left_3_left, B_left_3_right = trans_left_3
        B_right_3_top, B_right_3_left, B_right_3_right = trans_right_3

        #top blocks
        #update transition regions
        pool = mp.Pool(2)
        
        out_b_left_3_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_left_4_t_right, B_left_2_t_left, B_left_3_top))
        out_b_right_3_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_right_2_t_right, B_right_4_t_left, B_right_3_top))
        
        b_left_3_t = out_b_left_3_t.get()
        b_right_3_t = out_b_right_3_t.get()
        
        #SOR iteration
        
        out_potential_left_3_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_left_t[3], position_dict_left_t[3], b_left_3_t, potential_left_3_t, over_relax, active_cells_2))
        out_potential_right_3_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_right_t[3], position_dict_right_t[3], b_right_3_t, potential_right_3_t, over_relax, active_cells_2))
        
        potential_left_3_t = out_potential_left_3_t.get()
        potential_right_3_t = out_potential_right_3_t.get()

        #update transition regions
        
        out_trans_left_3_t = pool.apply_async(update_trans_regions_t, args=(potential_left_3_t, B_left_3_t_bottom, B_left_3_t_left, B_left_3_t_right, N_x_2, N_y_2))
        out_trans_right_3_t = pool.apply_async(update_trans_regions_t, args=(potential_right_3_t, B_right_3_t_bottom, B_right_3_t_left, B_right_3_t_right, N_x_2, N_y_2))
        
        trans_left_3_t = out_trans_left_3_t.get()
        trans_right_3_t = out_trans_right_3_t.get()
        
        pool.close()
        
        #unpack tuples
        B_left_3_t_bottom, B_left_3_t_left, B_left_3_t_right = trans_left_3_t
        B_right_3_t_bottom, B_right_3_t_left, B_right_3_t_right = trans_right_3_t

        #4-BLOCKS
        #######################################################################################################################
        #top blocks
        #update source terms
        pool = mp.Pool(2)
        
        out_b_left_4 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_B_left_right, B_left_3_left, B_left_4_t_bottom))
        out_b_right_4 = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_right_3_right, B_B_right_left, B_right_4_t_bottom))
        
        b_left_4 = out_b_left_4.get()
        b_right_4 = out_b_right_4.get()
        
        #SOR iteration
        
        out_potential_left_4 = pool.apply_async(SOR_one_iter, args=(coeff_dict_left[4], position_dict_left[4], b_left_4, potential_left_4, over_relax, active_cells_1))
        out_potential_right_4 = pool.apply_async(SOR_one_iter, args=(coeff_dict_right[4], position_dict_right[4], b_right_4, potential_right_4, over_relax, active_cells_1))
        
        potential_left_4 = out_potential_left_4.get()
        potential_right_4 = out_potential_right_4.get()
        
        #update transition regions

        out_trans_left_4 = pool.apply_async(update_trans_regions, args=(potential_left_4, B_left_4_top, B_left_4_left, B_left_4_right, N_x_1, N_y_1))
        out_trans_right_4 = pool.apply_async(update_trans_regions, args=(potential_right_4, B_right_4_top, B_right_4_left, B_right_4_right, N_x_1, N_y_1))
        
        trans_left_4 = out_trans_left_4.get()
        trans_right_4 = out_trans_right_4.get()
        
        pool.close()
        
        #unpack the tuples
        B_left_4_top, B_left_4_left, B_left_4_right = trans_left_4
        B_right_4_top, B_right_4_left, B_right_4_right = trans_right_4

        #top blocks
        #update source term
        pool = mp.Pool(2)    
        
        out_b_left_4_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_B_left_t_right, B_left_3_t_left, B_left_4_top))
        out_b_right_4_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_right_3_t_right, B_B_right_t_left, B_right_4_top))
        
        b_left_4_t = out_b_left_4_t.get()
        b_right_4_t = out_b_right_4_t.get()
        
        #SOR iteration
        
        out_potential_left_4_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_left_t[4], position_dict_left_t[4], b_left_4_t, potential_left_4_t, over_relax, active_cells_2))
        out_potential_right_4_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_right_t[4], position_dict_right_t[4], b_right_4_t, potential_right_4_t, over_relax, active_cells_2))
        
        potential_left_4_t = out_potential_left_4_t.get()
        potential_right_4_t = out_potential_right_4_t.get()
        
        #update transition regions

        out_trans_left_4_t = pool.apply_async(update_trans_regions_t, args=(potential_left_4_t, B_left_4_t_bottom, B_left_4_t_left, B_left_4_t_right, N_x_2, N_y_2))
        out_trans_right_4_t = pool.apply_async(update_trans_regions_t, args=(potential_right_4_t, B_right_4_t_bottom, B_right_4_t_left, B_right_4_t_right, N_x_2, N_y_2))
        
        trans_left_4_t = out_trans_left_4_t.get()
        trans_right_4_t = out_trans_right_4_t.get()
        
        pool.close()
        
        #unpack tuples
        B_left_4_t_bottom, B_left_4_t_left, B_left_4_t_right = trans_left_4_t
        B_right_4_t_bottom, B_right_4_t_left, B_right_4_t_right = trans_right_4_t

        #BOUNDARY BLOCKS
        #######################################################################################################################
        #bottom blocks
        #update source term
        pool = mp.Pool(2)
        
        out_b_B_left = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_B_left_left, B_left_4_left, B_B_left_t_bottom))
        out_b_B_right = pool.apply_async(block_with_segment_source, args=(pitch, height, mesh_size_1, B_right_4_right, B_B_right_right, B_B_right_t_bottom))
        
        b_B_left = out_b_B_left.get()
        b_B_right = out_b_B_right.get()
        
        #SOR iteration
        
        out_potential_B_left = pool.apply_async(SOR_one_iter, args=(coeff_dict_left[5], position_dict_left[5], b_B_left, potential_B_left, over_relax, active_cells_1))
        out_potential_B_right = pool.apply_async(SOR_one_iter, args=(coeff_dict_right[5], position_dict_right[5], b_B_right, potential_B_right, over_relax, active_cells_1))
        
        potential_B_left = out_potential_B_left.get()
        potential_B_right = out_potential_B_right.get()
        
        #update transition regions
        
        out_trans_B_left = pool.apply_async(update_B_block_left, args=(potential_B_left, B_B_left_top, B_B_left_right, N_x_1, N_y_1))
        out_trans_B_right = pool.apply_async(update_B_block_left, args=(potential_B_right, B_B_right_top, B_B_right_left, N_x_1, N_y_1))
        
        trans_B_left = out_trans_B_left.get()
        trans_B_right = out_trans_B_right.get()
        
        pool.close()
        
        #unpack tuples
        B_B_left_top, B_B_left_right = trans_B_left
        B_B_right_top, B_B_right_left = trans_B_right
        
        #top blocks
        #update source term
        pool = mp.Pool(2)
        
        out_b_B_left_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_B_left_t_left, B_left_4_left, B_B_left_top))
        out_b_B_right_t = pool.apply_async(block_without_segment_source, args=(pitch, height, mesh_size_2, B_right_4_t_right, B_B_right_t_right, B_B_right_top))
        
        b_B_left_t = out_b_B_left_t.get()
        b_B_right_t = out_b_B_right_t.get()
        
        #SOR iteration
        
        out_potential_B_left_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_left_t[5], position_dict_left_t[5], b_B_left_t, potential_B_left_t, over_relax, active_cells_2))
        out_potential_B_right_t = pool.apply_async(SOR_one_iter, args=(coeff_dict_right_t[5], position_dict_right_t[5], b_B_right_t, potential_B_right_t, over_relax, active_cells_2))
        
        potential_B_left_t = out_potential_B_left_t.get()
        potential_B_right_t = out_potential_B_right_t.get()
        
        #update transition regions
        
        out_trans_B_left_t = pool.apply_async(update_B_block_left_t, args=(potential_B_left_t, B_B_left_t_bottom, B_B_left_t_right, N_x_2, N_y_2))
        out_trans_B_right_t = pool.apply_async(update_B_block_left_t, args=(potential_B_right_t, B_B_right_t_bottom, B_B_right_t_left, N_x_2, N_y_2))
        
        trans_B_left_t = out_trans_B_left_t.get()
        trans_B_right_t = out_trans_B_right_t.get()
        
        pool.close()
        
        #unpack tuples
        B_B_left_t_bottom, B_B_left_t_right = trans_B_left_t
        B_B_right_t_bottom, B_B_right_t_left = trans_B_right_t

        #######################################################################################################################
        #update the source for the central block bottom
        b_central = block_with_segment_source(pitch, height, mesh_size_1, B_left_1_right, B_right_1_left, B_central_t_bottom)

        #######################################################################################################################

        #calculate the global residual for the entire domain
        
        pool = mp.Pool(8)
        
        out_central = pool.apply_async(global_residual, args=(coeff_central, position_central, b_central, potential_central, active_cells_1))
        out_central_t = pool.apply_async(global_residual, args=(coeff_central_t, position_central_t, b_central_t, potential_central_t, active_cells_2))
        
        out_left_1 = pool.apply_async(global_residual, args=(coeff_dict_left[1], position_dict_left[1], b_left_1, potential_left_1, active_cells_1))
        out_left_1_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[1], position_dict_left_t[1], b_left_1_t, potential_left_1_t, active_cells_2))
        out_right_1 = pool.apply_async(global_residual, args=(coeff_dict_right[1], position_dict_right[1], b_right_1, potential_right_1, active_cells_1))
        out_right_1_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[1], position_dict_right_t[1], b_right_1_t, potential_right_1_t, active_cells_2))
        
        out_left_2 = pool.apply_async(global_residual, args=(coeff_dict_left[2], position_dict_left[2], b_left_2, potential_left_2, active_cells_1))
        out_left_2_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[2], position_dict_left_t[2], b_left_2_t, potential_left_2_t, active_cells_2))
        out_right_2 = pool.apply_async(global_residual, args=(coeff_dict_right[2], position_dict_right[2], b_right_2, potential_right_2, active_cells_1))
        out_right_2_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[2], position_dict_right_t[2], b_right_2_t, potential_right_2_t, active_cells_2))
        
        out_left_3 = pool.apply_async(global_residual, args=(coeff_dict_left[3], position_dict_left[3], b_left_3, potential_left_3, active_cells_1))
        out_left_3_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[3], position_dict_left_t[3], b_left_3_t, potential_left_3_t, active_cells_2))
        out_right_3 = pool.apply_async(global_residual, args=(coeff_dict_right[3], position_dict_right[3], b_right_3, potential_right_3, active_cells_1))
        out_right_3_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[3], position_dict_right_t[3], b_right_3_t, potential_right_3_t, active_cells_2))
        
        out_left_3 = pool.apply_async(global_residual, args=(coeff_dict_left[3], position_dict_left[3], b_left_3, potential_left_3, active_cells_1))
        out_left_3_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[3], position_dict_left_t[3], b_left_3_t, potential_left_3_t, active_cells_2))
        out_right_3 = pool.apply_async(global_residual, args=(coeff_dict_right[3], position_dict_right[3], b_right_3, potential_right_3, active_cells_1))
        out_right_3_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[3], position_dict_right_t[3], b_right_3_t, potential_right_3_t, active_cells_2))

        out_left_4 = pool.apply_async(global_residual, args=(coeff_dict_left[4], position_dict_left[4], b_left_4, potential_left_4, active_cells_1))
        out_left_4_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[4], position_dict_left_t[4], b_left_4_t, potential_left_4_t, active_cells_2))
        out_right_4 = pool.apply_async(global_residual, args=(coeff_dict_right[4], position_dict_right[4], b_right_4, potential_right_4, active_cells_1))
        out_right_4_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[4], position_dict_right_t[4], b_right_4_t, potential_right_4_t, active_cells_2))
        
        out_B_left = pool.apply_async(global_residual, args=(coeff_dict_left[5], position_dict_left[5], b_B_left, potential_B_left, active_cells_1))
        out_B_left_t = pool.apply_async(global_residual, args=(coeff_dict_left_t[5], position_dict_left_t[5], b_B_left_t, potential_B_left_t, active_cells_2))
        out_B_right = pool.apply_async(global_residual, args=(coeff_dict_right[5], position_dict_right[5], b_B_right, potential_B_right, active_cells_1))
        out_B_right_t = pool.apply_async(global_residual, args=(coeff_dict_right_t[5], position_dict_right_t[5], b_B_right_t, potential_B_right_t, active_cells_2))
        
        R_central = out_central.get()
        R_central_t = out_central_t.get()
        R_left_1 = out_left_1.get()
        R_left_1_t = out_left_1_t.get()
        R_right_1 = out_right_1.get()
        R_right_1_t = out_right_1_t.get()
        R_left_2 = out_left_2.get()
        R_left_2_t = out_left_2_t.get()
        R_right_2 = out_right_2.get()
        R_right_2_t = out_right_2_t.get()
        R_left_3 = out_left_3.get()
        R_left_3_t = out_left_3_t.get()
        R_right_3 = out_right_3.get()
        R_right_3_t = out_right_3_t.get()
        R_left_4 = out_left_4.get()
        R_left_4_t = out_left_4_t.get()
        R_right_4 = out_right_4.get()
        R_right_4_t = out_right_4_t.get()
        R_B_left = out_B_left.get()
        R_B_left_t = out_B_left_t.get()
        R_B_right = out_B_right.get()
        R_B_right_t = out_B_right_t.get()
        
        pool.close()
        
        R_tot = R_central + R_central_t
        R_tot = R_tot + R_left_1 + R_left_1_t + R_right_1 + R_right_1_t
        R_tot = R_tot + R_left_2 + R_left_2_t + R_right_2 + R_right_2_t
        R_tot = R_tot + R_left_3 + R_left_3_t + R_right_3 + R_right_3_t
        R_tot = R_tot + R_left_4 + R_left_4_t + R_right_4 + R_right_4_t
        R_tot = R_tot + R_B_left + R_B_left_t + R_B_right + R_B_right_t

        print("Iteration = " + str(iteration))
        print("R_tot = " + str(R_tot))

        iteration = iteration + 1
    
    return {0 : potential_B_left, 1 : potential_left_4, 2 : potential_left_3, 3 : potential_left_2, 4 : potential_left_1, 5 : potential_central, 
            6 : potential_right_1, 7 : potential_right_2, 8 : potential_right_3, 9 : potential_right_4, 10 : potential_B_right}, {0 : potential_B_left_t, 
            1 : potential_left_4_t, 2 : potential_left_3_t, 3 : potential_left_2_t, 4 : potential_left_1_t, 5 : potential_central_t, 6 : potential_right_1_t, 
            7 : potential_right_2_t, 8 : potential_right_3_t, 9 : potential_right_4_t, 10 : potential_B_right_t}

