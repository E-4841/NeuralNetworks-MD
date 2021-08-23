import fileinput
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import keras

import network_activation_functions
import custom_layer
from custom_layer       import My_layer
from frame              import Frame
from keras              import backend as K
from keras              import optimizers
from keras.models       import Sequential, Model, save_model, load_model
from keras.layers       import Activation, Input, Flatten
from keras.layers.core  import Dense, Dropout
from keras.layers.merge import Add
from keras.utils        import plot_model
from keras.models       import load_model

# Construct argument parser
parser = argparse.ArgumentParser()
parser.add_argument( '-m', '--mode', type = str, required = False,
                     help = 'train or dynamics' )
parser.add_argument( '-f', '--file', type = str, required = False,
                     help = 'Path of the file with the required parameters' )
parser.add_argument( '-o', '--output', type = bool, required = False,
                     help = 'Save symmetry functions in function.data' )
args = vars( parser.parse_args() )


# C L A S S E S #-----------------------------------------------------------------------#

# radial symmetry function ##############################################################
class SymFunc_G2:
    def __init__(self, i, atom, parameters):
        self.type = 'radial'
        self.index = i
        self.i_atom = atom
        self.j_atom = parameters[0]
        self.eta = parameters[1]
        self.Rs = parameters[2]
        self.Rc = parameters[3]

# angular symmetry function #############################################################
class SymFunc_G3:
    def __init__(self, i, atom, parameters):
        self.type = 'angular'
        self.index = i
        self.i_atom = atom
        self.j_atom = parameters[0]
        self.k_atom = parameters[1]
        self.eta = parameters[2]
        self.lamnda = parameters[3]
        self.zeta = parameters[4]
        self.Rc = parameters[5]
        try:
            self.Rs = parameters[6]
        except:
            self.Rs = 0.0
            
# atom ##################################################################################          
class Atom:
    def __init__(self, atom_index, atom_data):
        self.index = atom_index
        self.position = atom_data[:-1].astype(np.float)
        self.symbol = atom_data[-1]
        self.velocity = np.zeros((3))
        self.force = np.zeros((3))
        
    def atomic_number(self, atom_number):
        self.number = atom_number
        return
    
    def atomic_mass(self, atomic_mass):
        self.mass = atomic_mass
        return
    
    def neighbor_atoms_list(self, neighbors_list):
        self.neighbors = neighbors_list
        return 
    
    def features(self, atom_features):
        self.G = atom_features
        return
    
    def network_derivative(self, derivatives):
        self.dEdG = derivatives
        return
    
    def features_derivative(self, derivatives):
        self.dGdr = derivatives
        return
    
    def neighbor_derivatives(self, derivatives):
        self.dGdn = derivatives
        return         
            
            
# F U N C I O N E S   P A R A   L E E R   A R C H I V O S #-----------------------------#

# archivo de control ####################################################################
def read_parameters(file):
    
    # Get parameters
    params = {}
    for line in fileinput.input(file):
        if line[0] not in ['#', '\n']:
            temp = line.split()
            if len(temp) > 2 and '[' not in temp[1]:
                params[temp[0]] = temp[1:]
            elif temp[1][0] == '[':           
                temp[1] = temp[1].replace('[','')
                temp[-1] = temp[-1].replace(']','')
                items = []
                for item in temp[1:]:
                    if ',' in item:
                        item = item.replace(',', '')
                    item = int(item)
                    items.append(item)
                params[temp[0]] = items
            else:
                try:
                    temp[1] = int(temp[1])
                except ValueError:
                    str(temp[1])
                params[temp[0]] = temp[1]
    fileinput.close()
    
    # Valid activation functions
    activationFunctions = ['hyperbolic_tangent', 'hyperbolic_tangent_with_linear_twist', 
                       'sigmoid', 'identity', 'rectilinear']
    if params['activation_function'] not in activationFunctions:
            print( '{}: Invalid activation function'.format(settings['activation_function']) )
            return
    
    # Group atomic info in lists
    params['atomic_symbol'] = []
    params['atomic_number'] = []
    params['atomic_mass'] = []
    params['sym_func_files'] = []
    
    for atype in range(int(params['number_of_atom_types'])):
        temp = 'atom_type_' + str(atype + 1)
        params['atomic_symbol'].append(params[temp][0])
        params['atomic_number'].append(int(params[temp][1]))
        params['atomic_mass'].append(float(params[temp][2]))
        params['sym_func_files'].append(params[temp][3])
        del params[temp]
    return params           
            
    
# parámetros de las funciones de simetría ###############################################
def read_SymFunc_parameters(files):
    G = []
    for file in files:
        indx = 0
        G_per_atom = []
        atom_i = file.split('-')[0]
        if '/' in atom_i:
            atom_i = atom_i.split('/')[-1]
        G_per_atom.append(atom_i)
        for line in fileinput.input(file):
            temp = line.split()
            if len(temp) > 0 and temp[0] != '#':
                if len(temp) == 4:
                    for i in range(1, 4):
                        temp[i] = float(temp[i])
                    temp = SymFunc_G2(indx, atom_i, temp)
                    G_per_atom.append(temp)
                    indx += 1
                    continue
                if len(temp) > 4:
                    for i in range(2, 6):
                        temp[i] = float(temp[i])
                    temp = SymFunc_G3(indx, atom_i, temp)
                    G_per_atom.append(temp)
                    indx += 1
        G.append(G_per_atom)
        fileinput.close()              
    
    return G           
            
    
# posiciones atómicas ###################################################################
def read_positions(file):
    
    frame = []   # This list will contain an item per frame
    
    for line in fileinput.input(file):
        temp = line.split()
        if temp[0] == 'begin':
            tempframe = []
        if temp[0] == 'atom':
            tempframe.append(temp[1:5])
        if temp[0] == 'energy':
            temp_en = [0.0, temp[1], temp[1], 0.0]
            tempframe.append(temp_en)
            tempframe = np.array(tempframe)
            
            frame.append(tempframe)
    
    fileinput.close()   
    frame = np.array(frame)
    return frame    

# todos los datos #######################################################################
def read_data(file):
    params = read_parameters(file)
    coordinates = read_positions(params['atomic_data_path'])
    G = read_SymFunc_parameters(params['sym_func_files'])
    
    return params, coordinates, G

# distancia entre un par de átomos ######################################################
def distance(atom_i, atom_j):
    dij = 0.0
    for i in range(3):
        dij += (float(atom_i[i]) - float(atom_j[i]))**2
    dij = np.sqrt(dij)
    return abs(dij)

def create_lists(total):
    k = []
    for i in range(total):
        k.append([])
    
    return k

# carga los parámetros optimizados del modelo de la red neuronal ########################
def get_weights(params, G):
    
    model = load_model(params['model_path'],
                   compile = True,
                   custom_objects = { 
                       params['activation_function']: getattr(network_activation_functions, params['activation_function']),
                       'My_layer' : My_layer })
    nn_params = model.get_weights()
    
    atom_types = len(G)
    neurons_per_layer = params['neurons_per_layer']
    
    w_lists = create_lists(atom_types)
    for i in range(atom_types):
        w_lists[i].append(G[i][0])
        for j in range(len(neurons_per_layer)):
            k = j * 2 * atom_types + i * 2
            w_lists[i].append(nn_params[k])
    weights = np.array(w_lists)
    
    b_lists = create_lists(atom_types)
    for i in range(atom_types):
        b_lists[i].append(G[i][0])
        for j in range(len(neurons_per_layer)):
            k = j * 2 * atom_types + i * 2 + 1
            b_lists[i].append(nn_params[k])
    biases = np.array(b_lists)
    
    return weights, biases

# funciones de corte ####################################################################
def cut_function(d_ij, r_c, fc):
    if int(fc) == 1:
        return 0.5 * np.cos(np.pi*d_ij/r_c) + 0.5
    if int(fc) == 2:
        return np.power( np.tanh(1.0 - (d_ij/r_c)), 3.0 )

# derivada de las funciones de corte ####################################################
def cut_function_derivative(d_ij, r_c, fc):
    if int(fc) == 1:
        return - 0.5 * np.pi * np.sin(np.pi*d_ij/r_c) / r_c
    if int(fc) == 2:
        temp = np.power( np.tanh(1.0 - (d_ij/r_c)), 2.0 )
        return 3 * temp * (temp - 1.0) / r_c   

# genera las listas de átomos vecinos para todo un frame ################################
def compute_neighbors_list(atoms, params, G):
    
    for atom_i in atoms:
        
        # Select type of G parameters
        g_index = 0
        for g in range(len(G)):
            if atom_i.symbol == G[g][0]:
                g_index = g
        
        NeighborsList = [] # lista de átomos vecinos del átomo central
        for atom_j in atoms:
            d_ij = distance(atom_i.position, atom_j.position)
            if d_ij != 0 and d_ij < G[g_index][1].Rc:
                NeighborsList.append(atom_j.index) 
        atom_i.neighbor_atoms_list(NeighborsList)
        
    return atoms

# cálculo de las funciones de simetría ###################################################
def features(atoms, params, G, derivatives=False):

    for atom_i in atoms:
        # Parámetros de las FS del átomo central #####################
        G_params = []
        for G_type in G:
            if atom_i.symbol == G_type[0]:
                G_params = G_type[1:]

        # Initialization of G, dGdr and dGdn #########################
        featuresList = np.zeros(len(G_params)) 
        derivativesList = np.zeros((len(G_params),3))
        neighborDerivatives = np.zeros((len(atom_i.neighbors), len(G_params), 3))

        # Compute radial symmetry functions G ########################
        for j in atom_i.neighbors:
            atom_j = atoms[j]
            d_ij = distance(atom_i.position, atom_j.position)
            n_index = 0
            for g in G_params: 
                if g.type == 'radial':
                    if atom_j.symbol == g.j_atom:
                        fc = cut_function(d_ij, g.Rc, params['cut_function_type'])
                        exp = - g.eta * np.power(d_ij - g.Rs, 2)
                        exp = np.exp(exp)
                        #### radial feature G #####################
                        featuresList[g.index] += exp * fc

                        if derivatives:
                            dfc = cut_function_derivative(d_ij, g.Rc, params['cut_function_type'])
                            dGdr = (1/d_ij) * exp * (dfc - 2.0 * g.eta * (d_ij-g.Rs) * fc)
                            dGdr = dGdr * (atom_i.position - atom_j.position)
                            #### derivative dGdr ######################
                            derivativesList[g.index] += dGdr
                            #### derivative dGdn ######################
                            neighborDerivatives[n_index][g.index] -= dGdr
            n_index += 1


        # Compute angular symmetry functions G ########################
        if params['angular_sf'] == 'wide':
            neigh_ind = 0
            for j in atom_i.neighbors[:-1]:
                atom_j = atoms[j]
                nj_index = 0
                neigh_ind += 1
                d_ij = distance(atom_i.position, atom_j.position)
                for k in atom_i.neighbors[neigh_ind:]:
                    atom_k = atoms[k]
                    nk_index = 0
                    d_ik = distance(atom_i.position, atom_k.position)
                    for g in G_params:
                        if g.type == 'angular':
                            condition1 = (atom_j.symbol == g.j_atom and atom_k.symbol == g.k_atom)
                            condition2 = (atom_j.symbol == g.k_atom and atom_k.symbol == g.j_atom)
                            if condition1 or condition2:
                                d_ij = distance(atom_i.position, atom_j.position)
                                d_ik = distance(atom_i.position, atom_k.position)
                                fc_ij = cut_function(d_ij, g.Rc, params['cut_function_type'])
                                fc_ik = cut_function(d_ik, g.Rc, params['cut_function_type'])
                                fc    = fc_ij * fc_ik
                                cosTheta = np.dot(atom_i.position - atom_j.position, 
                                                  atom_i.position - atom_k.position) / (d_ij * d_ik)
                                f_cos = 1.0 + g.lamnda * cosTheta
                                exp = - g.eta * ( (d_ij-g.Rs)*(d_ij-g.Rs) + (d_ik-g.Rs)*(d_ik-g.Rs) )
                                exp = np.exp(exp)
                                if f_cos == 0:
                                    exp = 0
                                else:
                                    f_cos = np.power(f_cos, g.zeta)
                                two_factor = np.power(2, 1 - g.zeta)
                                #### angular feature G ##################################################
                                featuresList[g.index] += two_factor * f_cos * exp * fc

                                if derivatives:
                                    # Derivatives ###########################################################
                                    df_cos = np.power(1.0 + g.lamnda * cosTheta, g.zeta - 1)
                                    dfc_ij = cut_function_derivative(d_ij, g.Rc, params['cut_function_type'])
                                    dfc_ik = cut_function_derivative(d_ik, g.Rc, params['cut_function_type'])
                                    coeff = two_factor * df_cos * exp
                                    lz = g.lamnda * g.zeta
                                    lam_cos = 1.0 + g.lamnda * cosTheta
                                    der_ij = coeff*(fc*( lz/(d_ij*d_ik) - lz*cosTheta/(d_ij*d_ij) - 2*g.eta*lam_cos*(d_ij-g.Rs)/d_ij + fc_ik*dfc_ij*lam_cos/d_ij ))
                                    der_ik = coeff*(fc*( lz/(d_ij*d_ik) - lz*cosTheta/(d_ik*d_ik) - 2*g.eta*lam_cos*(d_ik-g.Rs)/d_ik + fc_ij*dfc_ik*lam_cos/d_ik ))
                                    der_jk = lz * fc / (d_ij*d_ik) 
                                    r_ij = atom_j.position - atom_i.position
                                    r_ik = atom_k.position - atom_i.position
                                    r_jk = r_ik - r_ij
                                    der_rij = der_ij * r_ij
                                    der_rik = der_ik * r_ik
                                    der_rjk = der_jk * r_jk

                                    #### atom-i derivative dGdr #############################################
                                    derivativesList[g.index] += der_rij + der_rik                                            
                                    #### atom-j derivative dGdn #############################################
                                    neighborDerivatives[nj_index][g.index] -= der_rij + der_rjk
                                    #### atom-k derivative dGdn #############################################
                                    neighborDerivatives[nk_index][g.index] -= der_rik - der_rjk                                            
                    nj_index += 1
                    nk_index += 1   


        # Compute angular symmetry functions G ########################
        if params['angular_sf'] == 'narrow':
            neigh_ind = 0
            for j in atom_i.neighbors[:-1]:
                atom_j = atoms[j]
                nj_index = 0
                neigh_ind += 1
                d_ij = distance(atom_i.position, atom_j.position)
                for k in atom_i.neighbors[neigh_ind:]:
                    atom_k = atoms[k]
                    nk_index = 0
                    d_ik = distance(atom_i.position, atom_k.position)
                    for g in G_params:
                        if g.type == 'angular':
                            condition1 = (atom_j.symbol == g.j_atom and atom_k.symbol == g.k_atom)
                            condition2 = (atom_j.symbol == g.k_atom and atom_k.symbol == g.j_atom)
                            if condition1 or condition2:
                                
                                if condition1 or condition2:
                                    d_ij = distance(atom_i.position, atom_j.position)
                                    d_ik = distance(atom_i.position, atom_k.position)
                                    d_jk = distance(atom_j.position, atom_k.position)
                                    fc_ij = cut_function(d_ij, g.Rc, params['cut_function_type'])
                                    fc_ik = cut_function(d_ik, g.Rc, params['cut_function_type'])
                                    fc_jk = cut_function(d_jk, g.Rc, params['cut_function_type'])
                                    fc    = fc_ij * fc_ik * fc_jk
                                    cosTheta = np.dot(atom_i.position - atom_j.position, 
                                                      atom_i.position - atom_k.position) / (d_ij * d_ik)
                                    f_cos = 1.0 + g.lamnda * cosTheta
                                    exp = - g.eta * ( (d_ij-g.Rs)*(d_ij-g.Rs) + (d_ik-g.Rs)*(d_ik-g.Rs) )
                                    exp = np.exp(exp)
                                    if f_cos >= 0:
                                        exp = 0
                                    else:
                                        f_cos = np.power(f_cos, g.zeta)
                                    two_factor = np.power(2, 1 - g.zeta)
                                    #### angular feature G ##################################################
                                    featuresList[g.index] += two_factor * f_cos * exp * fc

                                    if derivatives:
                                        # Derivatives ###########################################################
                                        df_cos = np.power(1.0 + g.lamnda * cosTheta, g.zeta - 1)
                                        dfc_ij = cut_function_derivative(d_ij, g.Rc, params['cut_function_type'])
                                        dfc_ik = cut_function_derivative(d_ik, g.Rc, params['cut_function_type'])
                                        coeff = two_factor * df_cos * exp
                                        lz = g.lamnda * g.zeta
                                        lam_cos = 1.0 + g.lamnda * cosTheta
                                        der_ij = coeff*(fc*( lz/(d_ij*d_ik) - lz*cosTheta/(d_ij*d_ij) - 2*g.eta*lam_cos*(d_ij-g.Rs)/d_ij + fc_ik*dfc_ij*lam_cos/d_ij ))
                                        der_ik = coeff*(fc*( lz/(d_ij*d_ik) - lz*cosTheta/(d_ik*d_ik) - 2*g.eta*lam_cos*(d_ik-g.Rs)/d_ik + fc_ij*dfc_ik*lam_cos/d_ik ))
                                        der_jk = lz * fc / (d_ij*d_ik) 
                                        r_ij = atom_j.position - atom_i.position
                                        r_ik = atom_k.position - atom_i.position
                                        r_jk = r_ik - r_ij
                                        der_rij = der_ij * r_ij
                                        der_rik = der_ik * r_ik
                                        der_rjk = der_jk * r_jk

                                        #### atom-i derivative dGdr #############################################
                                        derivativesList[g.index] += der_rij + der_rik
                                        #### atom-j derivative dGdn #############################################
                                        neighborDerivatives[nj_index][g.index] -= der_rij + der_rjk
                                        #### atom-k derivative dGdn #############################################
                                        neighborDerivatives[nk_index][g.index] -= der_rik - der_rjk                                            
                    nj_index += 1
                    nk_index += 1                                   


        atom_i.features(featuresList)
        if derivatives:
            atom_i.features_derivative(derivativesList)
            atom_i.neighbor_derivatives(neighborDerivatives)


    return atoms


            
# E N T R E N A M I E N T O #-----------------------------------------------------------#

# procesamiento de los rasgos para la red ###############################################
def process_training_data(frames, params, G, save=False):
    
    if params['model_path'] != 'none':
        return
       
    if save:
        if (os.path.exists('outputs/function.data')):
            os.remove('outputs/function.data')
        f = open('outputs/function.data', 'a')
    
    data = []
    temp_data = []
    temp_outs = []
    for i in range(len(frames)):
        
        # Obtención de los rasgos #################################
        temp_features = features(frames[i][:-1], params, G)
        print('Frame {} features: Done'.format(i))
        
        # Creación de function.data ###############################
        if save:
            f.write('{} \n'.format(len(temp_features)))
            for atom in temp_features:
                f.write('{} \t'.format(atom.number))                
                for g in atom.G:
                    f.write('{:.10f} \t'.format(g))
                f.write('\n')
            for j in range(4):
                f.write(frames[i][-1][j] + '\t')
            f.write('\n')
        
        # Preparación de los datos para el entrenamiento ############################
        inputs = []
        for atype in range(len(G)):
            features_per_type = []
            for atom in temp_features:
                if atom.symbol == G[atype][0]:
                    if len(features_per_type) == 0:
                        features_per_type.append(np.array(atom.G))
                    else:
                        features_per_type = np.vstack((features_per_type, np.array(atom.G)))
            if len(features_per_type) == 1:
                features_per_type = np.array(features_per_type)
                
            inputs.append(features_per_type)
            
            if i == 0:
                #print(type(temp_data))
                temp_data.append(features_per_type)
            else:
                #print(type(temp_data))
                temp_data[atype] = np.vstack((temp_data[atype], features_per_type))
                

        output = frames[i][-1][1].astype(np.float)
        temp_outs.append(output)
        temp = Frame(inputs,output)
        data.append(temp)

          
    if save:
        f.close() 
        
    # Shifting and Scaling #########################################################
    
    # input data #
    max_inputs   = [np.max(temp_data_atom, axis = 0) for temp_data_atom in temp_data]
    min_inputs   = [np.min(temp_data_atom, axis = 0) for temp_data_atom in temp_data]
    mean_inputs  = [np.mean(temp_data_atom, axis = 0) for temp_data_atom in temp_data]
    std_inputs   = np.array([np.std(temp_data_atom, axis = 0) for temp_data_atom in temp_data])
    scale_inputs = [a - b for a,b in list(zip(max_inputs, min_inputs))]
    # output data #
    temp_outs    = np.array(temp_outs)
    max_outputs  = np.max(temp_outs)
    min_outputs  = np.min(temp_outs)
    mean_outputs = np.mean(temp_outs)
    std_outputs  = np.std(temp_outs)
    scale_outputs = np.max( [max_outputs - min_outputs, 0.0000001] )

    out_min = 1000    # Shifted/Scaled output data min threshold 
    out_max = -1000   # Shifted/Scaled output data max threshold

    # Apply shifting and scaling
    for datum in data:
        datum.shift_and_scale(mean_inputs, scale_inputs, mean_outputs, scale_outputs)
        if datum.get_energy_of_frame() > out_max:
            out_max = datum.get_energy_of_frame()  # Update max threshold
        if datum.get_energy_of_frame() < out_min:
            out_min = datum.get_energy_of_frame()  # Update min threshold

    # Save statistical analysis features
    if (os.path.exists('outputs/scales.out')):
        os.remove('outputs/scales.out')

    g = open('outputs/scales.out', 'ab')
    num_x_points = 200
    np.savetxt( 
        g,
        ['     min        max        average      range       std      range/std   atom  feature '],
        fmt = '%s')
    
    for atype in range(len(G)):
        for feature in range(len(G[atype][1:])):
            # Number of std's contained in range
            times_std_input = scale_inputs[atype][feature] / std_inputs[atype][feature]
            x = [min_inputs[atype][feature],
                 max_inputs[atype][feature],
                 mean_inputs[atype][feature],
                 scale_inputs[atype][feature],
                 std_inputs[atype][feature],
                 times_std_input]
            np.savetxt(g, x, fmt = '%11.8f', newline = ' ') # Save input statistical data
            x = [(atype + 1)]
            np.savetxt(g, x, fmt = '%4d', newline = ' ')    # Save number of atom type
            x = [(feature + 1)]
            np.savetxt(g, x, fmt = '%4d', newline = '\n')   # Save number of feature

    times_std_output = scale_outputs / std_outputs
    x = [min_outputs, max_outputs, mean_outputs, scale_outputs, std_outputs]
    np.savetxt(g, x, fmt = '%11.8f', newline = ' ')         # Save output statistical data
    x = [times_std_output]
    np.savetxt(g, x, fmt = '%11.8f', newline = '\n')
    g.close()

    
    # Create tensors ######################################################################
    # input tensor
    data_x = []
    number_atom_types = len(G)
    for atype in range(number_atom_types):
        data_x.append( np.array([frame.get_input(atype) for frame in data]) )
        
    # Create the target tensor
    data_y = np.array([frame.get_energy_of_frame_as_array() for frame in data])
    
    return data_x, data_y, scale_outputs
    
    
# creación del modelo de la red neuronal ################################################
def create_model(params, G):
    
    # Get some parameters
    atom_types = len(G)
    features   = [len(g[1:]) for g in G]
    layers     = params['neurons_per_layer']
    
    activation_function = getattr(network_activation_functions, params['activation_function'])
    
    np.random.seed(params['random_seed'])
        
    models = []  # Will contain the neural network of each atom type
    inputs = []  # Will contain the input layer of each atom type
    # Create the network architecture
    for i in range(atom_types):    
        # Input layer
        x = Input(shape = (None, features[i]),
                  name = 'Features_of_atoms_in_Wyckoff_site_'+ str(i+1))
        inputs.append(x)
        # Hidden layers
        j = 1
        for layer in layers[:-1]:
            x = Dense(layer,
                      activation = activation_function,
                      kernel_initializer = params['weight_initialization'],
                      bias_initializer = params['bias_initialization'],
                      name = 'Hidden_layer_'+ str(j) +'_for_atoms_in_site_' + str(i+1))(x)
            j = j + 1
        # Output layer
        x = Dense(layers[-1],
                  activation = 'linear',
                  kernel_initializer = params['weight_initialization'],
                  bias_initializer = params['bias_initialization'],
                  name = 'Energy_of_atoms_in_Wickoff_site_'+ str(i+1))(x)
        # This contains the network of a single atom type
        x = My_layer(output_dim = 1,
                     name = 'Total_energy_of_atoms_in_site_'+ str(i+1))(x)
        models.append(x)
    #  The final model with the networks of every atom type    
    final_model = Add( name = 'Neural_network_energy' )(models)
    model = Model(inputs = inputs, outputs = final_model)

    
    if params['optimization_method'] == 'Adagrad':
        model.compile(
            optimizer = optimizers.Adagrad(lr = float(params['learning_rate']),
                                           epsilon = float(params['epsilon']),
                                           decay = float(params['decay_value']),
                                           clipvalue = float(params['clipvalue'])),
            loss = params['loss_function'])
    else:
        pass
    
    print('Creation of artificial neural network model: Done')
    return model

# guardar el costo del entrenamiento ####################################################
def save_loss(history, energy_scale):
    
    # Factor to convert the error-energy units from (Ha/atom) into (meV/atom)
    error_units = energy_scale * 27211.38602
    # Get the error value
    y1 = np.array( history.history['loss'] )
    y2 = np.array( history.history['val_loss'] )
    # Apply the convertion factor
    y1 = np.sqrt(y1) * error_units
    y2 = np.sqrt(y2) * error_units
    
    # Save the training loss and validation loss 
    if ( os.path.exists('outputs/loss.out') ):
        os.remove('outputs/loss.out')
    if ( os.path.exists('outputs/val_loss.out') ):
        os.remove('outputs/val_loss.out')
        
    f = open('outputs/loss.out', 'ab')
    f1 = open('outputs/val_loss.out', 'ab')
    np.savetxt( f, y1, fmt = '% 12.6f', newline = '\n' )
    np.savetxt( f1, y2, fmt = '% 12.6f', newline = '\n' )
    f.close()
    f1.close()
    
    print('Saving training loss and validation loss: Done')
    return y1, y2

# genera una gráfica con el costo del entrenamiento #####################################
def plot_loss(y1, y2, model):
    
    plot_model(model,
               to_file = 'outputs/nn_model.png',
               show_shapes = True,
               show_layer_names = True)
    fig = plt.figure()
    plt.plot(y1)
    plt.plot(y2)
    plt.ylabel('RMS Error (meV/atom)')
    plt.xlabel('Epoch')
    plt.legend( ['train', 'test'], loc = 'upper right')
    plt.savefig('outputs/plot.png')
    
    return

# genera archivos con los pesos optimizados #############################################
def get_weights_file(model, params, G):
    
    # Get the weights
    weights = model.get_weights()
    atomic_numbers = params['atomic_number']
    layers = [int(l) for l in params['neurons_per_layer']]
    
    for i in range(len(atomic_numbers)):
        count = 1
        # Write the header of the file    
        with open('outputs/weights.%03d.data'%( atomic_numbers[i] ), 'w' ) as f:
            f.write( format('#', '#>80') + '\n')
            f.write( '# Neural network connection values (weights and biases).\n' )
            f.write( format('#', '#>80') + '\n')
            f.write( '#' + format('connection', '>24') )
            f.write( format('t', '>7') + format('index', '>7') + format('l_s', '>7') +
                     format('n_s', '>7') + format('l_e', '>7') + format('n_e', '>7') + '\n')
            f.write( format('#', '#>80') + '\n')
            
            for j in range(len(layers)):
                b = (i * 2) + (j * 2 * len(atomic_numbers))
                # Write the weights per layer
                for k in range( weights[b].shape[0] ):
                    for l in range(weights[b].shape[1] ):
                        f.write( "{:25.16e}".format(weights[b][k][l]) )
                        f.write( format('a', '>7') )           # t
                        f.write( format(count, '>7') )         # index
                        f.write( format(j, '>7') )             # l_s
                        f.write( format(k+1, '>7') )           # n_s
                        f.write( format(j+1, '>7') )           # l_e
                        f.write( format(l+1, '>7') )           # n_e
                        f.write('\n')
                        count += 1
                # Write the biases per layer
                for m in range( len(weights[ b + 1 ]) ):
                    f.write( "{:25.16e}".format(weights[b+1][m]) )
                    f.write( format('b', '>7') )               # t
                    f.write( format(count, '>7') )             # index
                    f.write( format(j+1, '>7') )               # l_s
                    f.write( format(m+1, '>7') )               # n_s
                    f.write('\n')  
                    count += 1
            print(f.name)
            f.close        
    return

# entrenamiento de la red neuronal ######################################################
def training(control_file, save):
    
    # read control parameters, SF parameters and initial positions ###############
    params, frames, G = read_data(control_file)
    
    # instantiate atoms and create neighbors lists ###############################
    framesList = []
    for frame in frames:
        AtomsList = []
        for i in range(len(frame[:-1])):
            temp_atom = Atom(i, frame[i]) # átomo central
            for g in range(len(G)):
                if temp_atom.symbol == G[g][0]:
                    temp_atom.atomic_number(params['atomic_number'][g])
                    temp_atom.atomic_mass(params['atomic_mass'][g])
            AtomsList.append(temp_atom)
            
        # Atomic neighbors list ##################################################
        AtomsList = compute_neighbors_list(AtomsList, params, G)
        AtomsList.append(frame[-1])
        framesList.append(AtomsList)
    
    # Get the data tensors and the hyperparameters
    data_x, data_y, scale_outputs = process_training_data(framesList, params, G, save)
    
    # Create the neural network model
    model = create_model(params, G)
    
    # Train the model
    history = model.fit(data_x, data_y,
                        epochs = int(params['number_of_epochs']),
                        batch_size = int(params['minibatch_size']),
                        validation_split = float(params['validation_fraction']) )
    print('Training of neural network: Done')
    
    # Save the model and the weights
    model.save_weights('outputs/weights.h5')
    save_model(model, 'outputs/trained_model.h5')
    
    # Save and plot the training loss and the validation loss
    loss, val_loss = save_loss(history, scale_outputs)
    plot_loss(loss, val_loss, model)
    
    # Generate the weight files
    get_weights_file(model, params, G)
        
    return 


# S I M U L A C I Ó N #-----------------------------------------------------------------#

# derivada de la energía dEdG ###########################################################
def energy_derivative(atoms, params, G):
    
    # Get weights and biases from the trained model
    weights, biases = get_weights(params, G)
    
    for atom in atoms:
    
        # weights and biases
        temp_weights = []
        for weights_type in weights:
            if atom.symbol == weights_type[0]:
                temp_weights = weights_type[1:]
        temp_biases = []
        for bias_type in biases:
            if atom.symbol == bias_type[0]:
                temp_biases = bias_type[1:]
        
        # activation function and its derivative
        activation_function = getattr(network_activation_functions, params['activation_function'])
        activation_function_derivative = getattr(network_activation_functions, params['activation_function'] + '_derivative')
        
        temp_neurons = []
        activation_derivatives = []
        neurons_per_layer = params['neurons_per_layer']
        
        # Input layer -> first hidden layer
        z1 = np.dot(temp_weights[0].T, atom.G) + temp_biases[0]
        a1 = activation_function(z1)
        da1 = activation_function_derivative(z1)
        temp_neurons.append(a1)
        activation_derivatives.append(da1)
        
        # First hidden layer -> Last hidden layer
        for i in range(1, len(neurons_per_layer) - 1):
            temp_z = np.dot(temp_weights[i].T, temp_neurons[i-1]) + temp_biases[i]
            temp_a = activation_function(temp_z)
            temp_da = activation_function_derivative(temp_z)
            temp_neurons.append(temp_a)
            activation_derivatives.append(temp_da)
        
            
        # Last hidden layer -> Output layer
        #z_last = np.dot(temp_weights[-1].T, temp_neurons[-1]) + temp_biases[-1]
        #a_last = network_activation_functions.identity(z_last)
        #da_last = network_activation_functions.identity_derivative(z_last)
        #temp_neurons.append(a_last)
        #activation_derivatives.append(da_last)

        # Compute the derivative
        dEdG = np.zeros(len(atom.G))
        for l in range(len(atom.G)):
            sum_j = 0.0
            for j in range(neurons_per_layer[1]):
                sum_k = 0
                for k in range(neurons_per_layer[0]):
                    sum_k += temp_weights[0][l][k] * activation_derivatives[0][k] * temp_weights[1][k][j]
                    
                sum_j += sum_k * activation_derivatives[1][j] * temp_weights[2][j][0]
            dEdG[l] = sum_j
            
                  
        atom.network_derivative(dEdG)
                    
    return atoms

# cálculo de las fuerzas atómicas #######################################################
def compute_forces(atoms, params, G):

    # atomic neighbors list
    atoms = compute_neighbors_list(atoms, params, G)
    
    # get the features G and its derivatives dGdr
    atoms = features(atoms, params, G, derivatives=True)
    
    # get the energy derivatives dEdG
    atoms = energy_derivative(atoms, params, G)
    
    # calculate all the atomic forces
    for atom_i in atoms:
        temp_forces = np.zeros(3)
        # add central atom contribution
        for l in range(len(atom_i.G)):
            temp_forces -= atom_i.dEdG[l] * atom_i.dGdr[l]
        # add neighbor atoms contributions
        for j in range(len(atoms)):
            atom_j = atoms[j]
            if atom_j.index in atom_i.neighbors:
                for i in range(len(atom_j.neighbors)):
                    if atom_i.index == atom_j.neighbors[i]:
                        n_index = i
                        for l in range(len(atom_j.G)):
                            temp_forces -= atom_j.dEdG[l] * atom_j.dGdn[n_index][l]
        atom_i.force = temp_forces
        
    return atoms

# ejecución de la simulación de dinámica molecular ######################################
def molecular_dynamics(control_file):
    
    # read control parameters, SF parameters and initial positions ###############
    params, positions, G = read_data(control_file)
    
    
    # instantiate atoms ##########################################################
    AtomsList = []
    for i in range(len(positions[0][:-1])):
        temp_atom = Atom(i, positions[0][i]) # átomo central
        for g in range(len(G)):
            if temp_atom.symbol == G[g][0]:
                temp_atom.atomic_number(params['atomic_number'][g])
                temp_atom.atomic_mass(params['atomic_mass'][g])
        AtomsList.append(temp_atom)
        
    # MD simulation ##############################################################
    dt = float(params['timestep'])
    
    with open('outputs/traj.data', 'w') as f:
        
        for it in range(1,params['iterations']+1):
            f.write('{}\n'.format(it))

            # update positions ###################################################
            f_t = []
            for atom in AtomsList:
                f_t.append(atom.force)   
                atom.position = atom.position + atom.velocity*dt + atom.force/2/atom.mass*dt*dt
                r = atom.position
                f.write('{}\t'.format(atom.index))
                f.write('{}\t'.format(r[0]))
                f.write('{}\t'.format(r[1]))
                f.write('{}\n'.format(r[2]))

            # update forces ######################################################
            AtomsList = compute_forces(AtomsList, params, G)

            # update velocity ####################################################
            for i in range(len(AtomsList)):
                AtomsList[i].velocity = AtomsList[i].velocity  + (AtomsList[i].force + f_t[i])/2/AtomsList[i].mass*dt

            print('Iteration {}/{}: Done'.format(it, params['iterations']))
     
    return 



# **************************************************************************** #    
# **************************************************************************** # 


if args['file'] is not None:
    if args['mode'] == 'train':
        training(args['file'], args['output'])
    elif args['mode'] == 'dynamics':
        molecular_dynamics(args['file'])
    