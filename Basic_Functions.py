import matplotlib.pyplot as plt
import matplotlib.style as style
import uuid
import os
from scipy.interpolate import make_interp_spline, BSpline

from mpl_toolkits.mplot3d import Axes3D


style.use('ggplot')

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sb
import glob
import re
from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy.stats import pearsonr


import copy


class RBM_NETWORK_BUILTING(object):

    def __init__(self, size_input, size_hidden, learning_rate = 0.1, alfa = 0.0, sparsety = 0.5, K = 0):

        # LAYERS INIT
        self.size_hidden = size_hidden
        self.size_input = size_input
        self.input = np.zeros((self.size_input))
        self.rec_input = np.zeros((self.size_input))
        self.hidden_output = np.zeros((self.size_hidden))

        # WEIGHTS INIT

        self.input_weights = np.zeros((self.size_input, self.size_hidden))
        self.input_update_weights = np.zeros((self.size_input, self.size_hidden))
        self.input_update_weights_prev = np.zeros((self.size_input, self.size_hidden))
        self.cumulated_input_hidden_weights = np.zeros((self.size_input, self.size_hidden))

        # BIASES

        # inputs_biases
        self.cumulated_bias_input_weights = np.zeros((1, self.size_hidden))
        self.bias_inputs_update_weights_prev = np.zeros((1, self.size_hidden))
        self.bias_inputs_weights = np.zeros((1, self.size_hidden))

        #hidden_biases
        self.cumulated_bias_hidden_weights = np.zeros((1, self.size_input))
        self.bias_hidden_update_weights_prev = np.zeros((1,self.size_input))
        self.bias_hidden_weights = np.zeros((1, self.size_input))

        #LEARNING PARAMETERS

        self.K = K
        self.sparsety = sparsety
        self.learning_rate = learning_rate
        self.alfa = alfa
        self.Max_epoc_R = 0
        self.ideal_actions_batch = []


#   FUNCTIONS FOR LOADING IMAGES, GETTING INPUTS

def Load_images(input_size):

    '''

    This function loads the inputs image from default paths. Depending on the lenght of input_size of network (lenght of input vector of net) the function
    chooses a specific file to load (the original file of '28 x 28 x 3' or the compact version computed by RBM thaT exectues a dim. reduction) and makes it
    a simple matrix (is a npz file).

    args:

        input_size: lenght of input vector of network

    return:

        inputs matrix

    '''

    # original inputs=  path =  ".\\ToLoadFiles\\RBM_ALL_MEASURE_polygons.npz"

    try:

        if input_size == (28*28*3): # ORIGINAL INPUTS

            Input_file_Path = '.\\IMAGES_DATASET.npz'

            originals_matrices = np.load(Input_file_Path)

            # UNZIP

            originals_matrices = [originals_matrices[key] for key in originals_matrices]
            originals_matrices = originals_matrices[0]

        else: # COMPACTED INPUTS (DIM. REDUCTION EXECUTED BY RBM)

            Input_file_Path = '.\\Weights_layers_activations\\Basic_enviroment\\Teste\\BSINGLE_HIDDEN_OUTPUT_2352_' + str(input_size) + str('.npy')

            originals_matrices = np.load(Input_file_Path)

    except:

        print(' NO INPUT FILE FOUND! ')

    return originals_matrices

def Get_input (Input_):

    '''

    This function take an Input_ and assign it to a variable Input.

    args:

        Input_: input (no manipulation)

    return:

        Input_: input (no manipulation)


    '''

    Input = Input_

    Original_Input = copy.deepcopy(Input)

    return Input

#   VARIABLES MANIPULATION

def Unison_shuffle(array_a, array_b):

    '''

        This function executes a shuffle reordering on two arrays (the same reordering is applied to each array to maintain
        the correspondences between each raw of both arrays)

        args:

            - array_a: first array to reorder
            - array_b: second array to reorder

        return:

            - array_a_shuffled: first array reordered
            - array_b_shuffled: second array reordered


    '''

    array_a_shuffled, array_b_shuffled = shuffle(array_a, array_b, random_state=0)

    return array_a_shuffled, array_b_shuffled


def sigmoid(x):

    '''
        This function executes a sigmoidal trasformation

    agrs:

        - x: variable to trasform

    return:

        - x_trasf: trasformed variable
    '''

    x_transf = 1.0/(1.0 + np.exp(-x))

    return x_transf

def natural_sort(l):

    '''

        This function executes a natural ordering.

        args:

            - l: list to order
        return:

            - l: reordered list
    '''

    convert = lambda text: int(text) if text.isdigit() else text.lower()

    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

    return sorted(l, key = alphanum_key)


# SAVE/LOAD/INITIALIZE WEIGHTS AND TRAINING DATA

def Random_initialization_weights(size_input, size_hidden):

    '''

    This function create the weights matrices (random valueS)

    args:

        size_input: lenght of input vector of network
        size_hidden: lenght of hidden vector of network

    return:

        weights of net (Weight_input)
        weights of bias_1 (Weights_bias_inputs_to_hidden)
        weights of bias_2 (Weights_bias_inputs_to_hidden)

    '''


    Weight_inputs = np.random.uniform(-0.01, 0.01, (size_input, size_hidden))

    Weights_bias_inputs_to_hiddens = np.random.uniform(-0.01, 0.01,(1, size_hidden))

    Weights_bias_hiddens_to_inputs = np.random.uniform(-0.01, 0.01,(1, size_input))


    return Weight_inputs, Weights_bias_inputs_to_hiddens, Weights_bias_hiddens_to_inputs

def save_weights(Weight_inputs, Weights_bias_inputs_to_hiddens, Weights_bias_hiddens_to_inputs, Hidden_output, Folder = ''):

    '''

    This function saves weights matrices on the basis of network topology (input size and output size) and paths

    args:

        Weight_inputs: net weights
        Weights_bias_inputs_to_hiddens: input_to_hidden bias (bias 1) weights
        Weights_bias_hiddens_to_inputs: hidden_to_input bias (bias 2) weights
        Hidden_output: Activation vector of Hiddens (Hidden_output)

    return:

        No return (npy files into folder)

    '''

    size_input = Weight_inputs.shape[0]

    size_hidden = Weight_inputs.shape[1]

    if Folder == '':

        Folder = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

    np.save(Folder + str('RBM_WEIGHTS_') + str(size_input) + str('_') + str(size_hidden), Weight_inputs)

    np.save(Folder + str('RBM_WEIGHTS_BIAS_INPUT_TO_HIDDEN_') + str(size_input) + str('_') + str(size_hidden), Weights_bias_inputs_to_hiddens)

    np.save(Folder + str('RBM_WEIGHTS_BIAS_HIDDEN_TO_INPUT_') + str(size_input) + str('_') + str(size_hidden), Weights_bias_hiddens_to_inputs)

    np.save(Folder + str('SINGLE_HIDDEN_OUTPUT_') + str(size_input) + str('_') + str(size_hidden), Hidden_output)

    #print(" WEIGHTS SAVED ")

def Load_weights(size_input, size_hidden, Folder = ''):

    '''

    This function loads pre_saved weights matrices on the basis of network topology (input size and output size) and paths

    args:

        size_input: lenght of input vector of network
        size_hidden: lenght of hidden vector of network

    return:

        weights of net (Weight_inputs)
        weights of bias_1 (Weights_bias_inputs_to_hiddens)
        weights of bias_2 (Weights_bias_hiddens_to_inputs)
        Activation vector of Hiddens (Hidden_output)

    '''

    if Folder == '':

        Folder = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

    Weight_inputs = np.load(Folder + str('RBM_WEIGHTS_') + str(size_input) + str('_') + str(size_hidden) + str('.npy'))

    Weights_bias_inputs_to_hiddens = np.load(Folder + str('RBM_WEIGHTS_BIAS_INPUT_TO_HIDDEN_') + str(size_input) + str('_') + str(size_hidden) + str('.npy'))

    Weights_bias_hiddens_to_inputs = np.load(Folder + str('RBM_WEIGHTS_BIAS_HIDDEN_TO_INPUT_') + str(size_input) + str('_') + str(size_hidden) + str('.npy'))

    #Hidden_output = np.load(Folder + str('SINGLE_HIDDEN_OUTPUT_') + str(size_input) + str('_') + str(size_hidden) + str('.npy'))

    return Weight_inputs, Weights_bias_inputs_to_hiddens, Weights_bias_hiddens_to_inputs

def save_ideals(ideals_batch, ideals, topology, learning_modality, save_path = ''):

    '''


    This function saves ideals (in case di Supervised learning) depending on topology

    args:

        ideals_batch: ideal actions of all batch (64)

        ideals: four prototypical actions

        learning_modality: modality of training (SL or RL)

        topology: topology of network

    return:

        No return (npy files into folder)

    '''

    if save_path == '':

        Folder = '.\\Weights_layers_activations\\Tested\\'

    else:

        Folder = copy.deepcopy(save_path)

    if len(topology) == 1:

        np.save(Folder + str('Ideals_batch') + str('_') + str(topology[0][0].shape[0]) + str('_') + str(topology[0][0].shape[1]) + str('_') + str(learning_modality), ideals_batch)

        np.save(Folder + str('Ideals') + str('_') + str(topology[0][0].shape[0]) + str('_') + str(topology[0][0].shape[1]) + str('_') + str(learning_modality), ideals)

    else:

        np.save(Folder + str('Ideals_batch') + str('_') + str(topology[0][0].shape[0]) + str('_') + str(topology[0][0].shape[1]) + str('_') + str(topology[1][0].shape[1]),
            ideals_batch)

        np.save(Folder + str('Ideals') + str('_') + str(topology[0][0].shape[0]) + str('_') + str(topology[0][0].shape[1]) + str('_') + str(topology[1][0].shape[1]),
                ideals)

def load_ideals(topology, learning_modality):
    '''

        This function loads pre_saved ideal actions in case of supervised learning depending on topology

        args:

            learning_modality: modality of training (SL or RL)

            topology: topology of network



        return:

            ideals_batch: ideal actions of all batch (64)

            ideals: four prototypical actions

    '''


    Folder = '.\\Weights_layers_activations\\Tested\\'

    if len(topology) == 1:


        ideals_batch = np.load(Folder + str('Ideals_batch') + str('_') + str(topology[0].shape[0]) + str('_') + str(topology[0].shape[1]) + str('_') + str(learning_modality) + str('.npy'))

        ideals = np.load(Folder + str('Ideals') + str('_') + str(topology[0].shape[0]) + str('_') + str(topology[0].shape[1]) + str('_') + str(learning_modality)  + str('.npy'))

    else:

        ideals_batch = np.load(Folder + str('Ideals_batch') + str('_') + str(topology[0].shape[0]) + str('_') + str(
            topology[0].shape[1]) + str('_') + str(topology[1].shape[1]) + str('.npy'))

        ideals = np.load(
            Folder + str('Ideals') + str('_') + str(topology[0].shape[0]) + str('_') + str(topology[0].shape[1]) + str('_') + str(topology[1].shape[1]) + str(
                '.npy'))


    return ideals, ideals_batch

def Initialize_variabiles_RBM(learning_modality = 'UL'):


    epoc = 0
    batch_single = 0
    Weight_inputs_update_prev = 0
    Weights_bias_inputs_to_hiddens_update_prev = 0
    Weights_bias_hiddens_to_inputs_update_prev = 0

    Rec_Epocs = []
    Hiddens_Activ_Epocs_each_input = []

    Errors_Epocs = []
    STDs_Errors_Epocs = []
    Errors_Epocs_each_input = []

    Weights_SUM_Epocs_each_input = []
    Weights_SUM_Epocs_bias_1_each_input = []
    Weights_SUM_Epocs_bias_2_each_input = []

    Weights_SUM_Epocs_input = []
    Weights_SUM_Epocs_bias_1 = []
    Weights_SUM_Epocs_bias_2 = []

    if learning_modality == 'RL':

        Max_R_prev = 0

        Reinforce_Epocs = []
        Reinforce_Epocs_each_input = []
        STDs_Reinforce_Epocs = []

        Surprise_Epocs = []
        Surprise_Epocs_each_input = []
        STDs_Surprise_Epocs = []


        Accuracy_Epocs = []
        STDs_Accuracy_Epocs = []
        Accuracy_Epocs_each_input = []

    if learning_modality == 'SL':


        Accuracy_Epocs = []
        STDs_Accuracy_Epocs = []
        Accuracy_Epocs_each_input = []





    if learning_modality == 'UL':

        return batch_single, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev,\
        Rec_Epocs, Hiddens_Activ_Epocs_each_input, Errors_Epocs, STDs_Errors_Epocs, Errors_Epocs_each_input, Weights_SUM_Epocs_each_input,\
        Weights_SUM_Epocs_bias_1_each_input, Weights_SUM_Epocs_bias_2_each_input, Weights_SUM_Epocs_input, Weights_SUM_Epocs_bias_1,\
        Weights_SUM_Epocs_bias_2, epoc

    elif learning_modality == 'RL':



        return batch_single, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev,\
        Rec_Epocs, Hiddens_Activ_Epocs_each_input, Errors_Epocs, STDs_Errors_Epocs, Errors_Epocs_each_input, Weights_SUM_Epocs_each_input,\
        Weights_SUM_Epocs_bias_1_each_input, Weights_SUM_Epocs_bias_2_each_input, Weights_SUM_Epocs_input, Weights_SUM_Epocs_bias_1,\
        Weights_SUM_Epocs_bias_2, Max_R_prev, Reinforce_Epocs, Reinforce_Epocs_each_input, STDs_Reinforce_Epocs, Surprise_Epocs,\
        Surprise_Epocs_each_input, STDs_Surprise_Epocs, Accuracy_Epocs, STDs_Accuracy_Epocs, Accuracy_Epocs_each_input, epoc


    elif learning_modality == 'SL':

        return batch_single, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev,\
        Rec_Epocs, Hiddens_Activ_Epocs_each_input, Errors_Epocs, STDs_Errors_Epocs, Errors_Epocs_each_input, Weights_SUM_Epocs_each_input,\
        Weights_SUM_Epocs_bias_1_each_input, Weights_SUM_Epocs_bias_2_each_input, Weights_SUM_Epocs_input, Weights_SUM_Epocs_bias_1,\
        Weights_SUM_Epocs_bias_2, Accuracy_Epocs, STDs_Accuracy_Epocs, Accuracy_Epocs_each_input, epoc

def Initialize_variabiles_RBM_K():

    Rec_inputs_on_hidden = []
    Rec_inputs_on_visible = []

    Hidden_Activations = []

    Rec_Errors_on_hidden = []
    Rec_Errors_on_visible = []


    Weights_AVG = []
    Weights_AVG_bias_to_hidden = []
    Weights_AVG_bias_to_visible = []


    return  [Rec_inputs_on_hidden, Rec_inputs_on_visible,
             Hidden_Activations,
             Rec_Errors_on_hidden, Rec_Errors_on_visible,
             Weights_AVG, Weights_AVG_bias_to_hidden, Weights_AVG_bias_to_visible]

def Initialize_variabiles_DBN(learning_modality):

    # INIT VARIABLES %%%%%%%%%%%%%%%%%
    epoc = 0
    batch_single = 0

    # FIRST RBM %%%%%

    Weight_inputs_update_prev_FIRST = 0
    Weights_bias_inputs_to_hiddens_update_prev_FIRST = 0
    Weights_bias_hiddens_to_inputs_update_prev_FIRST = 0

    Rec_Epocs_FIRST = []
    Hiddens_Activ_Epocs_each_input_FIRST = []

    Errors_Epocs_FIRST = []
    STDs_Errors_Epocs_FIRST = []
    Errors_Epocs_each_input_FIRST = []

    Weights_SUM_Epocs_each_input_FIRST = []
    Weights_SUM_Epocs_bias_1_each_input_FIRST = []
    Weights_SUM_Epocs_bias_2_each_input_FIRST = []

    Weights_SUM_Epocs_input_FIRST = []
    Weights_SUM_Epocs_bias_1_FIRST = []
    Weights_SUM_Epocs_bias_2_FIRST = []

    # SECOND RBM %%%%%

    Weight_inputs_update_prev_SECOND = 0
    Weights_bias_inputs_to_hiddens_update_prev_SECOND = 0
    Weights_bias_hiddens_to_inputs_update_prev_SECOND = 0

    Rec_Epocs_SECOND = []
    Hiddens_Activ_Epocs_each_input_SECOND = []

    Errors_Epocs_SECOND = []
    STDs_Errors_Epocs_SECOND = []
    Errors_Epocs_each_input_SECOND = []

    Weights_SUM_Epocs_each_input_SECOND = []
    Weights_SUM_Epocs_bias_1_each_input_SECOND = []
    Weights_SUM_Epocs_bias_2_each_input_SECOND = []

    Weights_SUM_Epocs_input_SECOND = []
    Weights_SUM_Epocs_bias_1_SECOND = []
    Weights_SUM_Epocs_bias_2_SECOND = []



    if learning_modality == 'RL':
        Max_R_prev = 0
        Max_Acc_prev = 0

        Reinforce_Epocs = []
        Reinforce_Epocs_each_input = []
        STDs_Reinforce_Epocs = []

        Surprise_Epocs = []
        Surprise_Epocs_each_input = []
        STDs_Surprise_Epocs = []

        Accuracy_Epocs = []
        STDs_Accuracy_Epocs = []
        Accuracy_Epocs_each_input = []

    if learning_modality == 'SL':
        Accuracy_Epocs = []
        STDs_Accuracy_Epocs = []
        Accuracy_Epocs_each_input = []



    if learning_modality == 'UL':

        return batch_single, Weight_inputs_update_prev_FIRST, Weights_bias_inputs_to_hiddens_update_prev_FIRST, \
               Weights_bias_hiddens_to_inputs_update_prev_FIRST, Rec_Epocs_FIRST, Hiddens_Activ_Epocs_each_input_FIRST, \
               Errors_Epocs_FIRST, STDs_Errors_Epocs_FIRST, Errors_Epocs_each_input_FIRST, Weights_SUM_Epocs_each_input_FIRST,\
               Weights_SUM_Epocs_bias_1_each_input_FIRST, Weights_SUM_Epocs_bias_2_each_input_FIRST, Weights_SUM_Epocs_input_FIRST, \
               Weights_SUM_Epocs_bias_1_FIRST, Weights_SUM_Epocs_bias_2_FIRST, Weight_inputs_update_prev_SECOND, \
               Weights_bias_inputs_to_hiddens_update_prev_SECOND, Weights_bias_hiddens_to_inputs_update_prev_SECOND, Rec_Epocs_SECOND, \
               Hiddens_Activ_Epocs_each_input_SECOND, Errors_Epocs_SECOND, STDs_Errors_Epocs_SECOND, Errors_Epocs_each_input_SECOND, \
               Weights_SUM_Epocs_each_input_SECOND, Weights_SUM_Epocs_bias_1_each_input_SECOND, Weights_SUM_Epocs_bias_2_each_input_SECOND, \
               Weights_SUM_Epocs_input_SECOND, Weights_SUM_Epocs_bias_1_SECOND, Weights_SUM_Epocs_bias_2_SECOND, epoc

    elif learning_modality == 'RL':



        return batch_single, Weight_inputs_update_prev_FIRST, Weights_bias_inputs_to_hiddens_update_prev_FIRST, \
               Weights_bias_hiddens_to_inputs_update_prev_FIRST, Rec_Epocs_FIRST, Hiddens_Activ_Epocs_each_input_FIRST, \
               Errors_Epocs_FIRST, STDs_Errors_Epocs_FIRST, Errors_Epocs_each_input_FIRST, Weights_SUM_Epocs_each_input_FIRST,\
               Weights_SUM_Epocs_bias_1_each_input_FIRST, Weights_SUM_Epocs_bias_2_each_input_FIRST, Weights_SUM_Epocs_input_FIRST, \
               Weights_SUM_Epocs_bias_1_FIRST, Weights_SUM_Epocs_bias_2_FIRST, Weight_inputs_update_prev_SECOND, \
               Weights_bias_inputs_to_hiddens_update_prev_SECOND, Weights_bias_hiddens_to_inputs_update_prev_SECOND, Rec_Epocs_SECOND, \
               Hiddens_Activ_Epocs_each_input_SECOND, Errors_Epocs_SECOND, STDs_Errors_Epocs_SECOND, Errors_Epocs_each_input_SECOND, \
               Weights_SUM_Epocs_each_input_SECOND, Weights_SUM_Epocs_bias_1_each_input_SECOND, Weights_SUM_Epocs_bias_2_each_input_SECOND, \
               Weights_SUM_Epocs_input_SECOND, Weights_SUM_Epocs_bias_1_SECOND, Weights_SUM_Epocs_bias_2_SECOND, Max_R_prev,\
               Reinforce_Epocs, Reinforce_Epocs_each_input, STDs_Reinforce_Epocs, Surprise_Epocs, Surprise_Epocs_each_input, \
               STDs_Surprise_Epocs, Accuracy_Epocs, STDs_Accuracy_Epocs, Accuracy_Epocs_each_input, epoc


    elif learning_modality == 'SL':

        return batch_single, Weight_inputs_update_prev_FIRST, Weights_bias_inputs_to_hiddens_update_prev_FIRST, \
               Weights_bias_hiddens_to_inputs_update_prev_FIRST, Rec_Epocs_FIRST, Hiddens_Activ_Epocs_each_input_FIRST, \
               Errors_Epocs_FIRST, STDs_Errors_Epocs_FIRST, Errors_Epocs_each_input_FIRST, Weights_SUM_Epocs_each_input_FIRST,\
               Weights_SUM_Epocs_bias_1_each_input_FIRST, Weights_SUM_Epocs_bias_2_each_input_FIRST, Weights_SUM_Epocs_input_FIRST, \
               Weights_SUM_Epocs_bias_1_FIRST, Weights_SUM_Epocs_bias_2_FIRST, Weight_inputs_update_prev_SECOND, \
               Weights_bias_inputs_to_hiddens_update_prev_SECOND, Weights_bias_hiddens_to_inputs_update_prev_SECOND, Rec_Epocs_SECOND, \
               Hiddens_Activ_Epocs_each_input_SECOND, Errors_Epocs_SECOND, STDs_Errors_Epocs_SECOND, Errors_Epocs_each_input_SECOND, \
               Weights_SUM_Epocs_each_input_SECOND, Weights_SUM_Epocs_bias_1_each_input_SECOND, Weights_SUM_Epocs_bias_2_each_input_SECOND, \
               Weights_SUM_Epocs_input_SECOND, Weights_SUM_Epocs_bias_1_SECOND, Weights_SUM_Epocs_bias_2_SECOND,\
               Accuracy_Epocs, STDs_Accuracy_Epocs, Accuracy_Epocs_each_input, epoc

def save_training_data_CHUNKED(Number_Single_Chunk_File, Topology, Errors, STDs_Errors, Accuracies = 0, STDs_Accuracies = 0, Errors_OTHER = 0, STDs_Errors_OTHER = 0, save_path = ''):

    '''

            This function saves the training data of NET in case of standard learning (CD) or supervised learning modification.
            It is adapt to save data both of RBM and DBN. In case of DBN there are duplicates for variables, e.g. Errors
            for first RBM (input -> first hidden) and Errors_OTHER for second RBM (first hidden -> second hidden).

            args:

                Topology: topology of network, e.g. [2352, 200, 10]
                Errors: list of means of inputs reconstructions (batch rec_error)
                STDs_Errors: list of standard deviations of....
                Accuracies: list of means of accuracies (batch accuracy)
                STDs_Accuracies: list of standard deviations of....

                save_path: folder path of file


            return:

                    None (Training file into a specific folder)

    '''

    if save_path != '':

        path = save_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'


    if len(Topology) == 2:



        if Accuracies == 0:

            file_name = 'Training_Data_Single_RBM_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_Unsupervised_') + str(Number_Single_Chunk_File)

            np.savez(path + file_name, Rec_errors = Errors, Rec_errors_STD = STDs_Errors)


        else:

            file_name = 'Training_Data_Single_RBM_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_Supervised_') + str(Number_Single_Chunk_File)

            np.savez(path + file_name, Rec_errors = Errors, Rec_errors_STD = STDs_Errors, Acc = Accuracies,
                     Acc_STD = STDs_Accuracies)




    else:

        if Accuracies == 0:

            file_name = 'Training_Data_Whole_DBN_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_')  + str(Topology[2]) + str('_Unsupervised_') + str(Number_Single_Chunk_File)

            np.savez(path + file_name, Rec_errors=Errors, Rec_errors_STD=STDs_Errors, Rec_errors_OTHER = Errors_OTHER,
                     Rec_errors_OTHER_STD= STDs_Errors_OTHER)


        else:

            file_name = 'Training_Data_Whole_DBN_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_')  + str(Topology[2]) + str('_Supervised_') + str(Number_Single_Chunk_File)

            np.savez(path + file_name, Rec_errors = Errors, Rec_errors_STD = STDs_Errors, Acc = Accuracies,
                     Acc_STD = STDs_Accuracies, Rec_errors_OTHER = Errors_OTHER, Rec_errors_OTHER_STD= STDs_Errors_OTHER)

def load_training_data_JOIN(Topology, learning_modality, load_path = ''):

    '''

            This function loads the training data of NET in case of standard learning (CD) or supervised learning modification.
            It is adapt to load data both of RBM and DBN. In case of DBN there are duplicates for variables, e.g. Errors
            for first RBM (input -> first hidden) and Errors_OTHER for second RBM (first hidden -> second hidden).

            args:

                Topology: topology of network, e.g. [2352, 200, 10]
                Net_depth: depth label 'RBM' or 'DBN' to identify the specific file to load
                earning_modality: learning label 'UL' or 'SL' to identify the specific file
                load_path: folder path of file


            return:

                    None (Training file into a specific folder)

    '''



    if load_path != '':

        path = load_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

    print('Training data processing: START.. \n')

    print('...\n')

    if len(Topology) == 2:


            if learning_modality == 'UL':

                file_name = 'Training_Data_Single_RBM_' + str(Topology[0]) + str('_') + str(Topology[1]) + str('_Unsupervised_*')

                Chunks_list = glob.glob(path + file_name)

                Chunks_list = natural_sort(Chunks_list)

                for (count, file)  in enumerate(Chunks_list):

                    if count == 0:

                        Training_data = np.load(file)

                        Errors_JOINED = Training_data['Rec_errors']

                        STDs_Errors_JOINED = Training_data['Rec_errors_STD']

                    else:

                        Training_data = np.load(file)

                        Errors = Training_data['Rec_errors']

                        STDs_Errors = Training_data['Rec_errors_STD']

                        Errors_JOINED = np.hstack((Errors_JOINED, Errors))

                        STDs_Errors_JOINED = np.hstack((STDs_Errors_JOINED, STDs_Errors))

                print('Training data processing: STOP.. \n')

                return Errors_JOINED, STDs_Errors_JOINED


            if learning_modality == 'SL':

                file_name = 'Training_Data_Single_RBM_' + str(Topology[0]) + str('_') + str(Topology[1])  + str('_Supervised_*')

                Chunks_list = glob.glob(path + file_name)

                Chunks_list = natural_sort(Chunks_list)

                for (count, file)  in enumerate(Chunks_list):

                    if count == 0:

                        Training_data = np.load(file)


                        Errors_JOINED = Training_data['Rec_errors']

                        STDs_Errors_JOINED = Training_data['Rec_errors_STD']

                        Accuracies_JOINED = Training_data ['Acc']

                        STDs_Accuracies_JOINED = Training_data ['Acc_STD']

                    else:

                        Training_data = np.load(file)

                        Errors = Training_data['Rec_errors']

                        STDs_Errors = Training_data['Rec_errors_STD']

                        Accuracies = Training_data['Acc']

                        STDs_Accuracies = Training_data['Acc_STD']

                        Errors_JOINED = np.hstack((Errors_JOINED, Errors))

                        STDs_Errors_JOINED = np.hstack((STDs_Errors_JOINED, STDs_Errors))

                        Accuracies_JOINED = np.hstack((Accuracies_JOINED, Accuracies))

                        STDs_Accuracies_JOINED = np.hstack((STDs_Accuracies_JOINED, STDs_Accuracies))

                print('Training data processing: STOP.. \n')

                return Errors_JOINED, STDs_Errors_JOINED, Accuracies_JOINED, STDs_Accuracies_JOINED


    elif len(Topology) == 3:

        if learning_modality == 'UL':

            file_name = '*Training_Data_Whole_DBN_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_') + str(Topology[2]) + str('_Unsupervised_*')

            Chunks_list = glob.glob(path + file_name)

            Chunks_list = natural_sort(Chunks_list)

            for (count, file) in enumerate(Chunks_list):

                if count == 0:

                    Training_data = np.load(file)

                    Errors_JOINED = Training_data['Rec_errors']

                    STDs_Errors_JOINED = Training_data['Rec_errors_STD']

                    Errors_OTHER_JOINED = Training_data['Rec_errors_OTHER']

                    STDs_Errors_OTHER_JOINED = Training_data['Rec_errors_OTHER_STD']

                else:

                    Training_data = np.load(file)

                    Errors = Training_data['Rec_errors']

                    STDs_Errors = Training_data['Rec_errors_STD']

                    Errors_OTHER = Training_data['Rec_errors_OTHER']

                    STDs_Errors_OTHER = Training_data['Rec_errors_OTHER_STD']

                    Errors_JOINED = np.hstack((Errors_JOINED, Errors))

                    STDs_Errors_JOINED = np.hstack((STDs_Errors_JOINED, STDs_Errors))

                    Errors_OTHER_JOINED = np.hstack((Errors_OTHER_JOINED, Errors_OTHER))

                    STDs_Errors_OTHER_JOINED =  np.hstack((STDs_Errors_OTHER_JOINED, STDs_Errors_OTHER))




            return Errors_JOINED, STDs_Errors_JOINED, Errors_OTHER_JOINED, STDs_Errors_OTHER_JOINED


        if learning_modality == 'SL':


            file_name = '*Training_Data_Whole_DBN_' + str(Topology[0]) + str('_')  + str(Topology[1]) + str('_') + str(Topology[2]) + str('_Supervised_*')

            Chunks_list = glob.glob(path + file_name)

            Chunks_list = natural_sort(Chunks_list)

            for (count, file) in enumerate(Chunks_list):

                if count == 0:

                    Training_data = np.load(file)

                    Errors_JOINED = Training_data['Rec_errors']

                    STDs_Errors_JOINED = Training_data['Rec_errors_STD']

                    Errors_OTHER_JOINED = Training_data['Rec_errors_OTHER']

                    STDs_Errors_OTHER_JOINED = Training_data['Rec_errors_OTHER_STD']

                    Accuracies_JOINED = Training_data['Acc']

                    STDs_Accuracies_JOINED = Training_data['Acc_STD']

                else:

                    Training_data = np.load(file)

                    Errors = Training_data['Rec_errors']

                    STDs_Errors = Training_data['Rec_errors_STD']

                    Errors_OTHER = Training_data['Rec_errors_OTHER']

                    STDs_Errors_OTHER = Training_data['Rec_errors_OTHER_STD']

                    Accuracies = Training_data['Acc']

                    STDs_Accuracies = Training_data['Acc_STD']

                    Errors_JOINED = np.hstack((Errors_JOINED, Errors))

                    STDs_Errors_JOINED = np.hstack((STDs_Errors_JOINED, STDs_Errors))

                    Errors_OTHER_JOINED = np.hstack((Errors_OTHER_JOINED, Errors_OTHER))

                    STDs_Errors_OTHER_JOINED = np.hstack((STDs_Errors_OTHER_JOINED, STDs_Errors_OTHER))

                    Accuracies_JOINED = np.hstack((Accuracies_JOINED, Accuracies))

                    STDs_Accuracies_JOINED = np.hstack((STDs_Accuracies_JOINED, STDs_Accuracies))


            return Errors_JOINED, STDs_Errors_JOINED, Errors_OTHER_JOINED, STDs_Errors_OTHER_JOINED, Accuracies_JOINED, STDs_Accuracies_JOINED

def clean_training_data_folder(path_to_clean):

        '''

            This function clean a specific data folder, i.e. delete al previously saved files.

        args:

            - path_to_clean: folder of path to clean

        return: none

        '''

        Previously_saved_training_data = glob.glob(path_to_clean + str('*'))

        for file in Previously_saved_training_data:
            os.remove(file)



#    SPREAD, RECONSTRUCTION, UPDATE, COMPUTATION OF ERRORS, LAYERS DIMENSIONAL REDUCTIONS FUNCTIONS

def Activation_Hidden_Layer(Input, Weight_inputs, Weights_bias_inputs_to_hiddens, stochasticity = False, learning_rate = False, target_sparsety = False):

        '''

        This function support the spread of RBM both in case of positive and negative product (see Hinton, 2006). Includes the sparsety implementation.

        args:

            Input: simple input
            Weight_inputs: weights matrix between input and hiddens (fully connected)
            Weights_bias_inputs_to_hiddens: weights matrix between bias (always activated with value 1) and hiddens (fully connected)
            learning_rate: learning rate of NET learning
            target_sparsety: level of sparsety you want (e.g. 0.25 corresponds to the activation of a quarter of hiddens units)
            stochasticity: boolean value that allows to discern positive product (with binary stochastic trasformation) and negative product (probabilities values)

        return:

            Hidden_output: spread output
            penalty (only in case of stocastic binary activation): penalty to apply to weights matrix for sparsety implementation


        '''

        Hidden_pot = np.dot(Input, Weight_inputs)

        Hidden_pot = Hidden_pot + Weights_bias_inputs_to_hiddens

        Hidden_output = 1 / (1 + np.exp(-(Hidden_pot)))

        # ii = np.isnan(Hidden_output)  # BARBA-TRICK: IN CASE OF VERY NEGATIVE WEIGHTS, THEY ARE RANSFORMED INTO 0 VALUES)
        # Hidden_output[ii] = 0

        # STOCHASTIC TRANSFORMATION (BINARY VALUES OF 0 OR 1) + IMPLEMENTATION OF SPARSETY (IF NECESSARY)

        if stochasticity == True:

            Hidden_output_probabilities = copy.deepcopy(Hidden_output)

            random_treshold = np.random.random_sample((Hidden_output.shape[0], Hidden_output.shape[1]))  # noise, case


            Hidden_output[Hidden_output > random_treshold] = 1

            Hidden_output[Hidden_output < random_treshold] = 0

            current_sparsety = np.mean(np.sum((Hidden_output), axis= 1) / Hidden_output.shape[1])

            current_update = (current_sparsety - target_sparsety)

            penalty = learning_rate * current_update * 0.1 # I SET K =  0.1

            if current_sparsety <= target_sparsety:

                penalty = 0 # if current sparsety is lower then target sparsety: penalty = 0

            Hidden_output_binary = copy.deepcopy(Hidden_output)

            return Hidden_output_binary, penalty


        else:

            return Hidden_output, 0

def Input_reconstruction(Hidden_output, Weight_inputs, Weights_bias_hiddens_to_inputs):

    '''

        This function support the inverse spread (recostruction of input) of RBM.

        args:

            Weight_inputs: weights of net (Weight_inputs)
            Weights_bias_hiddens_to_inputs: weights of bias_2 (Weights_bias_hiddens_to_inputs)
            Hidden_output: Activation vector of Hiddens (Hidden_output)

        return:

            Reconstructed_input: RBM reconstruction of input (hidden -> visible; see Hinton, 2006)


    '''

    Reconstructed_pot = np.dot(Hidden_output, Weight_inputs.T)

    Reconstructed_pot = Reconstructed_pot + Weights_bias_hiddens_to_inputs

    Reconstructed_input = 1 / (1 + np.exp(-(Reconstructed_pot)))

    # ii = np.isnan(Reconstructed_input)
    # Reconstructed_input[ii] = 0

    return Reconstructed_input

def Potential_update_CD(Input, Activation_Hidden_first, Rec_Input, Activation_Hidden_second, l_rate, alfa = 0, prev_update= 0,
                     prev_bias_1_update = 0, prev_bias_2_update = 0, penalty = 0):
        '''

            This function propose a weights update based on original Contrastive Divergence (CD, Contrastive Divergence; see Hinton, 2006).

            args:

                Input: visible original input
                Activation_Hidden_first: first activation of hidden layer depending on original input
                Rec_Input: visible reconstructed input (inverse spread result)
                Activation_Hidden_second: second activation of hidden layer depending on reconstructed input

                l_rate: learning rate
                alfa: momentum
                penalty: penalty linked to the sparsety implementation (higher penalty value causes a lower number of hidden units activated)

                prev_update: previous update of net weights
                prev_bias_1_update: previous update of input_to_hidden bias (bias 1)
                prev_bias_2_update: previous update of hidden_to_input bias (bias 2)

            return:

                Weight_inputs_update: update of net weights
                Weights_bias_inputs_to_hiddens_update: update of input_to_hidden bias (bias 1)
                Weights_bias_hiddens_to_inputs_update: update of hidden_to_input bias (bias 2)



        '''

        PositiveProduct = np.dot(Input.T, Activation_Hidden_first)

        NegativeProduct = np.dot(Rec_Input.T, Activation_Hidden_second)

        # UPDATE OF NET WEIGHTS

        Weight_inputs_update = l_rate * ((PositiveProduct - NegativeProduct) / Input.shape[0]) + alfa * prev_update

        Weight_inputs_update -=  penalty

        # UPDATE OF BIAS 1 WEIGHTS (INPUT_TO_HIDDEN)

        # Weights_bias_inputs_to_hiddens_update = l_rate / Input.shape[0] * (sum(Activation_Hidden_first) - sum(Activation_Hidden_second)) \
        #                                         + alfa * prev_bias_1_update

        Weights_bias_inputs_to_hiddens_update = (l_rate * ((sum(Activation_Hidden_first) - sum(Activation_Hidden_second))/(Input.shape[0]))) + alfa * prev_bias_1_update

        Weights_bias_inputs_to_hiddens_update -= penalty

        # UPDATE OF BIAS 1 WEIGHTS (HIDDEN_TO_INPUT
        # )
        # Weights_bias_hiddens_to_inputs_update = l_rate / Input.shape[0] * (sum(Input) - sum(Rec_Input)) + alfa * prev_bias_2_update

        Weights_bias_hiddens_to_inputs_update = (l_rate * ((sum(Input) - sum(Rec_Input)) / (Input.shape[0]))) + alfa * prev_bias_2_update


        Weights_bias_hiddens_to_inputs_update -= penalty

        return Weight_inputs_update, Weights_bias_inputs_to_hiddens_update, Weights_bias_hiddens_to_inputs_update

def Effective_update(Weight_inputs, Weights_bias_inputs_to_hiddens, Weights_bias_hiddens_to_inputs,
                     potential_update_weights, potential_update_bias1, potential_update_bias2):
        '''

                    This function applied the weights update to each weights matrix.

                    args:

                        Weight_inputs: net weights
                        Weights_bias_inputs_to_hiddens: input_to_hidden bias (bias 1) weights
                        Weights_bias_hiddens_to_inputs: hidden_to_input bias (bias 2) weights

                        potential_update_weights: update of net weights
                        potential_update_bias1: update of input_to_hidden bias (bias 1)
                        potential_update_bias2: update of hidden_to_input bias (bias 2)

                    return:


                        Weight_inputs: net weights
                        Weights_bias_inputs_to_hiddens: input_to_hidden bias (bias 1) weights
                        Weights_bias_hiddens_to_inputs: hidden_to_input bias (bias 2) weights

                        Weight_inputs_update_prev: (future) previous update of net weights for momentum computation
                        Weights_bias_inputs_to_hiddens_update_prev: (future) previous update of input_to_hidden bias (bias 1)
                                                for momentum computation
                        Weights_bias_hiddens_to_inputs_update_prev: (future) previous update of hidden_to_input bias (bias 2)
                                                for momentum computation


                '''

        Weight_inputs += potential_update_weights

        Weights_bias_inputs_to_hiddens += potential_update_bias1
        potential_update_bias2 = np.reshape(potential_update_bias2, (Weights_bias_hiddens_to_inputs.shape[0], Weights_bias_hiddens_to_inputs.shape[1]))
        Weights_bias_hiddens_to_inputs += potential_update_bias2

        # VARIABLES FOR MOMENTUM COMPUTATION

        Weight_inputs_update_prev = potential_update_weights

        Weights_bias_inputs_to_hiddens_update_prev = potential_update_bias1

        Weights_bias_hiddens_to_inputs_update_prev = potential_update_bias2

        return Weight_inputs, Weights_bias_inputs_to_hiddens, Weights_bias_hiddens_to_inputs, Weight_inputs_update_prev, \
               Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev


def Reconstruction_errors(Original_input, Reconstructed_input):

    '''

        This function computes the recostruction errors of RBM.

        args:

            Original_input: visible original input
            Reconstructed_input: visible reconstructed input (inverse spread result)


        return:

            errors_absolute_value_list = list of errors for each input
            errors_absolute_value_batch = avg reconstruction error for batch
            St_dev_errors_absolute_value_batch = standard deviation of previous variable for batch


    '''


    # RECONSTRUCTION ERRORS

    errors_absolute_value_list = []

    errors_percent_value_list = []

    for inp in range(0, Original_input.shape[0]):

        error_vector = np.abs(Original_input[inp, :] - Reconstructed_input[inp, :])

        errors_absolute_value = error_vector.sum(axis=0) / error_vector.shape[0]

        errors_percent_value = errors_absolute_value * 100

        errors_absolute_value_list.append(errors_absolute_value)

        errors_percent_value_list.append(errors_percent_value)


    # IN CASE OF BATCH LEARNING THE FOLLOWING LINES CORRESPOND TO MEANS AND STDS OF PREVIOUS LINES (MATRICES OF INPUTS X VALUES),

    errors_absolute_value_batch = np.mean(errors_absolute_value_list)

    errors_percent_value_batch = np.mean(errors_percent_value_list)

    St_dev_errors_absolute_value_batch = np.std(errors_absolute_value_list)

    St_dev_errors_percent_value_value_batch = np.std(errors_percent_value_list)


    return errors_absolute_value_list, errors_absolute_value_batch, St_dev_errors_absolute_value_batch

def Reconstruction_errors_K(Original_input, Reconstructed_input):

    '''

        This function computes the recostruction errors of RBM.

        args:

            Original_input: visible original input
            Reconstructed_input: visible reconstructed input (inverse spread result)


        return:

            errors_absolute_value=  error
            errors_percent_value = errors_absolute_value * 100


    '''


    # RECONSTRUCTION ERRORS



    error_vector = np.abs(Original_input- Reconstructed_input)

    errors_absolute_value = error_vector.sum() / error_vector.shape[1]

    errors_percent_value = errors_absolute_value * 100



    return errors_absolute_value,errors_percent_value

def dim_reduction(input, n_components = 3):

        '''

        This function executed a dimensional reduction (e.g. PCA) of a layer activation. This reduction is useful to visualize
        in a 3D plot the layers activations.

        args:

            input: 2D matrix composed by 64 inputs (rows) x units

            n_components: principal components you want to extract

        return:

            input_transf: compressed/reducted version of input variable
            Explained_Var: explained variance by principals components
            Explained_Var_ratio: explained variance ratio

        '''

        PCA_function = PCA(n_components, svd_solver = 'arpack')
        TSNE_function = TSNE(n_components)

        input = np.vstack(input)

        input_transf_PCA = PCA_function.fit_transform(input)

        input_transf_TSNE = TSNE_function.fit_transform(input)

        # PCA DATA

        Explained_Var = PCA_function.explained_variance_
        Explained_Var_ratio = PCA_function.explained_variance_ratio_
        #tot_expl = Explained_Var.cumsum()
        #tot_Explained_Var_ratio = Explained_Var_ratio.cumsum()

        input_transf = copy.deepcopy(input_transf_PCA)


        return input_transf, Explained_Var, Explained_Var_ratio

#   REPRESENTATIONS TESTER FUNCTIONS (PERCEPTRON LINKED TO INTERNAL REPRESENTATIONS)

def tester_weights_init(size_input, size_output = 10):

        '''

        This function initializes the weights of tester.

        args:

            size_input: input layer length (number of units)
            size_output: output layer length (number of units)

        return:

            weights_tester: weights matrix of tester
        '''

        weights_tester = np.random.uniform( - 0.01, 0.01, (size_input, size_output))

        return weights_tester

def tester_ideals_init(size_hidden, salient_feature, number_actions=4):
    '''

    This funtion is a copy of initialization of ideals function for reinforcement learning modiifcation of CD (contrastive divergence).
    In this case the ideals are labels for a tester that use original inputs and biased inputs to learn categories/attributes (utiliy test)

    args:

        size_hidden: length of last hidden layer (output) of tester
        salient_feature: feature that guides the creation of labels (specific actions for colors, form or size)
        number_actions: number of attributes for each feature (e.g. for color are red, green, blue and yellow).
                        It is 4 for default state.

    return:

        ideal_actions_batch: matrix of inputs (64 rows) x number of hidden units (columns).


    '''

    print('I m creating the ideal actions')

    ideal_actions_batch = []

    if size_hidden == 4:

        ideal_actions_batch = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    else:

        while len(ideal_actions_batch) < number_actions:

            new_ideal = np.random.randint(2, size=(1, size_hidden))

            exception = False

            if len(ideal_actions_batch) == 0:

                if (np.sum(new_ideal) > (size_hidden // 2)) or (np.sum(new_ideal) < (size_hidden // 2)):
                    exception = True
            else:

                for comb in ideal_actions_batch:

                    diff = np.sum(abs(comb - new_ideal))

                    if (diff < (size_hidden // 2)) or (np.sum(new_ideal) > (size_hidden // 2)) or (
                            np.sum(new_ideal) < (size_hidden // 2)):
                        exception = True

            if exception == False:
                ideal_actions_batch.append(new_ideal)

                print(str(len(ideal_actions_batch)) + str('° combination added...'))

    print('...Ideal combinations created')
    ideal_actions_batch = np.vstack(ideal_actions_batch)

    if salient_feature == 'color':  # COLOR IDEAL FORMAT

        ideal_actions_batch = np.tile(ideal_actions_batch, (16, 1))

    elif salient_feature == 'form':  # FORM IDEAL FORMAT

        ideal_actions_batch = np.repeat(ideal_actions_batch, 4, axis=0)
        ideal_actions_batch = np.tile(ideal_actions_batch, (4, 1))

    elif salient_feature == 'size':  # SIZE IDEAL FORMAT

        ideal_actions_batch = np.repeat(ideal_actions_batch, 16, axis=0)

    return ideal_actions_batch

def tester_spread_update(input_, expected, weights_tester, learning_rate = 0.001):

        '''

        This function computes the spread of tester.

        args:

            input_: representation to test (original or biased)
            expected: label of specific representation depending on specific attribute
            weights_tester: weights matrix of tester
            learning_rate: learning rate of tester update

        return:

            Output: output activation of tester
            weights_tester: weights matrix of tester
            Loss: prediction error

        '''

        Output_pot = np.dot(input_, weights_tester)
        Output = 1 / (1 + np.exp(-(Output_pot)))

        Loss = expected - Output

        weights_tester += (np.dot(input_.T, Loss) * learning_rate)# + (np.random.uniform( - 0.5, 0.5, (weights_tester.shape[0], weights_tester.shape[1])))

        Loss = Loss**2
        Loss = np.mean(Loss, axis =  1)
        Loss = np.mean(Loss)

        return Output, weights_tester, Loss

def tester_save_data(learning_params, errors, rewards, accuracies, adversary = False, save_path = ''):

    '''

        This function saves the learning data of tester (reinforcement or back-prop perceptron).


        args:
                - errors: errors of backprop (only in case of supervised executor)
                - rewards: rewards x epochs
                - accuracies: accuracies x epochs
                - learning_params: training params of tester, i.e. feature to focus (color, form, size), size represent.,
                                    size of hidden layer of tester,learning rate actor, learning rate critic
                - save_path: path of saved file

                - Adversary: "is this net an adversary of tested net?"

        return: none (file in folder)

    '''

    feature_to_focus = learning_params[0]

    size_input = learning_params[1]

    size_hidden = learning_params[2]

    if adversary:


        identity_label = 'Adversary_net'

    else:


        identity_label = 'Tested_net'

    if size_input == (28 * 28 * 3):

        layer_label = 'visible'

    elif size_input == 200:

        layer_label = 'hidden'

    else:

        layer_label = 'hidden second'

    if len(learning_params) == 5:

        learning_rule = 'REINFORCE'

        file_name = 'Tester_Data_' + str(learning_rule) + str('_') + str(feature_to_focus) + str('_') + str(
            layer_label) + str('_') + str(size_input) + str('_') + str(size_hidden) + \
                    str('_') + str(identity_label)

        np.savez(save_path + file_name, Accuracies= accuracies, Rewards= rewards, Params= learning_params)




    else:

        learning_rule = 'BACKPROP'

        file_name = 'Tester_Data_' + str(learning_rule) + str('_') + str(feature_to_focus) + str('_') + str(
            layer_label) + str('_') + str(size_input) + str('_') + str(size_hidden) + \
                    str('_') + str(identity_label)

        np.savez(save_path + file_name, Losses = errors, Params = learning_params)

def tester_load_data(learning_params, adversary = False, load_path = ''):

    '''

        This function loads the training data of tester.

        args:

            - learning_params: training params of tester, i.e. feature to focus (color, form, size), size represent.,
                                    size of hidden layer of tester,learning rate actor, learning rate critic
            - adversary: identity label (boolean value) to distinguish tested net and adversarial net
            - path: load path


        return:

            - Rewards: rewards x epochs
            - Accuracies: accuracies x epochs
            - Parameters: training params of tester, i.e. feature to focus (color, form, size), size represent.,
                                size of hidden layer of tester,learning rate actor, learning rate critic

    '''

    feature_to_focus = learning_params[0]

    size_input = learning_params[1]

    size_hidden = learning_params[2]




    if adversary:

        identity_label = 'Adversary_net'

    else:

        identity_label = 'Tested_net'

    if size_input == (28 * 28 * 3):

        layer_label = 'visible'

    elif size_input == 200:

        layer_label = 'hidden'

    else:

        layer_label = 'hidden second'

    if len(learning_params) == 5:

        learning_rule = 'REINFORCE'

        # if adversary and layer_label == 'hidden second':
        #
        #     size_input = 150

        file_name = 'Tester_Data_' + str(learning_rule) + str('_') + str(feature_to_focus) + str('_') + str(layer_label) +str('_')\
                    + str(size_input) + str('_') + str(size_hidden) + \
                str('_') + str(identity_label) + str('.npz')

        Training_data = np.load(load_path + file_name)

        Parameters = Training_data['Params']

        Accuracies = Training_data['Accuracies']

        Rewards = Training_data['Rewards']


        return Accuracies, Rewards, Parameters

    else:

        learning_rule = 'BACKPROP'

        # if adversary and layer_label == 'hidden second':
        #
        #     size_input = 150

        file_name = 'Tester_Data_' + str(learning_rule) + str('_') + str(feature_to_focus) + str('_') + str(
            layer_label) + str('_') + str(size_input) + str('_') + str(size_hidden) + \
                    str('_') + str(identity_label) + str('.npz')

        Training_data = np.load(load_path + file_name)

        Parameters = Training_data['Params']

        Errors = Training_data['Losses']

        return Errors, Parameters


#   WINDOWS BUILDER (MANY VISUAL INFORMATIONS REGARDING THE NET ACTIVATION SUCH AS INPUT, REC_INPUT, RFs..)

#   MATRICIAL/GRAPHICAL RECONSTRUCTIONS OF INPUTS AND RECEPTIVE FIELDS

def Inputs_recostructions(Input, Rec_input):

        '''

        This function transform Input/reconstructed input vectors (2352 x 1) to RGB matrices (28 x 28 x 3).

        args:

            Input: visible original input

            Rec_Input: visible reconstructed input (inverse spread result)


        return:

            Input_matix: RGB matrix of visible original input (28 x 28 x 3)

            Rec_input_matrix: RGB matrix of reconstructed input (28 x 28 x 3)


        '''

        image_side = np.sqrt(Input.shape[0] / 3)

        Input_matix = Input.reshape([int(image_side), int(image_side), 3], order='F')

        Rec_input_matrix = Rec_input.reshape([int(image_side), int(image_side), 3], order='F')

        return Input_matix, Rec_input_matrix

def First_Hiddens_RF_Reconstruction(Hidden_Output, Weight_inputs, Weights_bias_hiddens_to_inputs, hiddens_plotted):

    '''

            This function extracts the receptive fields of each hidden unit (first hidden layer) and bias (from hidden to input).
            A receptive field of single hidden unit corresponds to its contribution to the input reconstruction, i.e. the RGB reconstructed input matrix
            that you can see only in case of a recostruction only depending on this single hidden. Using the weights of network the function
            returns RGB matrices (28 x 28 x 3) of each hidden units contribution. In the same way the functions return the contribution
            of bias to the reconstruction process. NB: the sum of all these matrices (hidden and bias RF) corresponds to the reconstructed input.

            args:

                Hidden_Output: spread output

                Weight_inputs: net weights

                Weights_bias_hiddens_to_inputs: hidden_to_input bias weights

                hiddens_plotted: number of hidden units that you want to show


            return:

                Max_units_Activated_Indices: vector with indices of the most active N (hiddens_plotted) hiddens (it includes
                the label 'b' for bias)

                Matrix_winner_activated_RF: list of matrices (RFs) of the most active N (hiddens_plotted) hiddens.


    '''

    Matrix_winner_activated_RF = []

    Max_units_Activated_Indices = []

    indices_temp = (-Hidden_Output).argsort()[:hiddens_plotted]

    # HIDDEN UNITS RFs

    for (i, value) in enumerate(indices_temp):

            if i == (hiddens_plotted - 1):

                break

            single_RF = Weight_inputs[:, value]  #take single hidden receptor field

            #reconstruction of hiddens matrices

            single_RF_matrix = single_RF.reshape([28, 28, 3], order='F')

            Max_units_Activated_Indices.append(value)
            Matrix_winner_activated_RF.append(single_RF_matrix)

    # BIAS   RFs

    Bias_RF = Weights_bias_hiddens_to_inputs.reshape([28, 28, 3], order='F')

    # # change of values ina 0-1 range
    # max = np.max(Bias_RF)
    # min = np.min(Bias_RF)
    # m = interp1d([min, max], [0, 1])
    # Bias_RF = m(Bias_RF)

    Max_units_Activated_Indices.append('Bias')
    Matrix_winner_activated_RF.append(Bias_RF)

    return Max_units_Activated_Indices, Matrix_winner_activated_RF

def Second_Hiddens_RF_Reconstruction(Hidden_Output, Weight_inputs, Weight_inputs_second, Weights_bias_hiddens_to_inputs,
                                     Weights_bias_hiddens_to_inputs_second, hiddens_plotted):

    '''

            This function extracts the receptive fields of each hidden unit (second hidden layer) and bias (from second hidden to hidden and to input).
            A receptive field of single hidden unit corresponds to its contribution to the input reconstruction, i.e. the RGB reconstructed input matrix
            that you can see only in case of a recostruction only depending on this single hidden. Spreading an identiity matrix from the last hidden layer
            to the visible layer the function returns RGB matrices (28 x 28 x 3) of each hidden units contribution. In the same way the functions return the
            contribution of bias to the reconstruction process. NB: the sum of all these matrices (hidden and bias RF) corresponds to the reconstructed input.

            args:

                Hidden_Output: spread output

                Weight_inputs: net weights (from first hidden to visible)

                Weight_inputs_second: net weights (from second hidden to first hidden)

                Weights_bias_hiddens_to_inputs: hidden_to_input bias weights

                Weights_bias_hiddens_to_inputs_second: hidden_second_to_hidden_first bias weights

                hiddens_plotted: number of hidden units that you want to show


            return:

                Max_units_Activated_Indices: vector with indices of the most active N (hiddens_plotted) hiddens (it includes
                the label 'b' for bias)

                Matrix_winner_activated_RF: list of matrices (RFs) of the most active N (hiddens_plotted) hiddens.


    '''

    Matrix_winner_activated_RF = []

    Max_units_Activated_Indices = []

    indices_temp = (-Hidden_Output).argsort()[:hiddens_plotted]

    # HIDDEN UNITS RFs (INVERSE SPREAD OF NET WITH 'IDENTITY' MATRIX INPUT) %%%%%

    test_matrix = np.eye(Hidden_Output.shape[0], dtype = int)

    # FROM SECOND HIDDEN TO FIRST HIDDEN

    rec_test_matrix_on_first_hidden = np.dot(test_matrix, Weight_inputs_second.T)

    rec_test_matrix_on_first_hidden = rec_test_matrix_on_first_hidden + Weights_bias_hiddens_to_inputs_second

    rec_test_matrix_on_first_hidden = 1 / (1 + np.exp(-(rec_test_matrix_on_first_hidden)))

    # FROM FIRST HIDDEN TO VISIBLE

    rec_first_hidden_on_visible = np.dot(rec_test_matrix_on_first_hidden, Weight_inputs.T)

    rec_first_hidden_on_visible = rec_first_hidden_on_visible + Weights_bias_hiddens_to_inputs

    rec_first_hidden_on_visible = 1 / (1 + np.exp(-(rec_first_hidden_on_visible)))

    # BIAS (INVERSE SPREAD of NET BIAS) %%%%%

    # FROM SECOND HIDDEN TO FIRST HIDDEN

    rec_bias_second_on_first_hidden = 1 / (1 + np.exp(-(Weights_bias_hiddens_to_inputs_second)))

    # FROM FIRST HIDDEN TO VISIBLE

    rec_bias_second_on_visible = np.dot(rec_bias_second_on_first_hidden, Weight_inputs.T)

    rec_bias_second_on_visible = rec_bias_second_on_visible + Weights_bias_hiddens_to_inputs

    Second_bias_RF = 1 / (1 + np.exp(-(rec_bias_second_on_visible)))


    for (i, value) in enumerate(indices_temp):

            if i == (hiddens_plotted - 1):

                break

            single_RF = rec_first_hidden_on_visible[value, :] #take single hidden receptor field


            #reconstruction of hiddens matrices

            single_RF_matrix = single_RF.reshape([28, 28, 3], order='F')

            Max_units_Activated_Indices.append(value)

            Matrix_winner_activated_RF.append(single_RF_matrix)



    # bias receptor field reconstruction

    Second_bias_RF_matrix = Second_bias_RF.reshape([28, 28, 3], order='F')

    # # change of values ina 0-1 range
    # max = np.max(Second_bias_RF_matrix)
    # min = np.min(Second_bias_RF_matrix)
    # m = interp1d([min, max], [0, 1])
    # Second_bias_RF_matrix = m(Second_bias_RF_matrix)

    Max_units_Activated_Indices.append('Bias')
    Matrix_winner_activated_RF.append(Second_bias_RF_matrix)

    return Max_units_Activated_Indices, Matrix_winner_activated_RF

# GRAPHICAL RECONSTRUCTIONS

def Graphical_Input_recostruction(vertical_limit_graphic, oriz_limit_graphic, Original_matrix, Reconstructed_matrix, errors_absolute_value):

    '''

                This function adds to a previously opened window both the original input and rec_input images (plots)

                args:

                    Original_matrix: visible original input matrix
                    Reconstructed_matrix: visible reconstructed input matrix (inverse spread result)
                    errors_absolute_value: reconstruction error for each input (1 or many)
                    vertical_limit_graphic: vertical dimension of grid (rows)
                    oriz_limit_graphic: orizontal dimension of grid (columns)

                return:

                    None

    '''


    if vertical_limit_graphic == 7:

        plt.suptitle('Visual analysis of RBM activation', fontsize=15, fontweight = 'bold')

    else:

        plt.suptitle('Visual analysis of DBN activation', fontsize=15, fontweight = 'bold')


    #PLOT FIRST IMAGE (ORIGINAL INPUT)

    Original_Graph = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic), (0, 0), colspan = 3, rowspan = 3)

    Original_Graph.imshow(Original_matrix)

    Original_Graph.set_title(' Original image ')

    Original_Graph.axis('off')

    # PLOT SECOND IMAGE (RECONSTRUCTION)

    Reconstructed_Graph = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic), (0, oriz_limit_graphic - 3), colspan = 3, rowspan = 3)

    Reconstructed_Graph.imshow(Reconstructed_matrix)

    Reconstructed_Graph.set_title('Reconstructed Image (L1 norm = ' + str((np.around(errors_absolute_value, decimals=3))) + str(" )"))

    Reconstructed_Graph.axis('off')

def Graphical_Hiddens_RF_Reconstruction(vertical_limit_graphic, oriz_limit_graphic, hiddens_activations, RF_matrices, RF_indices,
                                        hiddens_activations_second = 0, RF_matrices_second = 0, RF_indices_second = 0):
    '''

                    This function adds to a previously opened window new visual informations:

                    - activation values of second hidden layer units (histogram)
                    - receptive fields of second hidden layer units (plots)

                    - activation values of first hidden layer units (histogram)
                    - receptive fields of second hidden layer units (plots)

                    args:

                        vertical_limit_graphic: vertical dimension of grid (rows)
                        oriz_limit_graphic: orizontal dimension of grid (columns)

                        hiddens_activations: activation values of first hidden layer units
                        RF_matrices: list of matrices (RFs) of the most active first hidden layer units.
                        RF_indices: vector with indices of the most active first hiddens layer (it includes
                        the label 'b' for bias)

                        hiddens_activations_second: activation values of second hidden layer units
                        RF_matrices_second: list of matrices (RFs) of the most active second hidden layer units.
                        RF_indices_second: vector with indices of the most active second hidden layer (it includes
                        the label 'b' for bias)

                    return:

                        None



    '''

    graph_shift_rows = 6

    if isinstance(hiddens_activations_second, np.ndarray): # IN CASE OF DBN ACTIVATION

        graph_shift_rows = 0

        # ACTIVATIONS PLOT OF SECOND HIDDEN LAYER (HISTOGRAM)

        Hiddens_Activations_plot_second = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic), (5, 0),
                                                    colspan= oriz_limit_graphic)

        X = np.arange(0, hiddens_activations_second.shape[0])

        plt.bar(X, hiddens_activations_second)
        plt.ylim(0, 1)
        plt.tick_params(axis='x', labelbottom='off')
        plt.title(str(" Hiddens Second Activated ( ") + str(
            np.around((len(hiddens_activations_second[hiddens_activations_second >= 0.1]) / len(X)) * 100, decimals = 2))
                  + str(' % )'), fontweight = 'bold')

        # RECEPTIVE FIELDS OF SECOND HIDDEN LAYER (PLOTS)

        for pos_fig, (matrix, index) in enumerate(zip(RF_matrices_second, RF_indices_second)):

            Hidden_RF_Single = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic),
                                                (7, pos_fig))

            Hidden_RF_Single.imshow(matrix)

            plt.title("H. " + str(index))

            plt.axis('off')

    # ACTIVATIONS PLOT OF FIRST HIDDEN LAYER (HISTOGRAM)

    Hiddens_Activations_plot = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic), (10 - graph_shift_rows, 0), colspan = oriz_limit_graphic)

    X = np.arange(0, hiddens_activations.shape[0])

    plt.bar(X, hiddens_activations)
    plt.ylim(0, 1)
    plt.tick_params(axis='x',labelbottom='off')
    plt.title(str(" Hiddens First Activated ( ") + str(np.around((len(hiddens_activations[hiddens_activations >= 0.1]) / len(X)) * 100, decimals = 2))
              + str(' % )'), fontweight = 'bold')

    # RECEPTIVE FIELDS OF FIRST HIDDEN LAYER (PLOTS)

    for pos_fig, (matrix, index)  in enumerate(zip(RF_matrices, RF_indices)):

        Hidden_RF_Single = plt.subplot2grid((vertical_limit_graphic, oriz_limit_graphic), (12 - graph_shift_rows, pos_fig))

        Hidden_RF_Single.imshow(matrix)

        plt.title("H. " + str(index))

        plt.axis('off')

#   MACRO FUNCTIONS OF WINDOWS BUILDERS

def Window_builder_ideals_recostruction(Ideal_actions, Weight_inputs, Weights_bias_hiddens_to_inputs,
                                                  Weight_inputs_other = 0, Weights_bias_hiddens_to_inputs_other = 0):
        '''

            This function creates a window with the ideals reconstructions images. These images are the result of an inverse
            spread of ideal actions (labels). In case of second RBM learning or DBN (2 hidden layers) learning the reconstructed images
            correspond to the first hidden layer activation from second hidden layer, and then this function executed an additional spread
            from first hidden to visible layer.

            args:

                Ideal_actions: matrix with the four ideal actions

                Weight_inputs: net weights
                Weight_inputs_other: net weights of other RBM

                Weights_bias_hiddens_to_inputs: hidden_to_input bias weights
                Weights_bias_hiddens_to_inputs_other: hidden_to_input bias weights of other RBM


            return:

                Fig: windows with 4 plots

        '''

        if isinstance(Weight_inputs_other, np.ndarray):

            Reconstructed_ideals_hidden_potential = np.dot(Ideal_actions, Weight_inputs.T)
            Reconstructed_ideals_hidden_potential = Reconstructed_ideals_hidden_potential + Weights_bias_hiddens_to_inputs
            Reconstructed_ideals_hidden = 1 / (1 + np.exp(-(Reconstructed_ideals_hidden_potential)))

            Reconstructed_ideals_potential = np.dot(Reconstructed_ideals_hidden, Weight_inputs_other.T)
            Reconstructed_ideals_potential = Reconstructed_ideals_potential + Weights_bias_hiddens_to_inputs_other
            Reconstructed_ideals = 1 / (1 + np.exp(-(Reconstructed_ideals_potential)))

        else:

            Reconstructed_ideals_potential = np.dot(Ideal_actions, Weight_inputs.T) + Weights_bias_hiddens_to_inputs
            Reconstructed_ideals = 1 / (1 + np.exp(-(Reconstructed_ideals_potential)))

        Matricial_Reconstructions = []

        for element in range(0, Reconstructed_ideals.shape[0]):

            single_rec = Reconstructed_ideals[element].reshape([28, 28, 3], order='F')

            Matricial_Reconstructions.append(single_rec)

        Fig = plt.figure()
        plt.suptitle('Ideal actions recostructions', fontweight = 'bold', fontsize = 25)

        single_col = 0
        single_row = 0

        for N, rec in enumerate(Matricial_Reconstructions):

            Grid_N = plt.subplot2grid((1,len(Matricial_Reconstructions)), (single_row, single_col), colspan= 1, rowspan= 1)

            Grid_N.imshow(rec)
            Title = plt.title("N. " + str(N), fontsize = 7, fontweight = 'bold')
            Title.set_position([.5, 0.95])
            plt.axis('off')

            single_col += 1

        return Fig

def Windows_builder_network_weights(Weights, Weights_bias_1, Weights_bias_2):

    '''

    This function create a window with many analisys regarding the weights of network and biases (from input to hidden and inverse).
    In particular for each weights matrix the window proposes a simple 2D image of matrix and an analisys of distribution (kernel density estimation).

    Args:

     - Weights: network weights matrix
     - Weights_bias_1: weights matrix of bias 1 (from input to hidden)
     - param Weights_bias_2: weights matrix of bias 2 (from hidden to input)

    return:

    - Fig:    Windows with plots

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))
    Fig.clf()

    Fig.suptitle('Weights distributions', fontsize=22, fontweight='bold')

    num_rows = 2
    num_columns = 3

    weights_flattened = np.reshape(Weights, (Weights.shape[0] * Weights.shape[1],))

    weights_b_1_flattened = np.reshape(Weights_bias_1, (Weights_bias_1.shape[0] * Weights_bias_1.shape[1],))


    weights_b_2_flattened = np.reshape(Weights_bias_2, (Weights_bias_2.shape[0] * Weights_bias_2.shape[1],))

    #          WEIGHTS MATRIX PLOT %%%%%%%%%%%

    Weights_plot = plt.subplot(num_rows, num_columns, 1)

    plt.pcolor(Weights, cmap='RdBu')

    Weights_plot.set_xlabel('Hiddens units')
    Weights_plot.set_ylabel('Inputs units')
    plt.colorbar()

    Weights_plot.set_title('NET weights matrix', fontweight='bold')


    Weights_b_1_plot = plt.subplot(num_rows, num_columns, 2)

    plt.pcolor(Weights_bias_1, cmap='RdBu')

    Weights_b_1_plot.set_xlabel('Hiddens units')
    Weights_b_1_plot.set_ylabel('Inputs units')
    plt.colorbar()

    Weights_b_1_plot.set_title('bias (input -> hidden) weights matrix', fontweight='bold')


    Weights_b_2_plot = plt.subplot(num_rows, num_columns, 3)

    plt.pcolor(Weights_bias_2, cmap='RdBu')

    Weights_b_2_plot.set_xlabel('Hiddens units')
    Weights_b_2_plot.set_ylabel('Inputs units')
    plt.colorbar()

    Weights_b_2_plot.set_title('bias (hidden -> input) weights matrix', fontweight='bold')


    #          WEIGHTS DISTRIBUTION PLOT %%%%%%%%%%%

    Weights_distribution_plot = plt.subplot(num_rows, num_columns, 4)

    sb.distplot(weights_flattened, kde=True, hist=True, color='darkgoldenrod')

    Weights_distribution_plot.set_title('NET weights distribution', fontweight='bold')

    Weights_b_1_distribution_plot = plt.subplot(num_rows, num_columns, 5)

    sb.distplot(weights_b_1_flattened, kde=True, hist=True, color='darkgoldenrod')

    Weights_b_1_distribution_plot.set_title('bias (input -> hidden) weights distribution', fontweight='bold')

    Weights_b_2_distribution_plot = plt.subplot(num_rows, num_columns, 6)

    sb.distplot(weights_b_2_flattened, kde=True, hist=True, color='darkgoldenrod')

    Weights_b_2_distribution_plot.set_title('bias (hidden -> input) weights distribution', fontweight='bold')

    return Fig

def Window_builder_all_inputs_recostructions(Reconstructed_inputs, Weight_inputs = 0, Weights_bias_hiddens_to_inputs = 0):

    '''

        This function creates a window with all inputs reconstructions images (8 x 8). In case of DBN (2 hidden layers)
        the reconstructed inputs correspond to the first hidden layer activation from second hidden layer, so this function
        executed an additional spread from first hidden to visible layer.

        args:

            Reconstructed_input: visible (input layer or first hidden layer) reconstructed input (inverse spread result)

            Weight_inputs: net weights (from first hidden to visible)
            Weights_bias_hiddens_to_inputs: hidden_to_input bias weights (from first hidden to input layer)


        return:

            Fig: windows with 64 plots
            Reconstructed_inputs: vectors of inputs reconstructions

    '''

    Fig = plt.figure(figsize=(19.20,10.80))

    plt.clf()

    if Reconstructed_inputs.shape[1] == (28 * 28 * 3):

        plt.suptitle('Inputs recostructions (RBM)', fontweight='bold', fontsize = 25)

    else:

        plt.suptitle('Inputs recostructions (DBN)', fontweight='bold', fontsize = 25)


    if isinstance(Weight_inputs, np.ndarray):  # IN CASE OF SECOND HIDDEN LAYER WE ADD ANOTHER SPREAD FROM HIDDEN (REC_INPUT OF SECOND RBM) TO VISIBLE

        Reconstructed_inputs_potential = np.dot(Reconstructed_inputs, Weight_inputs.T)
        Reconstructed_inputs_potential = Reconstructed_inputs_potential + Weights_bias_hiddens_to_inputs
        Reconstructed_inputs = 1 / (1 + np.exp(-(Reconstructed_inputs_potential)))

    Matricial_Reconstructions = []


    # TRASNFORM INPUT RECONSTRUCTIONS VECTORS IN MATRICES

    for element in range(0, len(Reconstructed_inputs)):

        single_rec = Reconstructed_inputs[element].reshape([28, 28, 3], order='F')

        Matricial_Reconstructions.append(single_rec)


    # PLOT RECONSTRUCTIONS INTO A 8 X 8 (64) GRID

    single_col = 0
    single_row = 0

    for N, rec in enumerate(Matricial_Reconstructions):

        Grid_N = plt.subplot2grid((8,8), (single_row, single_col), colspan= 1, rowspan= 1)
        Grid_N.imshow(rec)

        Title = plt.title("N. " + str(N), fontsize = 7, fontweight = 'bold')
        Title.set_position([.5, 0.95])
        plt.axis('off')

        if single_col == 7:
            single_row += 1
            single_col = 0
        else:
            single_col += 1

    return Fig, Reconstructed_inputs

def Window_builder_single_input_recostruction(Input, Rec_input, Hidden_Output, Weight_inputs, Weights_bias_hiddens_to_inputs,
                                                    errors_absolute_value, hiddens_plotted, Hidden_Output_second = 0, Weight_inputs_second = 0,
                                                    Weights_bias_hiddens_to_inputs_second = 0):


    '''

                    This function creates a window with many visual informations regarding the activation of net.

                    args:
                    
                        Input: visible original input
                        Rec_Input: visible reconstructed input (inverse spread result)
                        
                        Hidden_Output: spread output (first hidden layer)
                        Hidden_Output_second: spread output (second hidden layer)
                        
                        Weight_inputs: net weights (from first hidden to visible)
                        Weight_inputs_second: net weights (from second hidden to first hidden)

                        Weights_bias_hiddens_to_inputs: hidden_to_input bias weights
                        Weights_bias_hiddens_to_inputs_second: hidden_second_to_hidden_first bias weights

                        hiddens_plotted: number of hidden units that you want to show
                        errors_absolute_value: reconstruction error for each input (1 or many)

                    return:

                        Fig: windows with many informations regading the net activation

    '''

    # FIG INIT AND GRAPHICAL LIMITS

    Fig = plt.figure(figsize=(19.20, 10.80))

    vertical_limit_graphic = 7

    oriz_limit_graphic = max(hiddens_plotted, 10)

    Original_matrix, Reconstructed_matrix = Inputs_recostructions(Input, Rec_input)

    RF_indices, RF_matrices = First_Hiddens_RF_Reconstruction(Hidden_Output, Weight_inputs, Weights_bias_hiddens_to_inputs, hiddens_plotted)

    if isinstance(Weights_bias_hiddens_to_inputs_second, np.ndarray):

        vertical_limit_graphic = 13

        RF_indices_second, RF_matrices_second = Second_Hiddens_RF_Reconstruction(Hidden_Output_second, Weight_inputs, Weight_inputs_second,
                                                                             Weights_bias_hiddens_to_inputs, Weights_bias_hiddens_to_inputs_second,
                                                                             hiddens_plotted)

    # GRAPHICAL RECONSTRUCTION OF INPUT

    Graphical_Input_recostruction(vertical_limit_graphic, oriz_limit_graphic, Original_matrix, Reconstructed_matrix, errors_absolute_value)

    # GRAPHICAL RECONSTRUCTION OF HIDDEN RECEPTIVE FIELDS

    if isinstance(Weights_bias_hiddens_to_inputs_second, np.ndarray):

        Graphical_Hiddens_RF_Reconstruction(vertical_limit_graphic, oriz_limit_graphic, Hidden_Output, RF_matrices, RF_indices,
                                        Hidden_Output_second, RF_matrices_second, RF_indices_second)

    else:

        Graphical_Hiddens_RF_Reconstruction(vertical_limit_graphic, oriz_limit_graphic, Hidden_Output, RF_matrices,
                                            RF_indices)


    return Fig

def Window_builder_Information_Loss_single_input(Errors, Errors_2 = 0):

    '''

        This function creates a windows with one or two curves, representing the recostruction error along multiple resonances/
        Gibbs sampling steps. In case of DBN there two curves

        args:

            errors: rec. errors for multiple resonances steps of First RBM
            Errors_2: ...of Second RBM

        return:

            Fig: windows with rec. error plot.

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))
    Fig.clf()

    plt.suptitle(' Resonance/Gibbs sampling information Loss (RBM) ', fontsize=15, fontweight='bold')

    plt.plot(range(0, len(Errors)), Errors, linestyle='-', color='blue', label='Reconstruction error (RBM 1)')

    if isinstance(Errors_2, np.ndarray):

        plt.suptitle(' Resonance/Gibbs sampling information Loss (DBN) ', fontsize=15, fontweight='bold')

        plt.plot(range(0, len(Errors_2)), Errors_2, linestyle='-', color='red', label='Reconstruction error (RBM 2)')

    plt.title(' Reconstruction error of a single input')
    plt.legend(loc="best")

    return Fig



def Window_builder_reconstruction_errors(Errors, STDs_Errors, Min_Err_target, Errors_OTHER = 0, STDs_Errors_OTHER = 0, Min_Err_target_OTHER = 0):

        '''

        This function creates a window with two plots corresponding to the reconstruction error of inputs (Batch).
        The first plot reports the avg and sd of rec_errors of inputs while the second reports the same data transformed with
        a rolling/moving media that smooths the data and makes the plot more clearer. In case OF unsup DBN, the two plots of
        reconstruction errors are composed by rec_errors of first RBM and second RBM.

        args:

            Errors: list of means of inputs reconstructions (batch rec_error)
            STDs_Errors: list of standard deviations of....
            Min_Err_target: Min Error to achieve for RBM
            Min_Err_target_OTHER: same of previous variable for other RBM

        return:

            Fig: windows with two plots of batch reconstruction errors

        '''

        Fig = plt.figure(figsize=(19.20,10.80))
        Fig.clf()

        plt.suptitle(' Learning performance \n (Reconstruction error of whole datatset) ', fontsize=15, fontweight = 'bold')

        rolling_value = len(Errors) // 10

        # RECONSTRUCTION ERRORS%%%%%%%%%%%%%%%%%%%

        Min_Err = np.min(Errors)
        Max_Err = np.max(Errors)

        Pos_Min_Err = np.where(Errors == Min_Err)[0][0]
        Min_Err_std = STDs_Errors[Pos_Min_Err]

        if isinstance(Errors_OTHER, np.ndarray):

            Min_Err_OTHER = np.min(Errors_OTHER)
            Max_Err_OTHER = np.max(Errors_OTHER)
            Pos_Min_Err_OTHER = np.where(Errors_OTHER == Min_Err_OTHER)[0][0]
            Min_Err_std_OTHER = STDs_Errors_OTHER[Pos_Min_Err_OTHER]

        Bars_plot = plt.subplot(121)

        # BARS PLOT

        Bars_plot.errorbar(range(0, len(Errors)), Errors, STDs_Errors, linestyle='-', color='blue',
                           label='Reconstruction error (Min = ' + str(np.around(Min_Err, decimals=3)) + str(" +/- ") + str(
                           np.around(Min_Err_std, decimals=3)) + str(')'), ecolor='cornflowerblue')

        # Bars_plot.text(len(Errors) * 0.30, Max_Err * 0.50,
        #                ' Min_Err = ' + str(np.around(Min_Err, decimals=3)) + str(" +/- ") + str(
        #                    np.around(Min_Err_std, decimals=3)), fontsize=13, fontweight = 'bold')


        if isinstance(Errors_OTHER, np.ndarray):

            Bars_plot.errorbar(range(0, len(Errors_OTHER)), Errors_OTHER, STDs_Errors_OTHER, linestyle='-', color='red',
                               label='Reconstruction error (RBM 2, Min = ' + str(np.around(Min_Err_OTHER, decimals=3)) + str(
                               " +/- ") + str(
                               np.around(Min_Err_std_OTHER, decimals=3)) + str(')'), ecolor='darkred')

            # Bars_plot.text(len(Errors_OTHER) * 0.30, Max_Err * 0.46,
            #                ' Min_Err (RBM 2) = ' + str(np.around(Min_Err_OTHER, decimals=3)) + str(
            #                    " +/- ") + str(
            #                    np.around(Min_Err_std_OTHER, decimals=3)), fontsize=13, fontweight = 'bold')

        Bars_plot.set_xlabel('Epoc')
        Bars_plot.set_ylabel('Reconstruction error')
        Bars_plot.set_title('Reconstruction Error')

        Bars_plot.legend(loc='upper right')

        # LINE PLOT

        Line_plot = plt.subplot(122)

        # moving media

        Errors_list = copy.deepcopy(Errors)
        errors = pd.DataFrame({'Serie': Errors})
        errors = errors.rolling(rolling_value).mean()

        ii = np.isfinite(errors)
        errors = errors[ii]

        x = np.arange(0, len(errors))
        y = errors

        Line_plot.plot(x, y, linestyle='-', color='blue', label='Reconstruction error')
        Line_plot.axhline(y=Min_Err_target, linestyle='--', color='blue', label=' Rec. error target')

        if isinstance(Errors_OTHER, np.ndarray):
            # moving media

            Errors_list_OTHER = copy.deepcopy(Errors_OTHER)
            errors_OTHER = pd.DataFrame({'Serie': Errors_OTHER})
            errors_OTHER = errors_OTHER.rolling(rolling_value).mean()

            ii = np.isfinite(errors_OTHER)
            errors_OTHER = errors_OTHER[ii]

            x_OTHER = np.arange(0, len(errors_OTHER))
            y_OTHER = errors_OTHER

            Line_plot.plot(x_OTHER, y_OTHER, linestyle='-', color='red', label='Reconstruction error (RBM 2)')
            Line_plot.axhline(y=Min_Err_target_OTHER, linestyle='--', color='red', label='Rec. error target (RBM 2)')

        Line_plot.set_xlabel('Epoc')
        Line_plot.set_ylabel('Reconstruction error')
        Line_plot.set_title('Reconstruction Error (MOVING MEDIA)')


        Line_plot.legend(loc='upper right')


        return Fig


def Window_builder_reconstruction_errors_accuracies_supervised_learning(Errors, STDs_Errors, Min_Err_target, Accuracies, STDs_Accuracies,
                                                                        Errors_OTHER = 0, STDs_Errors_OTHER = 0, Min_Err_target_OTHER = 0):

    '''

    This function creates a window with four plots corresponding to the reconstruction error of inputs (Batch) and the accuracies (Batch).
    The first plot reports the avg and sd of rec_errors of inputs while the second reports the same data transformed with
    a rolling/moving media that smooths the data and makes the plot more clearer. The third and fourth  plots are identical to previous two
    for accuracy. In case of supervised learning of DBN, the two plots of reconstruction errors are composed by rec_errors of
    first RBM and second RBM.

    args:

        Errors: list of means of inputs reconstructions (batch rec_error)
        STDs_Errors: list of standard deviations of....
        Accuracies: list of means of accuracies (batch accuracy)
        STDs_Accuracies: list of standard deviations of....
        Min_Err_target: Min Error to achieve for RBM
        Min_Err_target_OTHER: same of previous variable for other RBM

    return:

        Fig: windows with two plots of batch reconstruction errors

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))
    Fig.clf()

    plt.suptitle(' Learning performance \n (Reconstruction error and Accuracy of whole datatset) ', fontsize=15, fontweight='bold')

    # RECONSTRUCTION ERRORS%%%%%%%%%%%%%%%%%%%

    Min_Err = np.min(Errors)
    Max_Err = np.max(Errors)
    Min_Err_std = STDs_Errors[np.where(Errors == Min_Err)[0][0]]

    if isinstance(Errors_OTHER, np.ndarray) or isinstance(Errors_OTHER, list):

        Min_Err_OTHER = np.min(Errors_OTHER)
        Max_Err_OTHER = np.max(Errors_OTHER)
        Min_Err_std_OTHER = STDs_Errors_OTHER[np.where(Errors_OTHER == Min_Err_OTHER)[0][0]]

    Bars_plot = plt.subplot(221)

    # BARS PLOT

    Bars_plot.errorbar(range(0, len(Errors)), Errors, STDs_Errors, linestyle='-', color='blue',
                       label='Reconstruction error (Min = ' + str(np.around(Min_Err, decimals=3)) + str(" +/- ") + str(
                       np.around(Min_Err_std, decimals=3)) + str(')'), ecolor='cornflowerblue')

    if isinstance(Errors_OTHER, np.ndarray) or isinstance(Errors_OTHER, list):

        Bars_plot.errorbar(range(0, len(Errors_OTHER)), Errors_OTHER, STDs_Errors_OTHER, linestyle='-', color='red',
                           label='Reconstruction error (RBM 2, Min = ' + str(np.around(Min_Err_OTHER, decimals=3)) + str(" +/- ") + str(
                       np.around(Min_Err_std_OTHER, decimals=3)) + str(')'), ecolor='darkred')

    Bars_plot.set_xlabel('Epoc')
    Bars_plot.set_ylabel('Reconstruction error')
    Bars_plot.set_title('Reconstruction Error ')

    # Bars_plot.text(len(Errors) * 0.30, Max_Err * 0.75,
    #                ' Min_Err = ' + str(np.around(Min_Err, decimals=3)) + str(" +/- ") + str(
    #                    np.around(Min_Err_std, decimals=3)), fontsize=13, fontweight = 'bold')
    #
    # if isinstance(Errors_OTHER, np.ndarray) or isinstance(Errors_OTHER, list):
    #
    #     Bars_plot.text(len(Errors) * 0.30, Max_Err * 0.65,
    #                    ' Min_Err (RBM 2) = ' + str(np.around(Min_Err_OTHER, decimals=3)) + str(" +/- ") + str(
    #                        np.around(Min_Err_std_OTHER, decimals=3)), fontsize=13, fontweight = 'bold')


    Bars_plot.legend(loc='upper right')

    # LINE PLOT

    Line_plot = plt.subplot(222)

    # moving media

    rolling_media_smoothing = len(Errors) // 10

    Errors_list = copy.deepcopy(Errors)
    errors = pd.DataFrame({'Serie': Errors})
    errors = errors.rolling(rolling_media_smoothing).mean()

    ii = np.isfinite(errors)
    errors = errors[ii]

    x = np.arange(0, len(errors))
    y = errors

    Line_plot.plot(x, y, linestyle='-', color='blue', label='Reconstruction error')
    Line_plot.axhline(y = Min_Err_target, linestyle='--', color = 'blue', label = ' Rec. error target')


    if isinstance(Errors_OTHER, np.ndarray) or isinstance(Errors_OTHER, list):

        # moving media

        Errors_list_OTHER = copy.deepcopy(Errors_OTHER)
        errors_OTHER = pd.DataFrame({'Serie': Errors_OTHER})
        errors_OTHER = errors_OTHER.rolling(rolling_media_smoothing).mean()

        ii = np.isfinite(errors_OTHER)
        errors_OTHER = errors_OTHER[ii]

        x_OTHER = np.arange(0, len(errors_OTHER))
        y_OTHER = errors_OTHER

        Line_plot.plot(x_OTHER, y_OTHER, linestyle='-', color='red', label='Reconstruction error (RBM 2)')
        Line_plot.axhline(y=Min_Err_target_OTHER, linestyle='--', color='red', label='Rec. error target (RBM 2)')

    Line_plot.set_xlabel('Epoc')
    Line_plot.set_ylabel('Reconstruction error')
    Line_plot.set_title('Reconstruction Error (MOVING MEDIA)')


    Line_plot.legend(loc='upper right')

    # ACCURACIES %%%%%%%%%%%%%%%%%%%

    Min_Acc = np.min(Accuracies)
    Max_Acc = np.max(Accuracies)
    Max_Acc_std = STDs_Accuracies[np.where(Accuracies == Max_Acc)[0][0]]

    Bars_plot_acc = plt.subplot(223)

    # BARS PLOT

    Bars_plot_acc.errorbar(range(0, len(Accuracies)), Accuracies, STDs_Accuracies, linestyle='-', color='green',
                       label='Accuracy (Max = ' + str(np.around(Max_Acc, decimals=3)) + str(" +/- ") + str(
                       np.around(Max_Acc_std, decimals=3)) + str(')'), ecolor='grey')

    Bars_plot_acc.set_xlabel('Epoc')
    Bars_plot_acc.set_ylabel('Accuracy')
    Bars_plot_acc.set_title('Accuracy')
    # Bars_plot_acc.text(len(Accuracies) * 0.50, Max_Acc * 0.25,
    #                ' Max_Acc (Batch) = ' + str(np.around(Max_Acc, decimals=3)) + str(" +/- ") + str(
    #                    np.around(Max_Acc_std, decimals=3)), fontsize=13, fontweight = 'bold')

    Bars_plot_acc.legend(loc='lower right')

    # LINE PLOT

    Line_plot_acc = plt.subplot(224)

    # moving media

    Accuracies_list = copy.deepcopy(Accuracies)
    accuracies = pd.DataFrame({'Serie': Accuracies})
    accuracies = accuracies.rolling(rolling_media_smoothing).mean()

    ii = np.isfinite(accuracies)
    accuracies = accuracies[ii]

    x = np.arange(0, len(accuracies))
    y = accuracies

    Line_plot_acc.plot(x, y, linestyle='-', color='green', label='Accuracy')
    Line_plot_acc.axhline(y = 1, linestyle='--', color = 'green', label = ' Accuracy target')

    Line_plot_acc.legend(loc="best")

    Line_plot_acc.set_xlabel('Epoc')
    Line_plot_acc.set_ylabel('Accuracy')
    Line_plot_acc.set_title('Accuracy (MOVING MEDIA)')

    Line_plot_acc.legend(loc='lower right')

    # plt.pause(1)

    return Fig


def Window_builder_layers_activations(reducted_inputs, whole_dataset_labels, foundamental_dataset_labels, verbal_foundamental_labels, specific_layer='input', dim = 2,
                                      reducted_inputs_not_biased='not available'):
    '''

        This function is an alternative of "Window_builder_layers_activations". This specific versio should be more adapt to
        be generalized to many dataset (previous version is adapt to Giovanni's dataset)

        args:

            reducted_inputs: array of inputs x number of principal components (default = 2)
            whole_dataset_labels: expected output for each input of dataset (useful to distinguish the spots)
            foundamental_dataset_labels: specific expected output for each unique vector
            verbal_foundamental_labels: label to show in the plot legend
            reducted_inputs_not_biased: identical to rpevious variable, referring to a potential adversarial network
            specific_layer: layer of RBM from which we extract the rappresentations
            salient_feature: specific feature for plotting the 3D points
            dim: principal components (defaults = 3)

        return:

            windows with a graphical representation of layer activations

    '''
    Fig = plt.figure(figsize=(19.20, 10.80))

    # SALIENT FEATURE TO HIGHLIGHT WITH FOUR FOUNDAMENTAL COLORS (RGBY)


    # LAYERS LABELS

    if specific_layer == 'input':

        Fig.suptitle(' DISTRIBUTION OF VISIBLE ACTIVATIONS \n\n ', fontsize=12,
                     fontweight='bold')

    elif specific_layer == 'FHL':

        Fig.suptitle(' DISTRIBUTION OF FIRST HIDDEN ACTIVATIONS \n\n ', fontsize=12,
                     fontweight='bold')

    elif specific_layer == 'SHL':

        Fig.suptitle(' DISTRIBUTION OF SECOND HIDDEN ACTIVATIONS \n\n ', fontsize=12,
                     fontweight='bold')

    # INITIALIZATION OF LABELS AND COLOR LIST


    Plot_points_colors = ['green', 'red', 'blue', 'yellow']

    if foundamental_dataset_labels.shape[0] == whole_dataset_labels.shape[0]:

        if 'Color'  in verbal_foundamental_labels[0]:

            Plot_points_colors = np.tile(Plot_points_colors, 16)

            verbal_foundamental_labels = np.tile(verbal_foundamental_labels, 16)

        elif 'Shape' in verbal_foundamental_labels[0]:

            Plot_points_colors = np.repeat(Plot_points_colors, 4)
            Plot_points_colors = np.tile(Plot_points_colors, 4)

            verbal_foundamental_labels = np.repeat(verbal_foundamental_labels, 4)
            verbal_foundamental_labels = np.tile(verbal_foundamental_labels, 4)

        elif 'Size' in verbal_foundamental_labels[0]:

            Plot_points_colors = np.repeat(Plot_points_colors, 16, axis=0)
            Plot_points_colors = np.tile(Plot_points_colors, 4)

            verbal_foundamental_labels = np.repeat(verbal_foundamental_labels, 16, axis=0)
            verbal_foundamental_labels = np.tile(verbal_foundamental_labels, 4)

    whole_dataset_verbal_labels = []
    whole_dataset_points_colors = []

    for count_subj in np.arange(0, whole_dataset_labels.shape[0]):

        single_repr  = whole_dataset_labels[count_subj, :]

        for count_label, spec_label in enumerate(foundamental_dataset_labels):

            if np.all(single_repr ==  spec_label):

                    whole_dataset_points_colors.append(Plot_points_colors[count_label])

                    whole_dataset_verbal_labels.append(verbal_foundamental_labels[count_label])



    if reducted_inputs_not_biased != 'not available':

        if dim == 3:

            ax = Fig.add_subplot(121, projection='3d')
            plt.title(' R-BIASED DBN ACTIVATIONS ', fontsize=10)

            bx = Fig.add_subplot(122, projection='3d')

            plt.title(' REGULAR DBN ACTIVATIONS ', fontsize=10)


        else:

            ax = Fig.add_subplot(121)
            plt.title(' R-BIASED DBN ACTIVATIONS ', fontsize=10)

            bx = Fig.add_subplot(122)

            plt.title(' REGULAR DBN ACTIVATIONS ', fontsize=10)



    else:

        if dim == 3:
            ax = Fig.add_subplot(111, projection='3d')
            plt.title(' DBN ACTIVATIONS ', fontsize=10)


        else:
            ax = Fig.add_subplot(111)
            plt.title(' DBN ACTIVATIONS ', fontsize=10)

    for point in np.arange(0, reducted_inputs.shape[0]):  # LOOP FOR DRAWING POINT

        X_point = reducted_inputs[point, 0]
        Y_point = reducted_inputs[point, 1]

        if dim == 3:

            Z_point = reducted_inputs[point, 2]

            ax.scatter(X_point, Y_point, Z_point, color= whole_dataset_points_colors[point], alpha=1,
                       label= whole_dataset_verbal_labels[point])

        else:

            ax.scatter(X_point, Y_point, color= whole_dataset_points_colors[point], alpha=1,
                       label= whole_dataset_verbal_labels[point])


    ymin_original, ymax_original = ax.get_ylim()
    xmin_original, xmax_original = ax.get_xlim()

    # ax.axis('square')

    ax.set_xlabel('component 1 (X)')
    ax.set_ylabel('component 2 (Y)')

    if dim == 3:
        zmin_original, zmax_original = ax.get_zlim()
        ax.set_zlabel('component 3 (Z)')

    Legend_elements = [
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[0],
                   color=Plot_points_colors[0], label=verbal_foundamental_labels[0], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[1],
                   color=Plot_points_colors[1], label=verbal_foundamental_labels[1], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[2],
                   color=Plot_points_colors[2], label=verbal_foundamental_labels[2], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[3],
                   color=Plot_points_colors[3], label=verbal_foundamental_labels[3], markersize=10)
    ]

    ax.legend(handles=Legend_elements, loc='best')

    if reducted_inputs_not_biased != 'not available':

        for point in np.arange(0, reducted_inputs_not_biased.shape[0]):  # LOOP FOR DRAWING POINT

            X_point = reducted_inputs_not_biased[point, 0]
            Y_point = reducted_inputs_not_biased[point, 1]

            if dim == 3:

                Z_point = reducted_inputs_not_biased[point, 2]

                bx.scatter(X_point, Y_point, Z_point, color= whole_dataset_points_colors[point], alpha=1,
                           label = whole_dataset_verbal_labels[point])

            else:

                bx.scatter(X_point, Y_point, color= whole_dataset_points_colors[point], alpha=1,
                           label= whole_dataset_verbal_labels[point])

        # bx.set_xlim(xmin_original, xmax_original)
        # bx.set_ylim(ymin_original, ymax_original)

        # bx.axis('square')

        bx.set_xlabel('component 1 (X)')
        bx.set_ylabel('component 2 (Y)')

        if dim == 3:
            # bx.set_zlim(zmin_original, zmax_original)

            bx.set_zlabel('component 3 (Z)')

        bx.legend(handles=Legend_elements, loc='best')

    return Fig



# def Window_builder_layers_activations_resonances_Gibbs_sampling(reducted_input, reducted_inputs_not_biased, specific_layer, salient_feature):
#
#         '''
#
#         This function creates a window with two plots. The first shows the reducted (PCA) representations of layers activations
#         of whole dataset (single spread and single reconstruction). The second shows the same activation of a specific input
#         that resonates (gibbs sampling) into the network for N steps. This function is adapt to plot both the hidden and visible
#         layers representations. For now it is implemented only the 3D form.
#
#         args:
#
#             reducted_input: array of a single input reconstruction (N steps) x number of principal components (default = 3)
#             reducted_inputs_dataset: array of inputs (64) x number of principal components (default = 3)
#             specific_layer: labe f specific layer title
#             salient_feature: label to plot 3D points
#
#         return:
#
#             Fig: fig with two plots
#
#
#         '''
#
#         Fig = plt.figure(figsize=(19.20,10.80))
#
#         if specific_layer == 'IL':
#
#             layer_label = 'VISIBLE LAYER'
#
#
#         elif specific_layer == 'FHL':
#
#             layer_label = 'FIRST HIDDEN LAYER'
#
#
#
#         elif specific_layer == 'FHLR':
#
#             layer_label = 'FIRST HIDDEN LAYER (REC. FROM SECOND HIDDEN)'
#
#
#         elif specific_layer == 'SHL':
#
#             layer_label = 'SECOND HIDDEN LAYER'
#
#         if salient_feature == 'none':
#
#             salient_feature = 'color'
#
#         if salient_feature == 'color':
#
#             # COLOR LABELS
#             foundamental_labels_legend = '[Green colour: Green, Red colour: Red, Blue colour: Blue, Yellow colour: Yellow]'
#             specific_labels = ['Color: green', 'Color: red', 'Color: blue', 'Color: yellow']
#             specific_labels = np.tile(specific_labels, 16)
#
#         elif salient_feature == 'form':
#
#             # FORM LABELS
#             foundamental_labels_legend = '[Green colour: Square, Red colour: Circle, Blue colour: Bar, Yellow colour: Triangle]'
#             specific_labels = np.repeat(['Shape: square', 'Shape: circle', 'Shape: bar', 'Shape: triangle'], 4)
#             specific_labels = np.tile(specific_labels, 4)
#
#
#         elif salient_feature == 'size':
#
#             # SIZE LABELS
#             foundamental_labels_legend = '[Green colour: Great, Red colour: Great-medium, Blue colour: Medium-small, Yellow colour: Small]'
#             specific_labels = np.repeat(['Size: great', 'Size: great-medium', 'Size: medium-small', 'Size: small'], 16,
#                                         axis=0)
#             specific_labels = np.tile(specific_labels, 4)
#
#
#         Fig.suptitle(str(' DISTRIBUTION OF ') + str(layer_label) + str(' ACTIVATIONS (PCA) \n\n Salient feature: ') + str(salient_feature)
#                      + str('\n\n Salient attributes: ') + str(foundamental_labels_legend), fontsize=12,
#                      fontweight='bold')
#
#
#         # FIRST PLOT (SINGLE RECONSTRUCTION)
#
#         ax = Fig.add_subplot(121, projection='3d') # fig.gca(projection='3d')
#
#         count_color = 0
#         count_form = 0
#         count_size = 0
#         specific_attribute_count = 0
#
#         Plot_points_colors = ['green', 'red', 'blue', 'yellow']
#
#         for point in np.arange(0, reducted_inputs_not_biased.shape[0]):  # LOOP FOR DRAWING POINT
#
#             X_point = reducted_inputs_not_biased[point, 0]
#             Y_point = reducted_inputs_not_biased[point, 1]
#             Z_point = reducted_inputs_not_biased[point, 2]
#
#             ax.scatter(X_point, Y_point, Z_point, color=Plot_points_colors[specific_attribute_count], alpha=1,
#                        label=specific_labels[point])
#
#             count_color += 1
#
#             if count_color == 4:
#                 count_color = 0
#                 count_form += 1
#
#                 if count_form == 4:
#                     count_form = 0
#                     count_size += 1
#
#                     if count_size == 4:
#                         count_size == 0
#
#             if salient_feature == 'color':
#
#                 specific_attribute_count = count_color
#
#             elif salient_feature == 'form':
#
#                 specific_attribute_count = count_form
#
#             elif salient_feature == 'size':
#
#                 specific_attribute_count = count_size
#
#
#         ax.set_zlabel('component 3 (Z)')
#         ax.set_xlabel('component 1 (X)')
#         ax.set_ylabel('component 2 (Y)')
#
#         plt.title(' WHOLE DATASET (SINGLE SPREAD/REC) ')
#
#         zmin_dataset, zmax_dataset = ax.get_zlim()
#         ymin_dataset, ymax_dataset = ax.get_ylim()
#         xmin_dataset, xmax_dataset = ax.get_xlim()
#
#
#         # SECOND PLOT (GIBBS SAMPLING)
#
#         bx = Fig.add_subplot(122, projection='3d')  # fig.gca(projection='3d')
#
#         trasp = 0
#         trasp_single = 0.95 / reducted_input.shape[0]
#
#         for point in np.arange(0, reducted_input.shape[0]):
#
#             trasp += trasp_single
#
#             X_point = reducted_input[point, 0]
#             Y_point = reducted_input[point, 1]
#             Z_point = reducted_input[point, 2]
#
#
#             if point == 0:
#
#                 color = 'green'
#                 bx.scatter(X_point, Y_point, Z_point, color = color, alpha = 1, s = 50, label = 'Start point')
#
#             elif point == reducted_input.shape[0] - 1:
#
#                 color = 'red'
#                 bx.scatter(X_point, Y_point, Z_point, color = color, alpha=1, s = 50, label = 'Stop point')
#
#             else:
#
#                 color = 'k'
#                 bx.scatter(X_point, Y_point, Z_point, color = color, alpha = 0.5)
#
#
#
#         plt.legend(loc='lower right')
#
#         bx.set_xlim(xmin_dataset, xmax_dataset)
#         bx.set_ylim(ymin_dataset, ymax_dataset)
#         bx.set_zlim(zmin_dataset, zmax_dataset)
#
#         bx.set_zlabel('component 3 (Z)')
#         bx.set_xlabel('component 1 (X)')
#         bx.set_ylabel('component 2 (Y)')
#
#         plt.title(' SINGLE INPUT (NUMBER OF STEP/RESONANCES = ' + str(reducted_input.shape[0]) + str(')'))
#
#
#         return Fig
#

def Window_builder_layers_activations_resonances_Gibbs_sampling(reducted_inputs,
                                      whole_dataset_labels,
                                      foundamental_dataset_labels,
                                      verbal_foundamental_labels,
                                      reducted_activations_temporal_sequence,
                                      specific_layer='NONE',
                                      ):
    '''

            This function shows the reducted activations of a whole initial dataset, in  2D or 3D form, and the temporal
             sequence of activations of a network (e.g. a first plot shows the initial inputs activations and a second plot
             shows the activation of a network for a specific input along many time steps)

            args:

                reducted_inputs: array of inputs x number of principal components
                whole_dataset_labels: expected output for each input of dataset (useful to distinguish the spots)
                foundamental_dataset_labels: specific expected output for each unique vector
                verbal_foundamental_labels: label to show in the plot legend
                reducted_activations_temporal_sequence: temporal sequence of activations
                specific_layer: layer of architecture from which we extract the rappresentations (none/input/HCNN/HLSTM)


            return:

                windows with a graphical representation of layer activations

        '''

    Fig = plt.figure(figsize=(19.20, 10.80))

    # LAYERS LABELS

    if specific_layer == 'IL':

        layer_label = 'VISIBLE LAYER'


    elif specific_layer == 'FHL':

        layer_label = 'FIRST HIDDEN LAYER'



    elif specific_layer == 'FHLR':

        layer_label = 'FIRST HIDDEN LAYER (REC. FROM SECOND HIDDEN)'


    elif specific_layer == 'SHL':

        layer_label = 'SECOND HIDDEN LAYER'



    Fig.suptitle(str(' DISTRIBUTION OF ') + str(layer_label) + str(' ACTIVATIONS (PCA or TSNE)'), fontsize=12,
                 fontweight='bold')

    # INITIALIZATION OF LABELS AND COLOR LIST

    Plot_points_colors = ['green', 'red', 'blue',
                          'yellow']  # TO CHANGE TO INCREASE THE SPECIFIC KIND OF INPUTS (FOR NOW 4 CLASSES)

    whole_dataset_verbal_labels = []
    whole_dataset_points_colors = []

    for count_subj in np.arange(0, whole_dataset_labels.shape[0]):

        single_repr = whole_dataset_labels[count_subj, :]

        for count_label, spec_label in enumerate(foundamental_dataset_labels):

            if np.all(single_repr == spec_label):
                whole_dataset_points_colors.append(Plot_points_colors[count_label])

                whole_dataset_verbal_labels.append(verbal_foundamental_labels[count_label])

    # INTIIALIZATION OF PLOTS AND TITLES

    if reducted_activations_temporal_sequence.shape[1] == 3:

        ax = Fig.add_subplot(121, projection='3d')

        plt.title(' WHOLE DATASET ACTIVATIONS ', fontsize=10)

        bx = Fig.add_subplot(122, projection='3d')

        plt.title(' ACTIVATIONS TEMPORAL SEQUENCE (NUMBER OF CYCLES = '+ str(reducted_activations_temporal_sequence.shape[0]) + str(')'), fontsize=10)


    else:

        ax = Fig.add_subplot(121)

        plt.title(' WHOLE DATASET ACTIVATIONS ', fontsize=10)

        bx = Fig.add_subplot(122)

        plt.title(' ACTIVATIONS TEMPORAL SEQUENCE (NUMBER OF CYCLES = '+ str(reducted_activations_temporal_sequence.shape[0]) + str(')'), fontsize=10)


    # LOOP FOR POINTS DRAWING - WHOLE DATASET

    for point in np.arange(0, reducted_inputs.shape[0]):


        X_point = reducted_inputs[point, 0]
        Y_point = reducted_inputs[point, 1]

        if reducted_inputs.shape[1] == 3:

            Z_point = reducted_inputs[point, 2]

            ax.scatter(X_point, Y_point, Z_point, color=whole_dataset_points_colors[point], alpha=1,
                       label=whole_dataset_verbal_labels[point])

        else:

            ax.scatter(X_point, Y_point, color=whole_dataset_points_colors[point], alpha=1,
                       label=whole_dataset_verbal_labels[point])

    ymin_original, ymax_original = ax.get_ylim()
    xmin_original, xmax_original = ax.get_xlim()

    # ax.axis('square')

    ax.set_xlabel('component 1 (X)')
    ax.set_ylabel('component 2 (Y)')

    if reducted_inputs.shape[1] == 3:

        zmin_original, zmax_original = ax.get_zlim()
        ax.set_zlabel('component 3 (Z)')

    Legend_elements = [
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[0],
                   color=Plot_points_colors[0], label=verbal_foundamental_labels[0], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[1],
                   color=Plot_points_colors[1], label=verbal_foundamental_labels[1], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[2],
                   color=Plot_points_colors[2], label=verbal_foundamental_labels[2], markersize=10),
        plt.Line2D([0], [0], linestyle='', marker='o', markerfacecolor=Plot_points_colors[3],
                   color=Plot_points_colors[3], label=verbal_foundamental_labels[3], markersize=10)
    ]

    ax.legend(handles=Legend_elements, loc='best')

    # LOOP FOR POINTS DRAWING - TEMPORAL SEQUENCE

    trasp = 0
    trasp_single = 0.95 / reducted_activations_temporal_sequence.shape[0]

    for point in np.arange(0, reducted_activations_temporal_sequence.shape[0]):  # LOOP FOR DRAWING POINT

        trasp += trasp_single

        X_point = reducted_activations_temporal_sequence[point, 0]
        Y_point = reducted_activations_temporal_sequence[point, 1]

        if reducted_activations_temporal_sequence.shape[1] == 3:

            Z_point = reducted_activations_temporal_sequence[point, 2]

        if point == 0:

            color = 'green'

            if reducted_activations_temporal_sequence.shape[1] == 3:

                bx.scatter(X_point, Y_point, Z_point, color=color, alpha=1, s=50, label='Start point')

            else:

                bx.scatter(X_point, Y_point, color=color, alpha=1, s=50, label='Start point')

        elif point == reducted_activations_temporal_sequence.shape[0] - 1:

            color = 'red'

            if reducted_activations_temporal_sequence.shape[1] == 3:

                bx.scatter(X_point, Y_point, Z_point, color=color, alpha=1, s=50, label='Stop point')

            else:

                bx.scatter(X_point, Y_point, color=color, alpha=1, s=50, label='Stop point')

        else:

            color = 'k'

            if reducted_activations_temporal_sequence.shape[1] == 3:

                bx.scatter(X_point, Y_point, Z_point, color=color, alpha=trasp) # change alpha with 0.5 for uniform coloured spots

            else:

                bx.scatter(X_point, Y_point, color=color, alpha=trasp) # change alpha with 0.5 for uniform coloured spots


        # bx.set_xlim(xmin_original, xmax_original)
        # bx.set_ylim(ymin_original, ymax_original)

        # bx.axis('square')

        bx.set_xlabel('component 1 (X)')
        bx.set_ylabel('component 2 (Y)')

        if reducted_activations_temporal_sequence.shape[1] == 3:

            # bx.set_zlim(zmin_original, zmax_original)

            bx.set_zlabel('component 3 (Z)')

        bx.legend(loc='lower right')

    return Fig

def Window_builder_tester_performances(Losses, layer_label, salient_feature):

        '''

        This function creates a windows with a loss curve. The curve corresponds to a perceptron loss that
        learns to classify the specific attributes (e.g. in case of 'color category' are 'red, green, blue, yellow') from
        an input. The input is extracted from RBM and can be the the visible layer,  the hidden layer reconstructed from second hidden or
         the second hidden.

        args:

            Losses:  errors of perceptron depending.

            layer_label: string of format "visible/hidden". Useful for title of window.

        return:

            Fig: figure with the curve that represents the loss of perceptron

        '''

        Fig = plt.figure(figsize=(19.20,10.80))

        Losses = np.array(Losses)


        plt.axhline(y = 0.01, linestyle='--', color='blue', label= 'target error (0.01)')


        min_err = np.min(Losses)



        if min_err > 0.01:

            achiev_status =  str('Min_err = ') + str(min_err) + str(' (target err not achieved)')

        else:

            pos_target_err_achieved = np.where(Losses < 0.01)[0]

            achiev_status = str('Min_err = ') + str(min_err) + str(' (target err achieved at epoc ') + str(pos_target_err_achieved[0]) + str(')')




        Loss_plot = plt.plot(range(0, len(Losses)), Losses, linestyle='-', color='blue', label=  achiev_status)



        plt.xlabel('Epoc')
        plt.ylabel('Error')

        if layer_label == 'visible':

            plt.title('feature-dependent extrinsic task (VISIBLE) \n salient feature of task: ' + str(salient_feature), fontsize=15, fontweight = 'bold')

        elif layer_label == 'hidden':

            plt.title('feature-dependent extrinsic task (HIDDENS - FIRST LAYER) \n salient feature of task: ' + str(salient_feature), fontsize=15, fontweight = 'bold')


        elif layer_label == 'hidden second':

            plt.title('feature-dependent extrinsic task (HIDDENS - SECOND LAYER) \n salient feature of task: ' + str(
                salient_feature), fontsize=15, fontweight='bold')

        else:

            plt.title('Rewards/Accuracies (Robotic-like enviroment training) \n salient feature of task: ' + str(
                salient_feature), fontsize=15, fontweight='bold')

        plt.legend(loc='upper right')

        return Fig


def Window_builder_tester_performances_DOUBLE_COMPARISON(Losses, layer_label, salient_feature):
    '''

    This function is a copy of "Window_builder_tester_performances" function.
    In this version (DOUBLE_COMPARISON) there is a visual comparison between two perceptron (one for the modified version of NET training e one for a default net training)


    args:

        Losses: list of two lists. The first list contains the errors of perceptron depending on "not biased represent.".
                Conversely, the second list contains the errors caused by biased represent.

        layer_label: string of format "visible/hidden". Useful for title of window.

    return:

        Fig: figure with two curves that represents the loss of two perceptrons

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))

    Losses = np.array(Losses)

    Losses_NOT_B = Losses[:, 0]
    # Losses_NOT_B = pd.DataFrame({'Serie': Losses_NOT_B})
    # Losses_NOT_B = Losses_NOT_B.rolling(1).mean()
    # ii = np.isfinite(Losses_NOT_B)
    # Losses_NOT_B = Losses_NOT_B[ii]
    # Losses_NOT_B = list(Losses_NOT_B)

    Losses_B = Losses[:, 1]
    # Losses_B = pd.DataFrame({'Serie': Losses_B})
    # Losses_B = Losses_B.rolling(1).mean()        #
    # ii = np.isfinite(Losses_B)
    # Losses_B = Losses_B[ii]
    # Losses_B = list(Losses_B)

    plt.axhline(y=0.01, linestyle='--', color='blue', label='target error (0.01)')

    min_err_B = np.min(Losses_B)

    min_err_NOT_B = np.min(Losses_NOT_B)

    if min_err_B > 0.01:

        achiev_status_B = str(' min err not achieved - min_err = ') + str(min_err_B) + str(')')

    else:

        pos_target_err_achieved_B = np.where(Losses_B < 0.01)[0]

        achiev_status_B = str(' target err achieved at epoc ') + str(pos_target_err_achieved_B[0]) + str(')')

    if min_err_NOT_B > 0.01:

        achiev_status_NOT_B = str(' min err not achieved - min_err = ') + str(min_err_NOT_B) + str(')')

    else:

        pos_target_err_achieved_not_B = np.where(Losses_NOT_B < 0.01)[0]

        achiev_status_NOT_B = str(' target err achieved at epoc ') + str(pos_target_err_achieved_not_B[0]) + str(')')

    if min_err_B > 0.01 or min_err_NOT_B > 0.01:

        winner_index = np.argmin([min_err_B, min_err_NOT_B])

    else:

        winner_index = np.argmin([pos_target_err_achieved_B[0], pos_target_err_achieved_not_B[0]])

    if winner_index == 0:

        label_b = 'Biased (Winner,'
        label_not_b = 'Not Biased (Loser,'
    else:

        label_b = 'Biased (Loser,'
        label_not_b = 'Not Biased (Winner,'

    Loss_plot_B = plt.plot(range(0, len(Losses)), Losses_B, linestyle='-', color='blue',
                           label=label_b + achiev_status_B)

    Loss_plot_NOT_B = plt.plot(range(0, len(Losses)), Losses_NOT_B, linestyle='-', color='red',
                               label=label_not_b + achiev_status_NOT_B)

    plt.fill_between(range(0, len(Losses)), Losses_B, Losses_NOT_B, facecolor='green')

    plt.xlabel('Epoc')
    plt.ylabel('Error')

    if layer_label == 'visible':

        plt.title('feature-dependent extrinsic task (VISIBLE) \n salient feature of task: ' + str(salient_feature),
                  fontsize=15, fontweight='bold')

    elif layer_label == 'hidden':

        plt.title('feature-dependent extrinsic task (HIDDENS - FIRST LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')


    elif layer_label == 'hidden second':

        plt.title('feature-dependent extrinsic task (HIDDENS - SECOND LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')

    plt.legend(loc='upper right')

    return Fig


def Window_builder_tester_performances_REINFORCE(layer_label, salient_feature, Rewards, Accuracies = False):
    '''

    This function is a variant o "Window_builder_tester_performances" function. In this variant I substitute the back-prop
    of perceptron that support the test (i.e. supervised learning) with a reward-based back-prop (see REINFORCE; Williams, 1992).
    In this case I plot the Reward and, optionally, the accuracy (in case of binary 0/1 reward, it corresponds to the accuracy)


    args:

        Rewards: list that contains the rewards of a single perceptron.

        Accuracies: same variable of "Losses" for accuracies

        layer_label: string of format "visible/hidden". Useful for title of window.

    return:

        Fig: figure with two curves that represents the loss of two perceptrons

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))

    Rewards = np.array(Rewards)

    Accuracies = np.array(Accuracies)

    plt.axhline(y=0.95, linestyle='--', color='black', label='Target Accuracy (0.95)')

    Max_Acc= np.around(np.max(Accuracies), decimals = 3)

    Max_R = np.around(np.max(Rewards), decimals = 3)

    if Max_Acc < 0.95:

        achiev_status = str(' Max_Acc = ') + str(Max_Acc) + str('(target Acc not achieved)')

    else:

        pos_target_Acc_achieved = np.where(Accuracies > 0.95)[0]

        achiev_status = str('Max Acc = ') + str(Max_Acc) + str('(target Acc achieved at epoc ') + str(pos_target_Acc_achieved[0]) + str(')')



    Accuracies_plot = plt.plot(range(0, len(Accuracies)), Accuracies, linestyle='-', color='cornflowerblue',
                           label= achiev_status)


    Rewards_plot = plt.plot(range(0, len(Rewards)), Rewards, linestyle='-', color='blue',
                           label= 'Max_R = ' + str(Max_R))



    plt.xlabel('Epoc')
    plt.ylabel('Reward/Accuracy')


    if layer_label == 'visible':

        plt.title('feature-dependent extrinsic task (VISIBLE) \n salient feature of task: ' + str(salient_feature),
                  fontsize=15, fontweight='bold')

    elif layer_label == 'hidden':

        plt.title('feature-dependent extrinsic task (HIDDENS - FIRST LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')


    elif layer_label == 'hidden second':

        plt.title('feature-dependent extrinsic task (HIDDENS - SECOND LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')

    else:

        plt.title('Rewards/Accuracies (Robotic-like enviroment training) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')



    plt.legend(loc="best")

    return Fig

def Window_builder_tester_performances_REINFORCE_DOUBLE_COMPARISON(layer_label, salient_feature, Rewards, Accuracies = False):

    '''

    This function is a copy of "Window_builder_tester_performances_REINFORCE" function. In this version (DOUBLE_COMPARISON)
    there is a visual comparison between two perceptron (one for the modified version of NET training e one for a default net training)


    args:

        Rewards: list of two lists. The first list contains the rewards of perceptron depending on "not biased represent.".
                Conversely, the second list contains the rewards caused by biased represent.

        Accuracies: same variable of "Losses" for accuracies

        layer_label: string of format "visible/hidden". Useful for title of window.

    return:

        Fig: figure with two curves that represents the loss of two perceptrons

    '''

    Fig = plt.figure(figsize=(19.20, 10.80))

    Rewards = np.array(Rewards)

    Rewards_NOT_B = Rewards[:, 0]
    # Rewards_NOT_B = pd.DataFrame({'Serie': Rewards_NOT_B})
    # Rewards_NOT_B = Rewards_NOT_B.rolling(10).mean()
    # Rewards_NOT_B = np.array(Rewards_NOT_B)
    # ii = np.isfinite(Rewards_NOT_B)
    # Rewards_NOT_B = Rewards_NOT_B[ii]

    Rewards_B = Rewards[:, 1]
    # Rewards_B = pd.DataFrame({'Serie': Rewards_B})
    # Rewards_B = Rewards_B.rolling(10).mean()        #
    # Rewards_B = np.array(Rewards_B)
    # ii = np.isfinite(Rewards_B)
    # Rewards_B = Rewards_B[ii]


    if not isinstance(Accuracies, (np.ndarray)):

        plt.axhline(y=0.95, linestyle='--', color='black', label='target reward/accuracy')

        Max_R_B = np.max(Rewards_B)

        Max_R_NOT_B = np.max(Rewards_NOT_B)

        if Max_R_B < 0.95:

            achiev_status_B = str(' target R not achieved - Max_R = ') + str(Max_R_B) + str(')')

        else:

            pos_target_R_achieved_B = np.where(Rewards_B > 0.95)[0]

            achiev_status_B = str(' target R achieved at epoc ') + str(pos_target_R_achieved_B[0]) + str(')')

        if Max_R_NOT_B < 0.95:

            achiev_status_NOT_B = str(' target R not achieved - Max_R = ') + str(Max_R_NOT_B) + str(')')

        else:

            pos_target_R_achieved_not_B = np.where(Rewards_NOT_B > 0.95)[0]

            achiev_status_NOT_B = str(' target R achieved at epoc ') + str(pos_target_R_achieved_not_B[0]) + str(')')

        if Max_R_B < 0.95 or Max_R_NOT_B < 0.95:

            winner_index = np.argmax([Max_R_B, Max_R_NOT_B])

        else:

            winner_index = np.argmax([pos_target_R_achieved_B[0], pos_target_R_achieved_not_B[0]])

        if winner_index == 0:

            label_b = 'Biased (Winner,'
            label_not_b = 'Not Biased (Loser,'
        else:

            label_b = 'Biased (Loser,'
            label_not_b = 'Not Biased (Winner,'

        Rewards_plot_B = plt.plot(range(0, len(Rewards)), Rewards_B, linestyle='-', color='blue',
                               label=label_b + achiev_status_B)

        Rewards_plot_NOT_B = plt.plot(range(0, len(Rewards)), Rewards_NOT_B, linestyle='-', color='red',
                                   label=label_not_b + achiev_status_NOT_B)

        plt.fill_between(range(0, len(Rewards)), Rewards_B, Rewards_NOT_B, facecolor='green')

        plt.xlabel('Epoc')
        plt.ylabel('Reward/Accuracy')

    else:

        Accuracies = np.array(Accuracies)

        Accuracies_NOT_B = Accuracies[:, 0]

        Accuracies_B = Accuracies[:, 1]

        plt.axhline(y=0.95, linestyle='--', color='black', label='target Accuracy (0.95)')

        Max_Acc_B = np.around(np.max(Accuracies_B), decimals = 3)

        Max_Acc_NOT_B = np.around(np.max(Accuracies_NOT_B), decimals = 3)

        Max_R_B = np.around(np.max(Rewards_B), decimals = 3)

        Max_R_NOT_B = np.around(np.max(Rewards_NOT_B), decimals = 3)

        if Max_Acc_B < 0.95:

            achiev_status_B = str(' target Acc not achieved - Max_Acc = ') + str(Max_Acc_B) + str(')')

        else:

            pos_target_Acc_achieved_B = np.where(Accuracies_B > 0.95)[0]

            achiev_status_B = str(' target Acc achieved at epoc ') + str(pos_target_Acc_achieved_B[0]) + str(')')

        if Max_Acc_NOT_B < 0.95:

            achiev_status_NOT_B = str(' target Acc not achieved - Max_Acc = ') + str(Max_Acc_NOT_B) + str(')')

        else:

            pos_target_Acc_achieved_not_B = np.where(Accuracies_NOT_B > 0.95)[0]

            achiev_status_NOT_B = str(' target Acc achieved at epoc ') + str(pos_target_Acc_achieved_not_B[0]) + str(')')

        if Max_Acc_B < 0.95 or Max_Acc_NOT_B < 0.95:

            winner_index = np.argmax([Max_Acc_B, Max_Acc_NOT_B])

        else:

            winner_index = np.argmax([pos_target_Acc_achieved_B[0], pos_target_Acc_achieved_not_B[0]])

        if winner_index == 0:

            label_b = 'Biased (Winner,'
            label_not_b = 'Not Biased (Loser,'
        else:

            label_b = 'Biased (Loser,'
            label_not_b = 'Not Biased (Winner,'

        Accuracies_plot_B = plt.plot(range(0, len(Accuracies)), Accuracies_B, linestyle='-', color='cornflowerblue',
                               label=label_b + achiev_status_B)

        Accuracies_plot_NOT_B = plt.plot(range(0, len(Accuracies)), Accuracies_NOT_B, linestyle='-', color='darkred',
                                   label=label_not_b + achiev_status_NOT_B)

        Rewards_plot_B = plt.plot(range(0, len(Rewards)), Rewards_B, linestyle='-', color='blue',
                               label= 'Biased Reward - Max_R = ' + str(Max_R_B))

        Rewards_plot_NOT_B = plt.plot(range(0, len(Rewards)), Rewards_NOT_B, linestyle='-', color='red',
                                   label='Not biased Reward - Max_R = ' + str(Max_R_NOT_B))

        plt.fill_between(range(0, len(Accuracies)), Accuracies_B, Accuracies_NOT_B, facecolor='green')

        plt.xlabel('Epoc')
        plt.ylabel('Reward/Accuracy')


    if layer_label == 'visible':

        plt.title('feature-dependent extrinsic task (VISIBLE) \n salient feature of task: ' + str(salient_feature),
                  fontsize=15, fontweight='bold')

    elif layer_label == 'hidden':

        plt.title('feature-dependent extrinsic task (HIDDENS - FIRST LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')


    elif layer_label == 'hidden second':

        plt.title('feature-dependent extrinsic task (HIDDENS - SECOND LAYER) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')

    else:

        plt.title('Rewards/Accuracies (Robotic-like enviroment training) \n salient feature of task: ' + str(
            salient_feature), fontsize=15, fontweight='bold')

    plt.legend(loc="best")

    return Fig

def Window_builder_Updates_comparison_CD_RL(min_history_CD, max_history_CD, min_history_RL, max_history_RL):


    '''

    This function compares the maximum and minimum values of update proposed by contrastive divergence and REINFORCE, with same
    conditions (same discount factor).

    args:

    - min_history_CD: history of min values of CD updates proposed
    - max_history_CD: same of previously variable for max values
    - min_history_RL: history of min values of RL updates proposed
    - max_history_RL: same of previously variable for max values

    return:

        Fig: windows with a plot che compares the histories arguments

    '''


    # rolling media computations

    rolling_value = 100

    # min_CD

    min_history_CD = copy.deepcopy(min_history_CD)
    min_history_CD = pd.DataFrame({'Serie': min_history_CD})
    min_history_CD = min_history_CD.rolling(rolling_value).mean()

    ii = np.isfinite(min_history_CD)
    surprises = min_history_CD[ii]

    # Max_CD

    max_history_CD = copy.deepcopy(max_history_CD)
    max_history_CD = pd.DataFrame({'Serie': max_history_CD})
    max_history_CD = max_history_CD.rolling(rolling_value).mean()

    ii = np.isfinite(max_history_CD)
    max_history_CD = max_history_CD[ii]

    # Min_RL

    min_history_RL = copy.deepcopy(min_history_RL)
    min_history_RL = pd.DataFrame({'Serie': min_history_RL})
    min_history_RL = min_history_RL.rolling(rolling_value).mean()

    ii = np.isfinite(min_history_RL)
    min_history_RL = min_history_RL[ii]

    # Max_RL

    max_history_RL = copy.deepcopy(max_history_RL)
    max_history_RL = pd.DataFrame({'Serie': max_history_RL})
    max_history_RL = max_history_RL.rolling(rolling_value).mean()

    ii = np.isfinite(max_history_RL)
    max_history_RL = max_history_RL[ii]

    Fig = plt.figure(figsize=(19.20, 10.80))

    plt.suptitle(' Localistic updates proposed by RL and CD \n (min and max update of weights updates matrix)')

    Min_values_plot = plt.subplot(1, 2, 1)

    Max_values_plot = plt.subplot(1, 2, 2)

    Min_values_plot.plot(range(0, len(min_history_CD)), min_history_CD, color = 'blue', label = ' min update value_CD')

    Min_values_plot.plot(range(0, len(min_history_RL)), min_history_RL, color = 'cornflowerblue', label = ' min update value_RL')

    Max_values_plot.plot(range(0, len(max_history_CD)), max_history_CD, color = 'red', label = ' max update value_CD')

    Max_values_plot.plot(range(0, len(max_history_RL)), max_history_RL, color = 'darkred', label = ' max update value_RL')

    Max_values_plot.set_ylabel('Local update')
    Max_values_plot.set_xlabel('Update_times')

    Max_values_plot.set_title(' Max local updates ', fontweight='bold')
    Max_values_plot.legend(loc="best")

    Min_values_plot.set_ylabel('Local update')
    Min_values_plot.set_xlabel('Update_times')

    Min_values_plot.set_title(' Min local updates ', fontweight='bold')
    Min_values_plot.legend(loc="best")


    return Fig




def Window_builder_check_panel_RBM(Reinforces, STDs_Reinforces, Surprises, STDs_Surprises, Reinforce_for_input,
                                             Surprise_for_input, Weight_inputs, W_NET_sum, W_bias_1_sum, W_bias_2_sum,
                                             Accuracies, STDs_Accuracies, Accuracy_for_input, Reconstruction_errors_epocs,
                                             learning_rate, learning_rate_critic, R_range, R_range_interp, ideals_actions, CD_weight = False, Executor_input = False
                                             ):
        '''

        This function creates a window with many visual informations regarding the learning of RBM (reinforcement version).

        ARGS:

            - Reinforces, STDs_Reinforces = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch reinforce (avg of dataset reinforces), while the second shows the corresponding stds

            - Surprises, STDs_Surprises = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch surprise (avg of dataset surprise), while the second shows the corresponding stds

            - Accuracies, STDs_Accuracies = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch accuracy (avg of dataset accuracy), while the second shows the corresponding stds

            - Reinforce_for_input, Surprise_for_input, Accuracy_for_input = Each variable is a single batch array corresponding to values for each input
              of last epoc. First values are reinforces for input, second values are the same for surprises and third values
              are the same for accuracies

            - Weight_inputs = weights of network (from visible to hidden)

            - W_NET_sum, W_bias_1_sum, W_bias_2_sum = sum of weights matrices of NETWORK, BIAS FROM INPUT TO HIDDEN, BIAS FROM HIDDEN TO INPUT

            - Reconstruction_errors_epocs = Unidimensional Arrays with lenght equal to epocs. Each value is
                the batch reconstruction error (avg of dataset reconstruction error)

            learning_rate = learning rate of CD (contrastive divergence)

            learning_rate_critic = learning rate of critic component of reinforcement algorithm

            R_range, R_range_interp = range of values that reward can show. The second is a linear interpolation from previous
                                      range (e.g. from [0, 10] to [0,1])

            ideals_actions = batch of ideal actions (one for each attribute, e.g. red, green...)


        return:

            - Fig:  window with many plots regarding the the learning of RBM reinforcement biased



            '''

        Fig = plt.figure(figsize=(19.20,10.80))
        Fig.clf()

        Fig.suptitle(' Learning - check panel (RL RBM)', fontsize=22, fontweight='bold')


        num_rows = 3
        num_columns = 4
        rolling_value = 300


        #          R PLOTS%%%%%%%%%%%

        Max_R = max(Reinforces)
        Max_R_std = STDs_Reinforces[np.where(Reinforces == Max_R)[-1][0]]

        # BARS PLOT______

        Bars_plot = plt.subplot(num_rows, num_columns, 1)

        Bars_plot.errorbar(range(0, len(Reinforces)), Reinforces, STDs_Reinforces, linestyle='-', marker='.',
                           color='red', label = 'Reinforce', ecolor = 'darkred')

        #Bars_plot.set_xlabel('Epocs')
        Bars_plot.set_ylabel('Reinforce')
        Bars_plot.set_title(' R +/- SD', fontweight = 'bold')
        Bars_plot.set_ylim(R_range_interp[0], R_range_interp[1])

        Bars_plot.legend(loc="best")

        # LINE PLOT______

        Line_plot = plt.subplot(num_rows, num_columns, 2)

        # moving media

        reinforces_list = copy.deepcopy(Reinforces)
        reinforces = pd.DataFrame({'Serie': Reinforces})
        reinforces = reinforces.rolling(rolling_value).mean()

        ii = np.isfinite(reinforces)
        FirstLayerErrors = reinforces[ii]

        x = np.arange(0, len(reinforces))
        y = reinforces

        Line_plot.plot(x, y, linestyle='-', marker='.', color='red', label='Reinforce')
        Line_plot.set_ylim(R_range_interp[0], R_range_interp[1])
        Line_plot.legend(loc="best")

        #Line_plot.set_xlabel('Epocs')
        Line_plot.set_ylabel('Reinforce')
        Line_plot.set_title('R (ROLLING MEDIA)', fontweight = 'bold')
        Line_plot.legend(loc="best")

        Bars_plot_inputs = plt.subplot(num_rows, num_columns, 3)

        # BARS INPUTS PLOT______

        Bars_plot_inputs.bar(range(0, len(Reinforce_for_input)), Reinforce_for_input, align='center', alpha=0.5, color='darkred', label='Reinforce for input')
        Bars_plot_inputs.axhline(y = np.mean(Reinforce_for_input), color='darkred', linestyle='solid')

        #Bars_plot_inputs.set_xlabel('Inputs')
        Bars_plot_inputs.set_ylabel('Reinforce')
        Bars_plot_inputs.set_title('R (each input)', fontweight = 'bold')
        Bars_plot_inputs.set_ylim(R_range_interp[0], R_range_interp[1])



        #          WEIGHTS PLOTS%%%%%%%%%%%

        Weights_plot = plt.subplot(num_rows, num_columns, 4)

        weights_flattened = np.reshape(Weight_inputs, (Weight_inputs.shape[0]* Weight_inputs.shape[1], ))

        sb.distplot(weights_flattened, kde = True, hist= True,color = 'darkgoldenrod')

        Weights_plot.set_title('Network Weights', fontweight = 'bold')

        #plt.pcolor(Weight_inputs, cmap='RdBu')

        # Weights_plot.set_xlabel('Hiddens units')
        # Weights_plot.set_ylabel('Inputs units')
        #plt.colorbar()




        #          SURPRISE PLOTS%%%%%%%%%%%

        Bars_plot_surprise = plt.subplot(num_rows, num_columns, 5)

        # BARS PLOT______

        Bars_plot_surprise.errorbar(range(0, len(Surprises)), Surprises, STDs_Surprises, linestyle='-', marker='.',
                           color='blue', label='Surprise', ecolor='cornflowerblue')

        #Bars_plot_surprise.set_xlabel('Epocs')
        Bars_plot_surprise.set_ylabel('Surprise')
        Bars_plot_surprise.set_title('S +/- SD', fontweight = 'bold')
        Bars_plot_surprise.legend(loc="best")

        # LINE PLOT______

        Line_plot_surprise = plt.subplot(num_rows, num_columns, 6)

        # moving media

        surprises_list = copy.deepcopy(Surprises)
        surprises = pd.DataFrame({'Serie': Surprises})
        surprises = surprises.rolling(rolling_value).mean()

        ii = np.isfinite(surprises)
        surprises = surprises[ii]

        x = np.arange(0, len(surprises))
        y = surprises

        # PLOT BASELINE

        baseline = reinforces - surprises
        Line_plot.plot(x, baseline, linestyle='-', color='green', label='Baseline', linewidth = 1)
        Line_plot.legend(loc="best")

        Line_plot_surprise.plot(x, y, linestyle='-', marker='.', color='blue', label='Surprise')

        #Line_plot_surprise.set_xlabel('Epocs')
        Line_plot_surprise.set_ylabel('Surprise')
        Line_plot_surprise.set_title('S (ROLLING MEDIA)', fontweight = 'bold')
        Line_plot_surprise.legend(loc="best")

        Bars_plot_inputs_surprise = plt.subplot(num_rows, num_columns, 7)

        # BARS INPUTS PLOT______

        Surprise_for_input = np.reshape(Surprise_for_input, (len(Surprise_for_input)))

        Bars_plot_inputs_surprise.bar(range(0, len(Surprise_for_input)), Surprise_for_input, align='center', alpha=0.5,
                             color='cornflowerblue', label='Surprise for input')

        Bars_plot_inputs_surprise.axhline(y = np.mean(Surprise_for_input), color='cornflowerblue', linestyle='solid')

        #Bars_plot_inputs_surprise.set_xlabel('Inputs')
        Bars_plot_inputs_surprise.set_ylabel('Surprise')
        Bars_plot_inputs_surprise.set_title('S (each input)', fontweight = 'bold')

        #          WEIGHTS PLOTS%%%%%%%%%%%

        Weights_plot_biases = plt.subplot(num_rows, num_columns, 8)

        Weights_plot_biases.plot(range(0, len(W_NET_sum)), np.array(W_NET_sum) / (Weight_inputs.shape[0] * Weight_inputs.shape[1]), linestyle='-', color='blue', label='AVG_Network_W')
        Weights_plot_biases.plot(range(0, len(W_bias_1_sum)), np.array(W_bias_1_sum) / Weight_inputs.shape[1], linestyle=':', color='red', label='AVG_bias1_W (inputs -> hiddens)')
        Weights_plot_biases.plot(range(0, len(W_bias_2_sum)), np.array(W_bias_2_sum) / Weight_inputs.shape[0], linestyle=':', color='black',label='AVG_bias2_W (hiddens -> inputs)')

        #Weights_plot_biases.set_xlabel('Epocs')
        Weights_plot_biases.set_ylabel('AVG_Weights')
        Weights_plot_biases.set_title(' AVG_W', fontweight = 'bold')

        Weights_plot_biases.legend(loc="best")

        #           ACCURACIES PLOTS%%%%%%%%%%%

        Max_Acc = max(Accuracies)
        Max_Acc_std = STDs_Reinforces[np.where(Accuracies == Max_Acc)[-1][0]]


        Bars_plot_acc = plt.subplot(num_rows, num_columns, 9)

        # BARS PLOT______

        Bars_plot_acc.errorbar(range(0, len(Accuracies)), Accuracies, STDs_Accuracies, linestyle='-', marker='.',
                           color='black', label='Accuracy', ecolor='grey')

        Bars_plot_acc.set_xlabel('Epocs')
        Bars_plot_acc.set_ylabel('Accuracy')
        Bars_plot_acc.set_title(' Acc +/- SD', fontweight = 'bold')
        Bars_plot_acc.set_ylim(0, 1)

        Bars_plot_acc.legend(loc="best")

        # LINE PLOT______

        Line_plot_acc = plt.subplot(num_rows, num_columns,  10)

        # moving media

        accuracies_list = copy.deepcopy(Accuracies)
        accuracies = pd.DataFrame({'Serie': Accuracies})
        accuracies = accuracies.rolling(rolling_value).mean()

        ii = np.isfinite(accuracies)
        accuracies = accuracies[ii]

        x = np.arange(0, len(accuracies))
        y = accuracies

        Line_plot_acc.plot(x, y, linestyle='-', marker='.', color='black', label='Accuracy')
        Line_plot_acc.legend(loc="best")

        Line_plot_acc.set_xlabel('Epocs')
        Line_plot_acc.set_ylabel('Accuracy')
        Line_plot_acc.set_title('Acc (ROLLING MEDIA)', fontweight = 'bold')
        Line_plot_acc.set_ylim(0, 1)
        Line_plot_acc.legend(loc="best")

        Bars_plot_inputs_acc = plt.subplot(num_rows, num_columns,  11)

        # BARS INPUTS PLOT______

        Bars_plot_inputs_acc.bar(range(0, len(Accuracy_for_input)), Accuracy_for_input, align='center', alpha=0.5,
                             color='black', label='Accuracy for input')

        Bars_plot_inputs_acc.axhline(y = np.mean(Accuracy_for_input), color='black', linestyle='solid')


        Bars_plot_inputs_acc.set_xlabel('Inputs')
        Bars_plot_inputs_acc.set_ylabel('Accuracy')
        Bars_plot_inputs_acc.set_title('Acc (each input)', fontweight = 'bold')
        Bars_plot_inputs_acc.set_ylim(0, 1)


        #             RECONSTRUCTION ERRORS PLOTS%%%%%%%%%%%

        REC_plot = plt.subplot(num_rows, num_columns, 12)

        Min_Rec_err = min(Reconstruction_errors_epocs)
        Max_Rec_err = max(Reconstruction_errors_epocs)


        # moving media

        Errors_list_REC = copy.deepcopy(Reconstruction_errors_epocs)
        Errors_REC = pd.DataFrame({'Serie': Reconstruction_errors_epocs})
        Errors_REC = Errors_REC.rolling(rolling_value).mean()

        ii = np.isfinite(Errors_REC)
        Errors_REC = Errors_REC[ii]

        x = np.arange(0, len(Errors_REC))
        y = Errors_REC

        REC_plot.plot(x, y, linestyle='-', marker='.', color='green', label='RBM REC. ERROR')

        REC_plot.legend(loc="best")

        REC_plot.set_xlabel('Epocs')
        REC_plot.set_ylabel('ERROR')
        REC_plot.set_title('Reconstr. Errors', fontweight = 'bold')

        REC_plot.legend(loc="best")

        # TEXT______

        Line_plot.text(0.3, 0.5, ' Max_R = ' + str(np.around(Max_R, decimals=3)) + str(" +/- ") + str(
            np.around(Max_R_std, decimals=3)), fontsize=8, fontweight='bold')

        Line_plot_acc.text(0, 0.5, ' Max_Acc = ' + str(np.around(Max_Acc, decimals=3)) + str(" +/- ") +
                 str(np.around(Max_Acc_std, decimals=3)), fontsize=8,
                 fontweight='bold')

        REC_plot.text(0.5, (Max_Rec_err + Min_Rec_err)/2, ' Min_Err = ' + str(np.around(Min_Rec_err, decimals=3)), fontsize=8,
                 fontweight='bold')

        # + str('\n Max stability = ') + str(Max_stability)

        Line_plot.text(0.3, 0.5, ' Max_R = ' + str(np.around(Max_R, decimals=3)) + str(" +/- ") + str(
            np.around(Max_R_std, decimals=3)), fontsize=8, fontweight='bold')

        if isinstance(CD_weight, (bool)):

            Fig.text(0.1, 0.04, 'L_Rate (CD) =  ' + str(learning_rate), fontsize = 10, fontweight='bold')

        else:
            Fig.text(0.1, 0.04, 'CD_Weight (CD) =  ' + str(CD_weight), fontsize = 10, fontweight='bold')

        if isinstance(Executor_input, (str)):

            Fig.text(0.52, 0.04, 'Controller Input =  ' + str(Executor_input), fontsize=10, fontweight='bold')

        Fig.text(0.3, 0.04, 'L_Rate (Critic) =  ' + str(learning_rate_critic), fontsize=10, fontweight='bold')


        # Fig.text(0.4, 0.04, 'Reinforce_range =  ' + str(R_range), fontsize=10, fontweight='bold')
        #Fig.text(0.52, 0.04, 'Reinforce_range_interp =  ' + str(R_range_interp), fontsize=10, fontweight='bold')


        if ideals_actions.shape[1] > 4:

            Fig.text(0.7, 0.04,' IDEAL ACTIONS = DISTRIBUTED FORM (BINARY UNITS) ', fontsize = 10, fontweight = 'bold')

        else:

            Fig.text(0.7, 0.04,' IDEAL ACTIONS = LOCALISTIC FORM (ONEHOTVECTORS) ', fontsize=10, fontweight = 'bold')

        return Fig

def Window_builder_check_panel_DBN(Reinforces, STDs_Reinforces, Surprises, STDs_Surprises, Reinforce_for_input,
                                             Surprise_for_input, Weight_inputs, W_NET_sum, W_bias_1_sum, W_bias_2_sum, Accuracies,
                                             STDs_Accuracies, Accuracy_for_input, Max_stability,
                                             Weight_inputs_FIRST_RBM, W_NET_sum_FIRST_RBM, W_bias_1_sum_FIRST_RBM, W_bias_2_sum_FIRST_RBM,
                                             Reconstruction_errors_epocs, Reconstruction_errors_epocs_FIRST_RBM, learning_rate,
                                             learning_rate_FIRST_RBM, learning_rate_critic, R_range, R_range_interp, ideals_actions):

        '''

        This function creates a window with many visual informations regarding the learning of DBN (reinforcement version).
        
        Premise: in this function there are variables of first RBM (from input to first hidden) and second RBM (from first hidden
        to second hidden). Since that, in case of same variable I use a specific  terminology, e.g. "Weight_inputs_FIRST_RBM" and 
        "Weight_inputs" respectively for weights of first RBM and second RBM (In case of variable without specific reference as "Weight_inputs"
        I refer to the second RBM).
        
        ARGS:

            - Reinforces, STDs_Reinforces = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch reinforce (avg of dataset reinforces), while the second shows the corresponding stds

            - Surprises, STDs_Surprises = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch surprise (avg of dataset surprise), while the second shows the corresponding stds

            - Accuracies, STDs_Accuracies = Unidimensional Arrays with lenght equal to epocs. Each value of first is
                the batch accuracy (avg of dataset accuracy), while the second shows the corresponding stds

            - Max_stability = the max number of epocs where accuracy is fixed to 1 without changes

            - Reinforce_for_input, Surprise_for_input = Each variable is a single batch array corresponding to values for each input
              of last epoc. First values are reinforces for input and second values are the same for surprises

            - Weight_inputs = weights of network (from visible to hidden)

            - W_NET_sum, W_bias_1_sum, W_bias_2_sum = sum of weights matrices of NETWORK, BIAS FROM INPUT TO HIDDEN, BIAS FROM HIDDEN TO INPUT

            - Reconstruction_errors_epocs = Unidimensional Arrays with lenght equal to epocs. Each value is
                the batch reconstruction error (avg of dataset reconstruction error)

            learning_rate = learning rate of CD (contrastive divergence)

            learning_rate_critic = learning rate of critic component of reinforcement algorithm

            R_range, R_range_interp = range of values that reward can show. The second is a linear interpolation from previous
                                      range (e.g. from [0, 10] to [0,1])

            ideals_actions = batch of ideal actions (one for each attribute, e.g. red, green...)


        return:

            - Fig:  window with many plots regarding the the learning of RBM reinforcement biased



            '''



        Fig = plt.figure(1, figsize=(19.20, 10.80))
        Fig.clf()


        Fig.suptitle(' Learning - check panel (RL DBN)', fontsize=22, fontweight='bold')

        num_rows = 3
        num_columns = 5
        rolling_value = 300

        #          R PLOTS%%%%%%%%%%%

        Max_R = max(Reinforces)
        Max_R_std = STDs_Reinforces[np.where(Reinforces == Max_R)[-1][0]]

        # BARS PLOT______

        Bars_plot = plt.subplot(num_rows, num_columns, 1)

        Bars_plot.errorbar(range(0, len(Reinforces)), Reinforces, STDs_Reinforces, linestyle='-', marker='.',
                           color='red', label='Reinforce', ecolor='darkred')

        # Bars_plot.set_xlabel('Epocs')
        Bars_plot.set_ylabel('Reinforce')
        Bars_plot.set_title(' R +/- SD', fontweight='bold')

        Bars_plot.legend(loc="best")

        # LINE PLOT______

        Line_plot = plt.subplot(num_rows, num_columns, 2)

        # moving media

        reinforces_list = copy.deepcopy(Reinforces)
        reinforces = pd.DataFrame({'Serie': Reinforces})
        reinforces = reinforces.rolling(rolling_value).mean()

        ii = np.isfinite(reinforces)
        reinforces = reinforces[ii]

        x = np.arange(0, len(reinforces))
        y = reinforces

        Line_plot.plot(x, y, linestyle='-', marker='.', color='red', label='Reinforce')
        Line_plot.legend(loc="best")

        # Line_plot.set_xlabel('Epocs')
        Line_plot.set_ylabel('Reinforce')
        Line_plot.set_title('R (ROLLING MEDIA)', fontweight='bold')
        Line_plot.legend(loc="best")

        Bars_plot_inputs = plt.subplot(num_rows, num_columns, 3)

        # BARS INPUTS PLOT______

        Bars_plot_inputs.bar(range(0, len(Reinforce_for_input)), Reinforce_for_input, align='center', alpha=0.5,
                             color='darkred', label='Reinforce for input')
        Bars_plot_inputs.axhline(y=np.mean(Reinforce_for_input), color='darkred', linestyle='solid')

        # Bars_plot_inputs.set_xlabel('Inputs')
        Bars_plot_inputs.set_ylabel('Reinforce')
        Bars_plot_inputs.set_title('R (each Input)', fontweight='bold')


        #          WEIGHTS PLOTS%%%%%%%%%%%


        Weights_plot_first = plt.subplot(num_rows, num_columns, 4)

        plt.pcolor(Weight_inputs_FIRST_RBM, cmap='RdBu')

        #Weights_plot_first.set_xlabel('Hiddens')
        #Weights_plot_first.set_ylabel('Inputs')
        Weights_plot_first.set_title('Weights (RBM 1) ', fontweight='bold')
        plt.colorbar()

        Weights_plot = plt.subplot(num_rows, num_columns, 5)

        plt.pcolor(Weight_inputs, cmap='RdBu')

        #Weights_plot.set_xlabel('Hiddens')
        #Weights_plot.set_ylabel('Inputs')
        Weights_plot.set_title(' Weights (RBM 2)', fontweight='bold')
        plt.colorbar()

        #          SURPRISE PLOTS%%%%%%%%%%%

        Bars_plot_surprise = plt.subplot(num_rows, num_columns, 6)

        # BARS PLOT______

        Bars_plot_surprise.errorbar(range(0, len(Surprises)), Surprises, STDs_Surprises, linestyle='-', marker='.',
                                    color='blue', label='Surprise', ecolor='cornflowerblue')

        # Bars_plot_surprise.set_xlabel('Epocs')
        Bars_plot_surprise.set_ylabel('Surprise')
        Bars_plot_surprise.set_title('S +/- SD', fontweight='bold')
        Bars_plot_surprise.legend(loc="best")

        # LINE PLOT______

        Line_plot_surprise = plt.subplot(num_rows, num_columns, 7)

        # moving media

        surprises_list = copy.deepcopy(Surprises)
        surprises = pd.DataFrame({'Serie': Surprises})
        surprises = surprises.rolling(rolling_value).mean()

        ii = np.isfinite(surprises)
        surprises = surprises[ii]

        x = np.arange(0, len(surprises))
        y = surprises

        # PLOT BASELINE

        baseline = reinforces-surprises

        Line_plot.plot(x, baseline, linestyle='-', color='green', label='Baseline', linewidth = 1)
        Line_plot.legend(loc="best")

        Line_plot_surprise.plot(x, y, linestyle='-', marker='.', color='blue', label='Surprise')


        # Line_plot_surprise.set_xlabel('Epocs')
        #Line_plot_surprise.set_ylabel('Surprise')
        Line_plot_surprise.set_title('S (ROLLING MEDIA)', fontweight='bold')
        Line_plot_surprise.legend(loc="best")

        Bars_plot_inputs_surprise = plt.subplot(num_rows, num_columns, 8)

        # BARS INPUTS PLOT______

        Surprise_for_input = np.reshape(Surprise_for_input, (len(Surprise_for_input)))
        X = range(0, len(Surprise_for_input))

        Bars_plot_inputs_surprise.bar(X, Surprise_for_input, align='center', alpha=0.5,
                                      color='cornflowerblue', label='Surprise for input')

        Bars_plot_inputs_surprise.axhline(y = np.mean(Surprise_for_input), color='cornflowerblue', linestyle='solid')

        # Bars_plot_inputs_surprise.set_xlabel('Inputs')
        Bars_plot_inputs_surprise.set_ylabel('Surprise')
        Bars_plot_inputs_surprise.set_title('S (each input)', fontweight='bold')

        #          WEIGHTS PLOTS%%%%%%%%%%%


        Weights_plot_biases_FIRST_RBM = plt.subplot(num_rows, num_columns, 9)

        Weights_plot_biases_FIRST_RBM.plot(range(0, len(W_NET_sum_FIRST_RBM)), np.array(W_NET_sum_FIRST_RBM) /
                                           (Weight_inputs_FIRST_RBM.shape[0] * Weight_inputs_FIRST_RBM.shape[1]),
                                           linestyle='-', color='blue', label='AVG_W')

        Weights_plot_biases_FIRST_RBM.plot(range(0, len(W_bias_1_sum_FIRST_RBM)), np.array(W_bias_1_sum_FIRST_RBM) /
                                           Weight_inputs_FIRST_RBM.shape[1], linestyle=':', color='red',
                                 label='AVG_bias1_W (Inp -> Hid)')
        Weights_plot_biases_FIRST_RBM.plot(range(0, len(W_bias_2_sum_FIRST_RBM)), np.array(W_bias_2_sum_FIRST_RBM) /
                                           Weight_inputs_FIRST_RBM.shape[0], linestyle=':', color='black',
                                 label='AVG_bias2_W (Hid -> Inp)')

        #Weights_plot_biases_FIRST_RBM.set_xlabel('Epocs')
        #Weights_plot_biases_FIRST_RBM.set_ylabel('AVG_Weights')
        Weights_plot_biases_FIRST_RBM.set_title(' AVG W (RBM 1)', fontweight='bold')

        Weights_plot_biases_FIRST_RBM.legend(loc = 'lower right', fontsize = 'x-small')


        Weights_plot_biases = plt.subplot(num_rows, num_columns, 10)

        Weights_plot_biases.plot(range(0, len(W_NET_sum)), np.array(W_NET_sum) /  (Weight_inputs.shape[0] * Weight_inputs.shape[1]),
                                 linestyle='-', color='blue', label='AVG_W')
        Weights_plot_biases.plot(range(0, len(W_bias_1_sum)), np.array(W_bias_1_sum) / Weight_inputs.shape[1],
                                 linestyle=':', color='red', label='AVG_bias1_W (Inp -> Hid)')
        Weights_plot_biases.plot(range(0, len(W_bias_2_sum)), np.array(W_bias_2_sum) / Weight_inputs.shape[0],
                                 linestyle=':', color='black',label='AVG_bias2_W (Hid -> Inp)')

        #Weights_plot_biases.set_xlabel('Epocs')
        #Weights_plot_biases.set_ylabel('AVG_Weights')
        Weights_plot_biases.set_title('AVG W (RBM 2)', fontweight='bold')

        Weights_plot_biases.legend(loc = 'lower right', fontsize = 'x-small')

        #           ACCURACIES PLOTS%%%%%%%%%%%

        Max_Acc = max(Accuracies)
        Max_Acc_std = STDs_Reinforces[np.where(Accuracies == Max_Acc)[-1][0]]

        Bars_plot_acc = plt.subplot(num_rows, num_columns, 11)

        # BARS PLOT______

        Bars_plot_acc.errorbar(range(0, len(Accuracies)), Accuracies, STDs_Accuracies, linestyle='-', marker='.',
                               color='black', label='Accuracy', ecolor='grey')

        Bars_plot_acc.set_xlabel('Epocs')
        Bars_plot_acc.set_ylabel('Accuracy')
        Bars_plot_acc.set_title(' Acc +/- SD', fontweight='bold')

        Bars_plot_acc.legend(loc="best")

        # LINE PLOT______

        Line_plot_acc = plt.subplot(num_rows, num_columns, 12)

        # moving media

        accuracies_list = copy.deepcopy(Accuracies)
        accuracies = pd.DataFrame({'Serie': Accuracies})
        accuracies = accuracies.rolling(rolling_value).mean()

        ii = np.isfinite(accuracies)
        accuracies = accuracies[ii]

        x = np.arange(0, len(accuracies))
        y = accuracies

        Line_plot_acc.plot(x, y, linestyle='-', marker='.', color='black', label='Accuracy')
        Line_plot_acc.legend(loc="best")

        Line_plot_acc.set_xlabel('Epocs')
        Line_plot_acc.set_ylabel('Accuracy')
        Line_plot_acc.set_title('Acc (ROLLING MEDIA)', fontweight='bold')

        Line_plot_acc.legend(loc="best")

        Bars_plot_inputs_acc = plt.subplot(num_rows, num_columns, 13)

        # BARS INPUTS PLOT______

        Bars_plot_inputs_acc.bar(range(0, len(Accuracy_for_input)), Accuracy_for_input, align='center', alpha=0.5,
                                 color='black', label='Accuracy for input')

        Bars_plot_inputs_acc.axhline(y=np.mean(Accuracy_for_input), color='black', linestyle='solid')

        Bars_plot_inputs_acc.set_xlabel('Inputs')
        Bars_plot_inputs_acc.set_ylabel('Accuracy')
        Bars_plot_inputs_acc.set_title('Acc (each input)', fontweight='bold')



        #             RECONSTRUCTION ERRORS PLOTS%%%%%%%%%%%

        REC_plot = plt.subplot(num_rows, num_columns, 14)

        # - FIRST RBM

        # moving media

        Errors_list_REC_FIRST_RBM = copy.deepcopy(Reconstruction_errors_epocs_FIRST_RBM)
        Errors_REC_FIRST_RBM = pd.DataFrame({'Serie': Reconstruction_errors_epocs_FIRST_RBM})
        Errors_REC_FIRST_RBM = Errors_REC_FIRST_RBM.rolling(rolling_value).mean()

        ii = np.isfinite(Errors_REC_FIRST_RBM)
        Errors_REC_FIRST_RBM = Errors_REC_FIRST_RBM[ii]

        x = np.arange(0, len(Errors_REC_FIRST_RBM))
        y = Errors_REC_FIRST_RBM

        REC_plot.plot(x, y, linestyle='-', marker='.', color='khaki', label='FIRST RBM REC. ERROR')

        # Line_plot.set_xlabel('Epocs')
        REC_plot.set_ylabel('ERROR')
        REC_plot.set_title('Reconstr. Errors (FIRST RBM)', fontweight='bold')


        # - SECOND RBM

        # moving media

        Errors_list_REC = copy.deepcopy(Reconstruction_errors_epocs)
        Errors_REC = pd.DataFrame({'Serie': Reconstruction_errors_epocs})
        Errors_REC = Errors_REC.rolling(rolling_value).mean()

        ii = np.isfinite(Errors_REC)
        FirstLayerErrors_REC = Errors_REC[ii]

        x = np.arange(0, len(Errors_REC))
        y = Errors_REC

        REC_plot.plot(x, y, linestyle='-', marker='.', color='green', label='RBM REC. ERROR')

        REC_plot    .set_xlabel('Epocs')
        REC_plot.set_ylabel('ERROR')
        REC_plot.set_title('Reconstr. Errors', fontweight='bold')

        REC_plot.legend(loc="best")

        # TEXT______

        Fig.text(0.75, 0.3, ' Max_R = ' + str(np.around(Max_R, decimals=3)) + str(" +/- ") + str(
            np.around(Max_R_std, decimals=3)), fontsize=10, fontweight='bold')

        Fig.text(0.75, 0.2, ' Max_Acc = ' + str(np.around(Max_Acc, decimals=3)) + str(" +/- ") +
                           str(np.around(Max_Acc_std, decimals=3)) + str('\n\n\n Max stability = ') + str(Max_stability),
                           fontsize=10,
                           fontweight='bold')

        Fig.text(0.1, 0.04, 'L_Rate (CD, FIRST/SECOND) =  ' + str(learning_rate_FIRST_RBM) + str('/') + str(learning_rate),
                 fontsize=10, fontweight='bold')
        Fig.text(0.1, 0.02, 'L_Rate (Critic) =  ' + str(learning_rate_critic), fontsize=10, fontweight='bold')

        Fig.text(0.4, 0.04, 'Reinforce_range =  ' + str(R_range), fontsize=10, fontweight='bold')
        Fig.text(0.4, 0.02, 'Reinforce_range_interp =  ' + str(R_range_interp), fontsize=10, fontweight='bold')

        if ideals_actions.shape[1] > 4:

            Fig.text(0.7, 0.04, ' IDEAL ACTIONS = DISTRIBUTED FORM (BINARY UNITS) ', fontsize=10, fontweight='bold')

        else:

            Fig.text(0.7, 0.04, ' IDEAL ACTIONS = LOCALISTIC FORM (ONEHOTVECTORS) ', fontsize=10, fontweight='bold')

        # Fig.text(0.20, 0.95, ' Max_R (Batch) = ' + str(np.around(Max_R, decimals=3)) + str(" +/- ") + str(
        #     np.around(Max_R_std, decimals=3)), fontsize=15)
        #
        # Fig.text(0.60, 0.95, ' Max_Acc (Batch) = ' + str(np.around(Max_Acc, decimals=3)) + str(" +/- ") +
        #          str(np.around(Max_Acc_std, decimals=3)) + str(', Max stability = ') + str(Max_stability), fontsize=15)
        #
        # if ideals_actions.shape[1] > 4:
        #
        #     Fig.text(0.75, 0.155,' IDEAL ACTIONS = DISTRIBUTED FORM (BINARY UNITS) ', fontsize=10, fontweight = 'bold')
        #
        # else:
        #
        #     Fig.text(0.75, 0.155,' IDEAL ACTIONS = LOCALISTIC FORM (ONEHOTVECTORS) ', fontsize=10, fontweight = 'bold')
        #
        #
        # Fig.text(0.75, 0.30, 'L_Rate (CD, FIRST RBM) =  ' + str(learning_rate_FIRST_RBM), fontsize=10, fontweight='bold')
        # Fig.text(0.75, 0.28, 'L_Rate (CD, SECOND RBM) =  ' + str(learning_rate), fontsize=10, fontweight='bold')
        # Fig.text(0.75, 0.26, 'L_Rate (Critic) =  ' + str(learning_rate_critic), fontsize=10, fontweight='bold')
        #
        # Fig.text(0.75, 0.24, ' --------------- ', fontsize=10, fontweight='bold')
        #
        #
        # Fig.text(0.75, 0.22, 'Reinforce_range =  ' + str(R_range), fontsize=10, fontweight='bold')
        # Fig.text(0.75, 0.195, 'Reinforce_range_interp =  ' + str(R_range_interp), fontsize=10, fontweight='bold')
        #
        # Fig.text(0.75, 0.175, ' --------------- ', fontsize=10, fontweight='bold')
        #plt.pause(1)

        return Fig

#   ALTERNATIVE FUNCTIONS FOR OTHER VERSIONS OF RBM/DBN LEARNING%%%%%%%%%%%%%%%%%%%%%%%

#   REINFORCEMENT LEARNING MODIFICATION FUNCTIONS

def Potential_update_Reinforcement(Input_, output_sigm, output_bin, l_rate, surprise):
    '''

        This function propose an update to a network using the Williams equation (1995)

        args:

            - Input_: input vector

            - output_sigm: sigmoidal activation caused by input vector

            - output_bin: binary vector based on sigmoidal output + noise

            - l_rate: learnin rate

            - surprise: surprise caused by input and output-based reward

        return:

            - DeltaW: potential update of weights

            - DeltaW_bias: potential update of bias

        '''

    Err = output_bin - output_sigm

    Gradient = np.dot(Err.T, Input_)

    DeltaW = l_rate * Gradient * surprise

    DeltaW_bias = l_rate * Err

    return DeltaW.T, DeltaW_bias.T

def Potential_update_reinforced_CD(potential_update_weights, potential_update_bias1, potential_update_bias2, surprise):
    '''

                        This function proposes a weights update biased from the surprise (reinforcement learning version)
                        to each weights matrix.

                        args:

                            potential_update_weights: update of net weights (CD)
                            potential_update_bias1: update of input_to_hidden bias (bias 1) (CD)
                            potential_update_bias2: update of hidden_to_input bias (bias 2) (CD)

                            surprise: surprise producted by RL algorithm

                        return:


                            potential_update_weights: update for net weights
                            potential_update_bias1: update for input_to_hidden bias (bias 1) weights
                            potential_update_bias2: update for hidden_to_input bias (bias 2) weights


                    '''

    #surprise = max(0, surprise)

    potential_update_weights = potential_update_weights * surprise

    potential_update_bias1 = potential_update_bias1 * surprise

    potential_update_bias2 = potential_update_bias2 * surprise

    return potential_update_weights, potential_update_bias1, potential_update_bias2

def Potential_update_hibridation_CD_Reinforcement(Weight_inputs_update_CD, Weights_bias_inputs_to_hiddens_update_CD, Weights_bias_hiddens_to_inputs_update_CD,
                                                  Weight_inputs_update_RL, Weights_bias_inputs_to_hiddens_update_RL, CD_weight):

    '''
                This function computes an hibridated update for the three network matrices (net weights, bias_1 and bias_2),
                composed by the updates proposed by an RL algrythm (Williams, 1995) and a UL algorythm (Contrastive Divergence; Hinton, 2006).
                Tee hibridated update corresponds to a weight sum of both potential updates following the CD_weight parameter.


                    args:

                            Weight_inputs_update_CD: update of net weights (CD)
                            Weights_bias_inputs_to_hiddens_update_CD: update of input_to_hidden bias (bias 1) (CD)
                            Weights_bias_hiddens_to_inputs_update_CD: update of hidden_to_input bias (bias 2) (CD)
                            Weight_inputs_update_RL: update of net weights (RL)
                            Weights_bias_inputs_to_hiddens_update_RL: update of input_to_hidden bias (bias 1) (RL)
                            CD_weight: Parameter that balances the update of CD and RL


                    return:
                            Weight_inputs_update_hibridated: update for net weights
                            Weights_bias_inputs_to_hiddens_update_hibridated: update for input_to_hidden bias (bias 1) weights
                            Weights_bias_hiddens_to_inputs_update_hibridated: update for hidden_to_input bias (bias 2) weights

    '''
    Weight_inputs_update_hibridated = (CD_weight * Weight_inputs_update_CD) + ((1 - CD_weight) * Weight_inputs_update_RL)

    Weights_bias_inputs_to_hiddens_update_hibridated = (CD_weight * np.reshape (Weights_bias_inputs_to_hiddens_update_CD, list (np.shape (Weights_bias_inputs_to_hiddens_update_CD)) + [1])) + ((1 - CD_weight) * Weights_bias_inputs_to_hiddens_update_RL)

    #Weights_bias_inputs_to_hiddens_update_hibridated = (CD_weight * Weights_bias_inputs_to_hiddens_update_CD) + ((1 - CD_weight) * Weights_bias_inputs_to_hiddens_update_RL)

    Weights_bias_hiddens_to_inputs_update_hibridated = CD_weight * Weights_bias_hiddens_to_inputs_update_CD

    return Weight_inputs_update_hibridated, Weights_bias_inputs_to_hiddens_update_hibridated, Weights_bias_hiddens_to_inputs_update_hibridated


def initialization_training_params(range_params):

    Parameters_set = []

    for param in enumerate(range_params):

        Range_Single_param = range_params[param]

        if isinstance(Range_Single_param[0], (bool)):

            Single_param = np.random.choice([True, False])

        elif isinstance(Range_Single_param[0], (float, int)):

            if len(Range_Single_param) == 3:

                Potential_values = np.arange(Range_Single_param[0], Range_Single_param[1], Range_Single_param[2])

                Single_param = np.random.choice(Potential_values)

            else:

                Single_param = np.random.choice(Range_Single_param)



        Parameters_set.append(Single_param)


    return Parameters_set

def Ideal_actions_initialization(size_hidden, salient_feature, number_actions = 4):


        '''

        This funtion creates the ideal actions (labels) for reinforcement learning modiifcation of CD (contrastive divergence).
        The ideal actions have to be balanced (same number of 0 and 1) and different from each other at least for half of units.
        In case of length equal to 4 it is automatically assign a localistic form (onehotvector) to ideals actions.

        args:

            size_hidden: length of last hidden layer (output) of RBM/DBN
            salient_feature: feature that guides the creation of ideal actions (specific actions for colors, form or size)
            number_actions: number of attributes for each feature (e.g. for color are red, green, blue and yellow).
                            It is 4 for default state.

        return:

            ideal_actions_batch: matrix of inputs (64 rows) x number of hidden units (columns).


        '''

        print('I m creating the ideal actions')

        ideal_actions_batch = []

        if size_hidden == 4:

            ideal_actions_batch = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        else:

            negative_attempts = 0

            while len(ideal_actions_batch) < number_actions:

                new_ideal =  np.random.randint(2, size=(1, size_hidden))

                exception = False

                N_1 = np.sum(new_ideal)

                if N_1 != (size_hidden // 2):

                    exception = True
                    motive = ' unbalance of 0 and 1'

                if len(ideal_actions_batch) != 0 and not exception:

                    for comb in ideal_actions_batch:

                            diff = np.sum(abs(comb - new_ideal))

                            if diff < (size_hidden//2):# or (np.sum(new_ideal) > (size_hidden // 2)):

                                exception = True

                                motive = ' too similar to other combinations'

                if not exception:

                    ideal_actions_batch.append(new_ideal)

                    print(str(len(ideal_actions_batch)) + str('° combination added (n_comb = ') + str(len(ideal_actions_batch)) + str('/4'))

                    negative_attempts = 0

                else:

                    negative_attempts += 1

                    print(str(new_ideal) + str(' is an invalid combination (motive =') + str(motive) + str('// n_comb already added = ')
                          + str(len(ideal_actions_batch)) + str('/4') + str('// negative attemps = ') + str(negative_attempts))

                if negative_attempts > 50:

                    ideal_actions_batch = []

        ideal_actions_batch = np.vstack(ideal_actions_batch)

        print('...Ideal combinations created = ', ideal_actions_batch)


        if salient_feature == 'color' or salient_feature == 'none': # COLOR IDEAL FORMAT

            ideal_actions = np.tile(ideal_actions_batch, (16, 1))

            Legend_lab = ['Color: green', 'Color: red', 'Color: blue', 'Color: yellow']

            if ideal_actions_batch.shape[0] > 4:

                Legend_lab = np.tile(Legend_lab, 16)


        elif salient_feature == 'form':  # FORM IDEAL FORMAT

            ideal_actions = np.repeat(ideal_actions_batch, 4, axis = 0)
            ideal_actions = np.tile(ideal_actions, (4,1))

            Legend_lab = ['Shape: square', 'Shape: circle', 'Shape: bar', 'Shape: triangle']

            if ideal_actions_batch.shape[0] > 4:

                Legend_lab = np.repeat(Legend_lab, 4)
                Legend_lab = np.tile(Legend_lab, 4)

        elif salient_feature == 'size': # SIZE IDEAL FORMAT

            ideal_actions = np.repeat(ideal_actions_batch, 16, axis = 0)

            Legend_lab = ['Size: great', 'Size: great-medium', 'Size: medium-small', 'Size: small']


            if ideal_actions_batch.shape[0] > 4:

                Legend_lab = np.repeat(Legend_lab, 16, axis=0)
                Legend_lab = np.tile(Legend_lab, 4)



        return ideal_actions, ideal_actions_batch, Legend_lab

def Critic_init(size_input):

        '''

        This function initializes the weights of critic component of reinforcement learning.

        args:

            size_input: size of input layer for critic

        return:

            weights_critic: matrix of weight for the critic comonent (units x 1)

        '''

        weights_critic = np.random.uniform(- 0.1, 0.1, (size_input, 1))

        return weights_critic

def Critic_init_MLP(size_input, size_hidden = 50):

    '''

            This function initializes the weights of critic component of reinforcement learning (Multi layer perceptron)
            args:

                size_input: size of input layer for critic
                size_hidden: size of hidden layer for critic

            return:

                weights_critic: matrix of weight for the critic comonent (units x 1)

            '''


    weights_critic_input_to_hidden = np.random.uniform(- 0.001, 0.001, (size_input, size_hidden))

    weights_critic_hidden_to_output = np.random.uniform(- 0.001, 0.001, (size_hidden, 1))


    return [weights_critic_input_to_hidden, weights_critic_hidden_to_output]

def Critic_spread(input_, weights_critic):

    '''

    This function computes the critic spread.

    args:

        input_: original input of net
        weights_critic: weights matrix of critic

    return:

        Pred: predicted value of critic (expected reward)

    '''

    Pred = np.dot(input_, weights_critic)

    if len(Pred) == 1:

        return Pred[0][0]

    else:
        return np.array(Pred)

def Critic_spread_MLP(input_, weights_critic):

    '''

    This function computes the critic spread (MLP)

    args:

        input_: original input of net
        weights_critic: weights matrices of critic

    return:

        Pred: predicted value of critic (expected reward)

    '''

    Hidden_activation = np.dot(input_.T, weights_critic[0])

    Hidden_activation = sigmoid(Hidden_activation)



    Pred = np.dot(Hidden_activation, weights_critic[1])

    if len(Pred) == 1:

        #Pred = Pred.reshape((1,1))

        return Pred[0][0], Hidden_activation

    else:
        return np.array(Pred), Hidden_activation

def Critic_update(Input_, weights_critic, Pred, Real, learning_rate_critic = 0.001):

        '''

        This function computes the critic Update.

        args:

            input_: original input of net
            weights_critic: weights matrix of critic
            Pred: predicted value of critic (expected reward)
            Real: reward produced by network
            learning_rate_critic: learning rate of critic

        return:

            weights_critic: weights matrix of critic


        '''

        DeltaW = np.mean(learning_rate_critic * (Real - np.mean(Pred)) * Input_.T, axis = 1)
        DeltaW =  np.reshape(DeltaW, (DeltaW.shape[0], 1))
        weights_critic += DeltaW

        return weights_critic

def Critic_update_MLP(Input_, Hidden_, weights_critic, Pred, Real, learning_rate_critic=0.001):
    '''

    This function computes the critic update (MLP)

    args:

        input_: original input of net
        Hidden_: activation of hidden layer of critic
        weights_critic: weights matrices of critic
        Pred: predicted value of critic (expected reward)
        Real: reward produced by network
        learning_rate_critic: learning rate of critic

    return:

        weights_critic: weights matrix of critic


    '''


    Err_output = Real - Pred

    Err_hidden = np.dot(Err_output, weights_critic[1].T)

    DeltaW_input_to_hidden = learning_rate_critic * np.dot(Err_hidden.T, Input_)

    DeltaW_hidden_to_output = learning_rate_critic * np.dot(Err_output.T, Hidden_)

    # if not isinstance(Err_output, (float, int)):
    #
    #     DeltaW_input_to_hidden = np.mean(DeltaW_input_to_hidden, axis = 1)
    #
    #     DeltaW_hidden_to_output = np.mean(DeltaW_hidden_to_output, axis = 1)
    #
    #
    #     DeltaW_input_to_hidden = np.reshape(DeltaW_input_to_hidden, (DeltaW_input_to_hidden.shape[0], 1))
    #
    #     DeltaW_hidden_to_output = np.reshape(DeltaW_hidden_to_output, (DeltaW_hidden_to_output.shape[0], 1))


    weights_critic[0] += DeltaW_input_to_hidden.T

    weights_critic[1] += DeltaW_hidden_to_output.T


    return weights_critic

def Executor_init(size_input, size_output):
    '''

        This function initializes the weights of a simple network (perceptron)

    args:

        size_input: size of input layer for critic

    return:

        weights_critic: matrix of weight for the critic comonent (units x 1)

    '''

    weights_critic = np.random.uniform(- 0.01, 0.01, (size_input, size_output))

    bias = np.random.uniform(- 0.01, 0.01, (1, size_output))

    return [weights_critic, bias]

def Executor_init_MLP(size_input, size_output, size_hidden = 50):
    '''

        This function initializes the weights of a simple network (multi-layer perceptron) that represent
        an "executor controller" for a robotic architecture (DBN + MLP)

    args:

        size_input: size of input layer for controller
        size_hidden: size of hidden layer for controller


    return:

        weights_critic: matrix of weight for the executor component

    '''

    weights_executor_input_to_hidden = np.random.uniform(- 0.01, 0.01, (size_input, size_hidden))

    weights_executor_hidden_to_output = np.random.uniform(- 0.01, 0.01, (size_hidden, size_output))

    bias_hidden = np.random.uniform(- 0.01, 0.01, (1, size_hidden))

    bias_output = np.random.uniform(- 0.01, 0.01, (1, size_output))

    return [weights_executor_input_to_hidden, weights_executor_hidden_to_output, bias_hidden, bias_output]

def Executor_spread(input_, weights_executor):

    '''
        This function execute a stochastic spread of a simple network.

        args:

            - input_: input of NET (hidden representation of RBM)
            - weights_critic: weights list of NET (input -> output, bias)

        return:

            - output_sigm: sigmoidal output of vector

            - output_bin: binary outoput vector of net (0/1)

    '''

    Output_pot = np.dot(input_.T, weights_executor[0]) + weights_executor[1]

    Output_sigm = 1 / (1 + np.exp(-(Output_pot)))

    Output_bin = copy.deepcopy(Output_sigm)

    random_treshold = np.random.random_sample((Output_bin.shape[0], Output_bin.shape[1]))  # noise, case


    Output_bin[Output_bin > random_treshold] = 1

    Output_bin[Output_bin < random_treshold] = 0

    return Output_sigm, Output_bin

def Executor_spread_MLP(input_, weights_executor):

    '''
        This function execute a stochastic spread of a simple network (multi layer perceptron).

        args:

            - input_: input of NET (hidden representation of RBM)
            - weights_critic: weights list of NET (input -> output, bias)

        return:

            - output_sigm: sigmoidal output of vector

            - output_bin: binary outoput vector of net (0/1)

    '''

    Hidden_activation = np.dot(input_.T, weights_executor[0])

    Hidden_activation = sigmoid(Hidden_activation + weights_executor[2])

    Output = np.dot(Hidden_activation, weights_executor[1])

    Output_sigm = sigmoid(Output + weights_executor[3])

    Output_bin = copy.deepcopy(Output_sigm)

    random_treshold = np.random.random_sample((Output_bin.shape[0], Output_bin.shape[1]))  # noise, case


    Output_bin[Output_bin > random_treshold] = 1

    Output_bin[Output_bin < random_treshold] = 0

    return Output_sigm, Output_bin, Hidden_activation

def Executor_Potential_update(Input_, output_sigm, output_bin, l_rate, surprise):

    '''

        This function propose an update to a simple network (perceptron) using the Williams equation (1995)
    args:

        - Input_: input vector

        - output_sigm: sigmoidal activation caused by input vector

        - output_bin: binary vector based on sigmoidal output + noise

        - l_rate: learnin rate

        - surprise: surprise caused by input and output-based reward

    return:

        - DeltaW: potential update of weights

        - DeltaW_bias: potential update of bias

    '''

    Err = output_bin - output_sigm
    Gradient = np.dot(Err.T, Input_.T) # remember to set Err to Err.T

    DeltaW = l_rate * Gradient * surprise

    DeltaW_bias = l_rate * Err

    return Gradient, DeltaW, DeltaW_bias

def Executor_Potential_update_MLP(Input_, Hidden_, output_sigm, output_bin, weights_executor, l_rate, surprise):

    '''

        This function propose an update to a simple network (multi-layer perceptron) using the Williams equation (1995)
    args:

        - Input_: input vector

        - output_sigm: sigmoidal activation caused by input vector

        - output_bin: binary vector based on sigmoidal output + noise

        - l_rate: learnin rate

        - surprise: surprise caused by input and output-based reward

    return:

        - DeltaW: potential update of weights

        - DeltaW_bias: potential update of bias

    '''

    Err_output = output_bin - output_sigm


    Err_hidden = np.dot(Err_output, weights_executor[1].T)

    DeltaW_hidden_to_output = l_rate * np.dot(Err_output.T, Hidden_) * surprise

    DeltaW_input_to_hidden = l_rate * np.dot(Err_hidden.T, Input_) * surprise


    DeltaW_bias_hidden = l_rate * Err_hidden

    DeltaW_bias_output = l_rate * Err_output

    return [DeltaW_input_to_hidden, DeltaW_hidden_to_output], [DeltaW_bias_hidden, DeltaW_bias_output]

def Executor_Effective_update(Weight_net, Weights_bias, potential_update_weights_net,
                     potential_update_weights_bias):

    '''
        This function applied a weights update to each weights matrix of a perceptron.

    args:

        - Weight_net: weights matrix of network
        - Weights_bias: weight matrix of network bias
        - potential_update_weights_net: update for network matrix
        - potential_update_weights_bias: update for bias matrix

    return:

        - Weight_net: updated  weights matrix of network
        - Weights_bias: updated weight matrix of network bias

    '''

    Weight_net += potential_update_weights_net.T

    Weights_bias += potential_update_weights_bias

    return [Weight_net, Weights_bias]

def Executor_Effective_update_MLP(Weight_net, potential_update_weights_net, potential_update_weights_bias):

    '''
        This function applied a weights update to each weights matrix of a ML perceptron

    args:

        - Weight_net: weights matrix of network
        - Weights_bias: weight matrix of network bias
        - potential_update_weights_net: update for network matrix
        - potential_update_weights_bias: update for bias matrix

    return:

        - Weight_net: updated  weights matrix of network
        - Weights_bias: updated weight matrix of network bias

    '''

    Weight_net[0] += potential_update_weights_net[0].T

    Weight_net[1] += potential_update_weights_net[1].T


    Weight_net[2] += potential_update_weights_bias[0]

    Weight_net[3] += potential_update_weights_bias[1]


    return Weight_net

def Reinforce_processing(executed, ideal, Pred, size_hidden, binary_reward_format = True):

        '''

        This function computes the reward of net activation depending on Williams algorithm (see REINFORCE, 1995).

        args:

            executed: hidden activation of net
            ideal: ideal action (label) depending on salient feature (e.g. red, green,blue, yellow for colors)
            Pred: output of critic
            size_hidden: length of hidden layer (number of units)

        return:

            executed: hidden activation of net
            ideal: ideal action (label) depending on salient feature (e.g. red, green,blue, yellow for colors)

            reinforce: reward produced by net
            Reinforce_for_input: ... for each input (batch)

            surprise: difference between reward and critic output (Pred)
            Surprise_for_input: ... for each input (batch)

            Accuracy: accuracy of this specific activation (1/0)
            Accuracy_for_input: ... for each input (batch)

            R_range: range of reinforcements values
            R_range_interp: interpolated range of reinforcements values

        '''

        R_range = [0, size_hidden]
        R_range_interp =  [0, 1]

        # ACCURACIES COMPUTATIONS

        accuracies_list = []

        for element in range(0, executed.shape[0]):

            first_term = executed[element, :]

            second_term = ideal[element, :]

            if np.array_equal(first_term, second_term):

                accuracies_list.append(1)

            else:

                accuracies_list.append(0)

        Reinforce_for_input = np.sum(abs(ideal - executed), axis = 1)
        Reinforce_for_input = 1 - np.interp(Reinforce_for_input, (R_range[0], R_range[1]), (R_range_interp[0], R_range_interp[1]))

        if binary_reward_format:

            Reinforce_for_input[Reinforce_for_input != 1] = 0 # HARD CONDITION (1/0)

        Reinforce_for_input = np.reshape(Reinforce_for_input,(Reinforce_for_input.shape[0], 1))
        reinforce = np.mean(Reinforce_for_input)


        if not isinstance(Pred, (int, float)):

            Surprise_for_input = (Reinforce_for_input - (Pred.T)[:,0]).T

        else:

            Surprise_for_input = (Reinforce_for_input - Pred)


        surprise = reinforce - np.mean(Pred)

        Accuracy_for_input = np.array(accuracies_list)
        Accuracy = np.mean(Accuracy_for_input)



        return executed, ideal, reinforce, Reinforce_for_input, surprise, Surprise_for_input, Accuracy, Accuracy_for_input, R_range, R_range_interp


def save_training_data_reinforced_CD_CHUNKED(Number_Single_Chunk_File, Reinforces, Surprises, Accuracies, Reconstruction_errors, Weight_inputs, Weight_inputs_sum,
                       Weights_bias_inputs_to_hiddens_sum, Weights_bias_hiddens_to_inputs_sum, learning_rate, learning_rate_critic,
                       R_range, R_range_interp, ideal_actions, Reconstruction_errors_other = 0, Weight_inputs_other = 0,
                       Weight_inputs_sum_other = 0, Weights_bias_inputs_to_hiddens_sum_other = 0,
                       Weights_bias_hiddens_to_inputs_sum_other = 0, learning_rate_other = 0, save_path = ''):

    '''

        This function saves the training data of NET in case of reinforcement learning modification

        args:

            Reinforces: list of rewards for each input (64) for each epoc
            Surprises: list of surprises for each input (64) for each epoc
            Accuracies: list of accuracies for each input (64) for each epoc
            Reconstruction_errors: list of rec_errors for each input (64) for each epoc
            Weight_inputs =  network matrix
            Weight_inputs_sum: list of sums of network matrix (I used the sum to have a gross measure of weights)
            Weights_bias_inputs_to_hiddens_sum: list of sums of bias from input to hidden weights matrix (I used the sum to have a gross measure of weights)
            Weights_bias_hiddens_to_inputs_sum: lists of sums of bias from hidden to input weights network matrix (I used the sum to have a gross measure of weights)

            learning_rate: learning rate of CD update
            learning_rate_critic: learning rate of critic update
            R_range: range of reward before interpolation
            R_range_interp: range of reward after interpolation
            ideal_actions: ideal actions that drive the reinforcement learning (labels)

            Reconstruction_errors_other: Reconstruction_errors variable for second RBM
            Weight_inputs_other: Weight_inputs variable for second RBM
            Weights_bias_inputs_to_hiddens_other: Weights_bias_inputs_to_hiddens variable for second RBM
            Weights_bias_hiddens_to_inputs_other: Weights_bias_hiddens_to_inputs variable for second RBM
            learning_rate_other:  learning rate of CD update for second RBM

            save_path: folder path of files

        return:

                None (Training file into a specific folder)

        '''

    if save_path != '':

        path = save_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

    if not isinstance(Weight_inputs_other, (list, np.ndarray)):

        file_name = 'Training_Data_Single_RBM' + str('_') + str(Weight_inputs.shape[0]) + str('_') + str(Weight_inputs.shape[1]) + str('_Reinforced_') + str(Number_Single_Chunk_File)

        params = np.array([learning_rate, learning_rate_critic])

        R_ranges = np.vstack((R_range, R_range_interp))

        np.savez(path + file_name, Inputs_R = Reinforces, Inputs_S = Surprises, Inputs_Acc = Accuracies, Inputs_Rec = Reconstruction_errors,
                 Net_W_NOT_SUM = Weight_inputs, Net_W = Weight_inputs_sum, W_b1 = Weights_bias_inputs_to_hiddens_sum,
                 W_b2 = Weights_bias_hiddens_to_inputs_sum, PARAMS = params, IDEAL = ideal_actions)

    else:

        file_name = 'Training_Data_Whole_DBN' + str('_') + str(Weight_inputs.shape[0]) + str('_') + str(
            Weight_inputs.shape[1]) + str('_') + str(Weight_inputs_other.shape[1]) + str('_Reinforced_') + str(Number_Single_Chunk_File)

        params = np.array([learning_rate, learning_rate_other, learning_rate_critic])

        R_ranges = np.vstack((R_range, R_range_interp))


        np.savez(path + file_name, Inputs_R = Reinforces, Inputs_S = Surprises, Inputs_Acc = Accuracies, Inputs_Rec = Reconstruction_errors,
                 Net_W_NOT_SUM = Weight_inputs, Net_W = Weight_inputs_sum, W_b1 = Weights_bias_inputs_to_hiddens_sum,
                 W_b2 = Weights_bias_hiddens_to_inputs_sum, Inputs_Rec_other = Reconstruction_errors_other,
                 Net_W_NOT_SUM_other = Weight_inputs_other, Net_W_other = Weight_inputs_sum_other, W_b1_other = Weights_bias_inputs_to_hiddens_sum_other,
                 W_b2_other = Weights_bias_hiddens_to_inputs_sum_other,
                 PARAMS = params, R_RANGES = R_ranges, IDEAL = ideal_actions)

def load_training_data_reinforced_CD_JOIN(structure, load_path = ''):

    '''

        This function loads the training data of NET in case of reinforcement learning modification

        args:

            structure: vector with the topology of RBM/DBN (e.g. [2352, 150, 10])

            load_path: folder path of files



        return:

            in case of RBM:

            Reinforces_for_each_epoc: list of avg rewards (batch) for each epoc
            Reinforces_for_each_epoc_STD: standard deviation of previous variable
            Surprises_for_each_epoc: list of avg surprises (batch)  for each epoc
            Surprises_for_each_epoc_STD: standard deviation of previous variable
            Accuracies_for_each_epoc: list of avg accuracies (batch)  for each epoc
            Accuracies_for_each_epoc_STD: standard deviation of previous variable
            Reconstruction_errors_for_each_epoc: list of avg rec_errors (batch) for each epoc

            Weight_inputs = network matrix
            Sum_weights_network_for_each_epoc: sum of network matrix (I used the sum to have a gross measure of weights) for each epoc
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc: sum of bias from input to hidden weights matrix for each epoc
            Sum_bias_hiddens_to_inputs_weights_for_each_epoc: sum of bias from hidden to input weights matrix for each epoc

            Reinforces_for_each_input_for_each_epoc: list of rewards for each input (64) for each epoc
            Surprises_for_each_input_for_each_epoc: list of surprises for each input (64) for each epoc
            Accuracies_for_each_input_for_each_epoc: list of accuracies for each input (64) for each epoc


            learning_rate: learning rate of CD update
            learning_rate_critic: learning rate of critic update
            R_range: range of reward before interpolation
            R_range_interp: range of reward after interpolation
            ideal_actions: ideal actions that drive the reinforcement learning (labels)

            in case of DBN there are additional variables:

            learning_rate_Other: learning_rate for second RBM
            Reconstruction_errors_for_each_epoc_Other: ... for second RBM
            Weight_inputs_other = ... for second RBM
            Sum_weights_network_for_each_epoc_Other: ... for second RBM
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other: ... for second RBM
            Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other: ... for second RBM


        '''

    if load_path != '':

        path = load_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

    print('Training data processing: START.. \n')

    print('...\n')

    if len(structure) == 2:

        file_name = 'Training_Data_Single_RBM' + str('_') + str(structure[0]) + str('_') + str(structure[1]) + str('_Reinforced_*')

        Chunks_list = glob.glob(path + file_name)

        Chunks_list = natural_sort(Chunks_list)

        for (count, file) in enumerate(Chunks_list):

            #print('...')

            if count == 0:

                Training_data = np.load(file)

                Learning_rate = Training_data['PARAMS'][0]

                Learning_rate_critic = Training_data['PARAMS'][1]

                R_range = [0, structure[1]]

                R_range_interp = [0, 1]

                ideal_actions = Training_data['IDEAL']

                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Acc'].T

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Rec'].T


                Sum_weights_network_for_each_input_for_each_epoc_JOIN  = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN  = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN  = Training_data['W_b2'].T

                # CALCULATED DATA

                Reinforces_for_each_epoc_JOIN = np.mean(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Reinforces_for_each_epoc_STD_JOIN = np.std(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_JOIN = np.mean(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_STD_JOIN = np.std(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_JOIN = np.mean(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_STD_JOIN = np.std(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Reconstruction_errors_for_each_epoc_JOIN = np.mean(Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                              axis=0)

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.std(Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                                 axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.mean(Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                axis=0)

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.std(Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                   axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                else:

                    Sum_weights_network_for_each_epoc_JOIN = copy.deepcopy(Sum_weights_network_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN)

            else:

                Training_data = np.load(file)


                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc = Training_data['Inputs_Acc'].T

                Reconstruction_errors_for_each_input_for_each_epoc = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc = Training_data['W_b2'].T

                # CALCULATED DATA

                Reinforces_for_each_epoc = np.mean(Reinforces_for_each_input_for_each_epoc, axis=0)

                Reinforces_for_each_epoc_STD = np.std(Reinforces_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc = np.mean(Surprises_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc_STD = np.std(Surprises_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc = np.mean(Accuracies_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc_STD = np.std(Accuracies_for_each_input_for_each_epoc, axis=0)

                Reconstruction_errors_for_each_epoc = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc,
                    axis=0)

                Reconstruction_errors_for_each_epoc_STD = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc,
                    axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc.ndim == 2:

                    Sum_weights_network_for_each_epoc = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_weights_network_for_each_epoc_STD = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc, axis=0)

                else:

                    Sum_weights_network_for_each_epoc = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc)

                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_input_for_each_epoc_JOIN,
                                                                          Reinforces_for_each_input_for_each_epoc))

                Surprises_for_each_input_for_each_epoc_JOIN = np.hstack((Surprises_for_each_input_for_each_epoc_JOIN,
                                                                         Surprises_for_each_input_for_each_epoc))

                Accuracies_for_each_input_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_input_for_each_epoc_JOIN,
                                                                          Accuracies_for_each_input_for_each_epoc))

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                                                     Reconstruction_errors_for_each_input_for_each_epoc))

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                                   Sum_weights_network_for_each_input_for_each_epoc))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN,
                                                                                                  Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN,
                                                                                                  Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc))

                # CALCULATED DATA

                Reinforces_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_epoc_JOIN,
                                                          Reinforces_for_each_epoc))


                Reinforces_for_each_epoc_STD_JOIN = np.hstack((Reinforces_for_each_epoc_STD_JOIN,
                                                          Reinforces_for_each_epoc_STD))

                Surprises_for_each_epoc_JOIN = np.hstack((Surprises_for_each_epoc_JOIN,
                                                          Surprises_for_each_epoc))

                Surprises_for_each_epoc_STD_JOIN = np.hstack((Surprises_for_each_epoc_STD_JOIN,
                                                          Surprises_for_each_epoc_STD))

                Accuracies_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_epoc_JOIN,
                                                          Accuracies_for_each_epoc))

                Accuracies_for_each_epoc_STD_JOIN = np.hstack((Accuracies_for_each_epoc_STD_JOIN,
                                                          Accuracies_for_each_epoc_STD))

                Reconstruction_errors_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_JOIN,
                                                          Reconstruction_errors_for_each_epoc))

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_STD_JOIN,
                                                          Reconstruction_errors_for_each_epoc_STD))

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                          Sum_weights_network_for_each_epoc))

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_STD_JOIN,
                                                          Sum_weights_network_for_each_epoc_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD))

                else:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                          Sum_weights_network_for_each_epoc))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

        Weights_network = Training_data['Net_W_NOT_SUM']

        print('Training data processing: STOP \n')

        return Learning_rate, Learning_rate_critic, R_range, R_range_interp, ideal_actions, Reinforces_for_each_epoc_JOIN, \
               Reinforces_for_each_epoc_STD_JOIN, Surprises_for_each_epoc_JOIN, Surprises_for_each_epoc_STD_JOIN, Accuracies_for_each_epoc_JOIN, \
               Accuracies_for_each_epoc_STD_JOIN,\
               Reconstruction_errors_for_each_epoc_JOIN, Weights_network, Sum_weights_network_for_each_epoc_JOIN, \
               Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN, Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN, \
               Reinforces_for_each_input_for_each_epoc_JOIN[:, -1], Surprises_for_each_input_for_each_epoc_JOIN[:, -1], Accuracies_for_each_input_for_each_epoc_JOIN[:, -1]






    else:

        file_name = '*Training_Data_Whole_DBN'  + str('_') + str(structure[0]) + str('_') + str(structure[1]) + str('_') + str(structure[2]) + str('_Reinforced_*')

        Chunks_list = glob.glob(path + file_name)

        Chunks_list = natural_sort(Chunks_list)

        for (count, file) in enumerate(Chunks_list):

            #print('...')

            if count == 0:

                Training_data = np.load(file)

                # LEARNING PARAMS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Learning_rate = Training_data['PARAMS'][0]

                Learning_rate_Other = Training_data['PARAMS'][1]

                Learning_rate_critic = Training_data['PARAMS'][2]

                R_range = Training_data['R_RANGES'][0]

                R_range_interp = Training_data['R_RANGES'][1]

                ideal_actions = Training_data['IDEAL']


                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Acc'].T

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = Training_data['W_b2'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc_JOIN = np.mean(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Reinforces_for_each_epoc_STD_JOIN = np.std(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_JOIN = np.mean(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_STD_JOIN = np.std(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_JOIN = np.mean(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_STD_JOIN = np.std(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc_JOIN = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                    axis=0)

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                    axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                        axis=0)

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                else:

                    Sum_weights_network_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN = Training_data['Inputs_Rec_other'].T

                Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN = Training_data['Net_W_other'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN = Training_data[
                    'W_b1_other'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN = Training_data[
                    'W_b2_other'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other_JOIN = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                Reconstruction_errors_for_each_epoc_Other_STD_JOIN = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                    Sum_weights_network_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)


                else:

                    Sum_weights_network_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN)


            else:

                Training_data = np.load(file)

                Weights_network = Training_data['Net_W_NOT_SUM']

                Weights_network_other = Training_data['Net_W_NOT_SUM_other'].T

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc = Training_data['Inputs_Acc'].T

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc = Training_data['W_b2'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc = np.mean(Reinforces_for_each_input_for_each_epoc, axis=0)

                Reinforces_for_each_epoc_STD = np.std(Reinforces_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc = np.mean(Surprises_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc_STD = np.std(Surprises_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc = np.mean(Accuracies_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc_STD = np.std(Accuracies_for_each_input_for_each_epoc, axis=0)

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc = np.mean(Reconstruction_errors_for_each_input_for_each_epoc,
                                                              axis=0)

                Reconstruction_errors_for_each_epoc_STD = np.std(Reconstruction_errors_for_each_input_for_each_epoc,
                                                                 axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc.ndim == 2:

                    Sum_weights_network_for_each_epoc = np.mean(Sum_weights_network_for_each_input_for_each_epoc,
                                                                axis=0)

                    Sum_weights_network_for_each_epoc_STD = np.std(Sum_weights_network_for_each_input_for_each_epoc,
                                                                   axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc,
                        axis=0)

                else:

                    Sum_weights_network_for_each_epoc = copy.deepcopy(Sum_weights_network_for_each_input_for_each_epoc)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other = Training_data['Inputs_Rec_other'].T

                Sum_weights_network_for_each_input_for_each_epoc_Other = Training_data['Net_W_other'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other = Training_data['W_b1_other'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other = Training_data['W_b2_other'].T

                # ----- SECOND RBM VARIABLES -----

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other, axis=0)

                Reconstruction_errors_for_each_epoc_Other_STD = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other, axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_Other.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_Other, axis=0)

                    Sum_weights_network_for_each_epoc_Other_STD = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_Other, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                else:

                    Sum_weights_network_for_each_epoc_Other = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_Other)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_input_for_each_epoc_JOIN,
                                                                      Reinforces_for_each_input_for_each_epoc))

                Surprises_for_each_input_for_each_epoc_JOIN = np.hstack((Surprises_for_each_input_for_each_epoc_JOIN,
                                                                         Surprises_for_each_input_for_each_epoc))

                Accuracies_for_each_input_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_input_for_each_epoc_JOIN,
                                                                          Accuracies_for_each_input_for_each_epoc))

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                     Reconstruction_errors_for_each_input_for_each_epoc))

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                     Sum_weights_network_for_each_input_for_each_epoc))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN,
                     Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN,
                     Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc))

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_epoc_JOIN,
                                                           Reinforces_for_each_epoc))

                Reinforces_for_each_epoc_STD_JOIN = np.hstack((Reinforces_for_each_epoc_STD_JOIN,
                                                               Reinforces_for_each_epoc_STD))

                Surprises_for_each_epoc_JOIN = np.hstack((Surprises_for_each_epoc_JOIN,
                                                          Surprises_for_each_epoc))

                Surprises_for_each_epoc_STD_JOIN = np.hstack((Surprises_for_each_epoc_STD_JOIN,
                                                              Surprises_for_each_epoc_STD))

                Accuracies_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_epoc_JOIN,
                                                           Accuracies_for_each_epoc))

                Accuracies_for_each_epoc_STD_JOIN = np.hstack((Accuracies_for_each_epoc_STD_JOIN,
                                                               Accuracies_for_each_epoc_STD))
                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_JOIN,
                                                                      Reconstruction_errors_for_each_epoc))

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_STD_JOIN,
                                                                          Reconstruction_errors_for_each_epoc_STD))

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                                        Sum_weights_network_for_each_epoc))

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_STD_JOIN,
                                                                            Sum_weights_network_for_each_epoc_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD))


                else:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                                        Sum_weights_network_for_each_epoc))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN,
                         Reconstruction_errors_for_each_input_for_each_epoc_Other))

                Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_weights_network_for_each_input_for_each_epoc_Other))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other))

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_Other_JOIN,
                                                                      Reconstruction_errors_for_each_epoc_Other))

                Reconstruction_errors_for_each_epoc_Other_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_Other_STD_JOIN,
                                                                          Reconstruction_errors_for_each_epoc_Other_STD))


                if Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other))

                    Sum_weights_network_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD))


                else:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other))

        print('Training data processing: STOP \n')


        return Learning_rate, Learning_rate_Other, Learning_rate_critic, R_range, R_range_interp, ideal_actions, \
               Reinforces_for_each_epoc_JOIN, Reinforces_for_each_epoc_STD_JOIN, Surprises_for_each_epoc_JOIN, Surprises_for_each_epoc_STD_JOIN, \
               Accuracies_for_each_epoc_JOIN, Accuracies_for_each_epoc_STD_JOIN, Reconstruction_errors_for_each_epoc_JOIN, Weights_network, \
               Sum_weights_network_for_each_epoc_JOIN, Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN, Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN, \
               Reconstruction_errors_for_each_epoc_Other_JOIN, Weights_network_other, Sum_weights_network_for_each_epoc_Other_JOIN, Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,\
               Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN, Reinforces_for_each_input_for_each_epoc_JOIN[:, -1], Surprises_for_each_input_for_each_epoc_JOIN[:, -1],\
               Accuracies_for_each_input_for_each_epoc_JOIN[:, -1]


#   SUPERVISED LEARNING MODIFICATION FUNCTIONS

def Ideal_actions_initialization_Supervised_CD(size_hidden, salient_feature = 0, number_actions = 4):

    '''

            This funtion creates the ideal actions (labels) for supervised modifcation of CD (contrastive divergence).
            In case of size_hidden = 12 it produces 64 semi-localistic representations ([[1, 0, 0, 0], [0, 1, 0, 0], ..])
            In case of a different number of size this function call the funciton "Ideal_actions_initialization".

            args:

                size_hidden: length of last hidden layer (output) of RBM/DBN
                salient_feature: feature that guides the creation of ideal actions (specific actions for colors, form or size)
                number_actions: number of attributes for each feature (e.g. for color are red, green, blue and yellow).
                                It is 4 for default state.

            return:

                ideal_actions: matrix of inputs (64 rows) x number of hidden units (columns).
                ideal_actions_batch: matrix of four foundamental ideal actions.


    '''


    if size_hidden == 12:

        ideal_actions_batch = np.zeros((64, size_hidden))

        # Greate figures (Square,circles,bars,triangles)(green,red,blu,yellow)
        # Square
        ideal_actions_batch[0, [0, 4, 8]] = 1
        ideal_actions_batch[1, [1, 4, 8]] = 1
        ideal_actions_batch[2, [2, 4, 8]] = 1
        ideal_actions_batch[3, [3, 4, 8]] = 1
        # Circles
        ideal_actions_batch[4, [0, 5, 8]] = 1
        ideal_actions_batch[5, [1, 5, 8]] = 1
        ideal_actions_batch[6, [2, 5, 8]] = 1
        ideal_actions_batch[7, [3, 5, 8]] = 1
        # Bars
        ideal_actions_batch[8, [0, 6, 8]] = 1
        ideal_actions_batch[9, [1, 6, 8]] = 1
        ideal_actions_batch[10, [2, 6, 8]] = 1
        ideal_actions_batch[11, [3, 6, 8]] = 1
        # Triangles
        ideal_actions_batch[12, [0, 7, 8]] = 1
        ideal_actions_batch[13, [1, 7, 8]] = 1
        ideal_actions_batch[14, [2, 7, 8]] = 1
        ideal_actions_batch[15, [3, 7, 8]] = 1
        # Medium-great figures (Square,circles,bars,triangles)(green,red,blu,yellow)
        # Square
        ideal_actions_batch[16, [0, 4, 9]] = 1
        ideal_actions_batch[17, [1, 4, 9]] = 1
        ideal_actions_batch[18, [2, 4, 9]] = 1
        ideal_actions_batch[19, [3, 4, 9]] = 1
        # Circles
        ideal_actions_batch[20, [0, 5, 9]] = 1
        ideal_actions_batch[21, [1, 5, 9]] = 1
        ideal_actions_batch[22, [2, 5, 9]] = 1
        ideal_actions_batch[23, [3, 5, 9]] = 1
        # Bars
        ideal_actions_batch[24, [0, 6, 9]] = 1
        ideal_actions_batch[25, [1, 6, 9]] = 1
        ideal_actions_batch[26, [2, 6, 9]] = 1
        ideal_actions_batch[27, [3, 6, 9]] = 1
        # Triangles
        ideal_actions_batch[28, [0, 7, 9]] = 1
        ideal_actions_batch[29, [1, 7, 9]] = 1
        ideal_actions_batch[30, [2, 7, 9]] = 1
        ideal_actions_batch[31, [3, 7, 9]] = 1
        # Medium-small figures (Square,circles,bars,triangles)(green,red,blu,yellow)
        # Square
        ideal_actions_batch[32, [0, 4, 10]] = 1
        ideal_actions_batch[33, [1, 4, 10]] = 1
        ideal_actions_batch[34, [2, 4, 10]] = 1
        ideal_actions_batch[35, [3, 4, 10]] = 1
        # Circles
        ideal_actions_batch[36, [0, 5, 10]] = 1
        ideal_actions_batch[37, [1, 5, 10]] = 1
        ideal_actions_batch[38, [2, 5, 10]] = 1
        ideal_actions_batch[39, [3, 5, 10]] = 1
        # Bars
        ideal_actions_batch[40, [0, 6, 10]] = 1
        ideal_actions_batch[41, [1, 6, 10]] = 1
        ideal_actions_batch[42, [2, 6, 10]] = 1
        ideal_actions_batch[43, [3, 6, 10]] = 1
        # Triangles
        ideal_actions_batch[44, [0, 7, 10]] = 1
        ideal_actions_batch[45, [1, 7, 10]] = 1
        ideal_actions_batch[46, [2, 7, 10]] = 1
        ideal_actions_batch[47, [3, 7, 10]] = 1
        # Small figures (Square,circles,bars,triangles)(green,red,blu,yellow)
        # Square
        ideal_actions_batch[48, [0, 4, 11]] = 1
        ideal_actions_batch[49, [1, 4, 11]] = 1
        ideal_actions_batch[50, [2, 4, 11]] = 1
        ideal_actions_batch[51, [3, 4, 11]] = 1
        # Circles
        ideal_actions_batch[52, [0, 5, 11]] = 1
        ideal_actions_batch[53, [1, 5, 11]] = 1
        ideal_actions_batch[54, [2, 5, 11]] = 1
        ideal_actions_batch[55, [3, 5, 11]] = 1
        # Bars
        ideal_actions_batch[56, [0, 6, 11]] = 1
        ideal_actions_batch[57, [1, 6, 11]] = 1
        ideal_actions_batch[58, [2, 6, 11]] = 1
        ideal_actions_batch[59, [3, 6, 11]] = 1
        # Triangles
        ideal_actions_batch[60, [0, 7, 11]] = 1
        ideal_actions_batch[61, [1, 7, 11]] = 1
        ideal_actions_batch[62, [2, 7, 11]] = 1
        ideal_actions_batch[63, [3, 7, 11]] = 1

        ideal_actions = copy.deepcopy(ideal_actions_batch)


    else:

        ideal_actions, ideal_actions_batch = Ideal_actions_initialization(size_hidden, salient_feature, number_actions)


    return ideal_actions, ideal_actions_batch

def Reconstruction_errors_supervised_CD(Original_input, Reconstructed_input, hidden_output, ideal):

    '''

        This function computes the recostruction errors and the accuracy of a supervised RBM.

        args:

            Original_input: visible original input
            Reconstructed_input: visible reconstructed input (inverse spread result)
            hidden_output: hidden activation depending on input
            ideal: ideal activation for specific inputs (labels)


        return:

            errors_absolute_value_list = reconstruction error for each input
            errors_absolute_value_batch = avg reconstruction error for batch
            St_dev_errors_absolute_value_batch = standard deviation of previous variable for batch
            accuracies_value_batch = avg accuracy for batch
            St_dev_accuracies_value_batch = standard deviation of previous variable for batch


    '''

    hidden_output = copy.deepcopy(hidden_output)

    # RECONSTRUCTION ERRORS

    errors_absolute_value_list = []

    errors_percent_value_list = []

    for inp in range(0, Original_input.shape[0]):

        error_vector = np.abs(Original_input[inp, :] - Reconstructed_input[inp, :])

        errors_absolute_value = error_vector.sum(axis=0) / error_vector.shape[0]

        errors_percent_value = errors_absolute_value * 100

        errors_absolute_value_list.append(errors_absolute_value)
        errors_percent_value_list.append(errors_percent_value)

    # IN CASE OF BATCH LEARNING THE FOLLOWING LINES CORRESPOND TO MEANS AND STDS OF PREVIOUS LINES (MATRICES OF INPUTS X VALUES),

    errors_absolute_value_batch = np.mean(errors_absolute_value_list)

    errors_percent_value_batch = np.mean(errors_percent_value_list)

    St_dev_errors_absolute_value_batch = np.std(errors_absolute_value_list)

    St_dev_errors_percent_value_value_batch = np.std(errors_percent_value_list)


    # ACCURACIES COMPUTATIONS

    accuracies_list = []

    for element in range(0, hidden_output.shape[0]):

        first_term = hidden_output[element, :]

        second_term = ideal[element, :]

        if hidden_output.shape[1] == 12:

            colors = first_term[0:4]
            highest_color = np.argmax(colors)
            colors[:] = 0
            colors[highest_color] = 1

            shapes = first_term[4:8]
            highest_shapes = np.argmax(shapes)
            shapes[:] = 0
            shapes[highest_shapes] = 1

            sizes = first_term[8:12]
            highest_sizes = np.argmax(sizes)
            sizes[:] = 0
            sizes[highest_sizes] = 1

        elif hidden_output.shape[1] == 4:

            attributes = first_term[0:4]
            highest_attribute = np.argmax(attributes)
            attributes[:] = 0
            attributes[highest_attribute] = 1

        else:

            first_term[first_term > 0.95] = 1
            first_term[first_term < 0.05] = 0



        if np.array_equal(first_term, second_term):

            accuracies_list.append(1)

        else:

            accuracies_list.append(0)

    accuracies_value_batch = np.mean(accuracies_list)

    St_dev_accuracies_value_batch = np.std(accuracies_list)



    return np.array(errors_absolute_value_list), errors_absolute_value_batch, St_dev_errors_absolute_value_batch, \
           accuracies_list, accuracies_value_batch, St_dev_accuracies_value_batch

def Supervised_modification_CD(hidden_output, ideals, specific_ideal = 0):

    '''

        This function substitute the stochastic random activation of RBM with a specific ideal activation.

        args:

        hidden_output: hidden activation
        ideals: whole dataste of ideals actions (64 inputs x hidden units)
        specific_ideal: in case of online learning, specific ideal action

        return:

            ideals: whole dataset of ideals actions (64 inputs x hidden units)
            specific_ideal: in case of online learning, specific ideal action

    '''
    if hidden_output.shape[0] > 1:

        return ideals

    else:

        return specific_ideal

def Top_Down_Manipulation(hidden_output, salient_feature):

    Manipulated_Activation = copy.deepcopy(hidden_output)

    if hidden_output.shape[0] == 1:

        if hidden_output.shape[1] == 12:

            # LATERAL INHIBITION OF ATTRIBUTES
            colors = Manipulated_Activation[:, 0:4]
            highest_color = np.argmax(colors)
            colors[:] = 0
            colors[:, highest_color] = 1

            shapes = Manipulated_Activation[:, 4:8]
            highest_shapes = np.argmax(shapes)
            shapes[:] = 0
            shapes[:, highest_shapes] = 1

            sizes = Manipulated_Activation[:, 8:12]
            highest_sizes = np.argmax(sizes)
            sizes[:] = 0
            sizes[:, highest_sizes] = 1

            # DISHINIBITION OF A SPECIFIC FEATURE

            if salient_feature == 'color':
                shapes[:] = 0
                sizes[:] = 0
            elif salient_feature == 'form':
                colors[:] = 0
                sizes[:] = 0
            elif salient_feature == 'size':
                colors[:] = 0
                shapes[:] = 0

        elif hidden_output.shape[1] == 4:

            # LATERAL INHIBITION OF ATTRIBUTES
            attributes = Manipulated_Activation[0:4]
            highest_attribute = np.argmax(attributes)
            attributes[:] = 0
            attributes[highest_attribute] = 1

        else:

            Manipulated_Activation[Manipulated_Activation > 0.5] = 1
            Manipulated_Activation[Manipulated_Activation < 0.5] = 0

    elif hidden_output.shape[0] == 64:

        if hidden_output.shape[1] == 12:

            # LATERAL INHIBITION OF ATTRIBUTES
            colors = Manipulated_Activation[:, 0:4]
            highest_color = np.argmax(colors, axis = 1)
            colors[:,:] = 0
            colors[np.arange(0, 64), highest_color] = 1

            shapes = Manipulated_Activation[:, 4:8]
            highest_shapes = np.argmax(shapes, axis = 1)
            shapes[:] = 0
            shapes[np.arange(0, 64),highest_shapes] = 1

            sizes = Manipulated_Activation[:, 8:12]
            highest_sizes = np.argmax(sizes, axis = 1)
            sizes[:] = 0
            sizes[np.arange(0, 64), highest_sizes] = 1

            # DISHINIBITION OF A SPECIFIC FEATURE

            if salient_feature == 'color':
                shapes[:] = 0
                sizes[:] = 0

            elif salient_feature == 'form':
                colors[:] = 0
                sizes[:] = 0
            elif salient_feature == 'size':
                colors[:] = 0
                shapes[:] = 0

        elif hidden_output.shape[1] == 4:

            # LATERAL INHIBITION OF ATTRIBUTES
            attributes = Manipulated_Activation[:, 0:4]
            highest_attribute = np.argmax(attributes, axis=1)
            attributes[:] = 0
            attributes[np.arange(0, 64), highest_attribute] = 1

        else:

            Manipulated_Activation[hidden_output > 0.5] = 1
            Manipulated_Activation[hidden_output < 0.5] = 0


    return Manipulated_Activation

# %%%%%%%%%%%%% MACRO FUNCTIONS FOR RANDOM SEARCH OF FITTING PARAMETERS %%%%%%%%%%%%%

def Random_parameters_generation(parameters, parameter_to_generate):

    '''

    This function randomly generates many parameters, using fixed ranges.

    Args:

        parameter_to_generate: list of strings that suggests the parameters to randomly generate
        parameters: default parameters

    return:

            parameters: modified list of parameters (some of them are substituted with randomly generated)
    '''


    if 'CD_contribution' in parameter_to_generate:

        par_options = [0, 0.001, 0.01, 0.1, 1]

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[0] = copy.deepcopy(par_chosen)

    if 'layer_as_input' in parameter_to_generate:

        par_options = ['visible', 'first hidden', 'second hidden']

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[1] = copy.deepcopy(par_chosen)


    if 'second_hidden_layer_units' in parameter_to_generate:

        par_options = [10, 50]

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[2] = copy.deepcopy(par_chosen)


    if 'controller_units' in parameter_to_generate:

        par_options = [4, 10]

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[3] = copy.deepcopy(par_chosen)

        if par_chosen == 4:

            binary_reward_format = True

        else:

            binary_reward_format = False

        parameters[4] = copy.deepcopy(binary_reward_format)

    if 'critic_learning_rate' in parameter_to_generate:

        par_options = [0.1, 0.01, 0.001]

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[5] = copy.deepcopy(par_chosen)



    if 'salient_feature' in parameter_to_generate:

        par_options = ['color', 'form', 'size']

        chosen_option = np.random.randint(len(par_options))

        par_chosen = par_options[chosen_option]

        parameters[6] = copy.deepcopy(par_chosen)


    return parameters

def Random_folder_creation(saves_folder_path):

    '''

    This function creates one folder for each system simulation with a randomic name. It is adapt to parameters search.

    args:

        saves_folder_path: path of saves folders

    return:

        random_name: random name of folder
    '''

    random_name = uuid.uuid4()
    random_name = random_name.hex

    os.makedirs(saves_folder_path + str(random_name))

    os.makedirs(saves_folder_path + str(random_name) + str('\\Weights_layers_activations'))

    os.makedirs(saves_folder_path + str(random_name) + str('\\Training_Visual_Outputs'))

    os.makedirs(saves_folder_path + str(random_name) + str('\\Training_data'))

    return random_name

def save_training_data_reinforced_CD_CHUNKED_SERVER(Number_Single_Chunk_File,
                                                    Reinforces,
                                                    Surprises,
                                                    Accuracies,
                                                    Reconstruction_errors,
                                                    Weight_inputs,
                                                    Weight_inputs_sum,
                                                    Weights_bias_inputs_to_hiddens_sum,
                                                    Weights_bias_hiddens_to_inputs_sum,
                                                    CD_weight,
                                                    learning_rate,
                                                    learning_rate_executor,
                                                    learning_rate_critic,
                                                    R_range,
                                                    R_range_interp,
                                                    ideal_actions, save_path = '',
                                                    Reconstruction_errors_other = 0, Weight_inputs_other = 0,
                       Weight_inputs_sum_other = 0, Weights_bias_inputs_to_hiddens_sum_other = 0,
                       Weights_bias_hiddens_to_inputs_sum_other = 0, learning_rate_other = 0):

    '''

        This function saves the training data of NET in case of reinforcement learning modification

        args:

            Reinforces: list of rewards for each input (64) for each epoc
            Surprises: list of surprises for each input (64) for each epoc
            Accuracies: list of accuracies for each input (64) for each epoc
            Reconstruction_errors: list of rec_errors for each input (64) for each epoc
            Weight_inputs =  network matrix
            Weight_inputs_sum: list of sums of network matrix (I used the sum to have a gross measure of weights)
            Weights_bias_inputs_to_hiddens_sum: list of sums of bias from input to hidden weights matrix (I used the sum to have a gross measure of weights)
            Weights_bias_hiddens_to_inputs_sum: lists of sums of bias from hidden to input weights network matrix (I used the sum to have a gross measure of weights)

            learning_rate: learning rate of CD update
            learning_rate_critic: learning rate of critic update
            R_range: range of reward before interpolation
            R_range_interp: range of reward after interpolation
            ideal_actions: ideal actions that drive the reinforcement learning (labels)

            Reconstruction_errors_other: Reconstruction_errors variable for second RBM
            Weight_inputs_other: Weight_inputs variable for second RBM
            Weights_bias_inputs_to_hiddens_other: Weights_bias_inputs_to_hiddens variable for second RBM
            Weights_bias_hiddens_to_inputs_other: Weights_bias_hiddens_to_inputs variable for second RBM
            learning_rate_other:  learning rate of CD update for second RBM

            save_path: folder path of files

        return:

                None (Training file into a specific folder)

        '''

    if save_path != '':

        path = save_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

    if not isinstance(Weight_inputs_other, (list, np.ndarray)):

        file_name = 'Training_Data_Single_RBM' + str('_') + str(Weight_inputs.shape[0]) + str('_') + str(Weight_inputs.shape[1]) + str('_Reinforced_') + str(Number_Single_Chunk_File)

        params = np.array([learning_rate, learning_rate_critic, learning_rate_executor, CD_weight])

        R_ranges = np.vstack((R_range, R_range_interp))

        np.savez(path + file_name, Inputs_R = Reinforces, Inputs_S = Surprises, Inputs_Acc = Accuracies, Inputs_Rec = Reconstruction_errors,
                 Net_W_NOT_SUM = Weight_inputs, Net_W = Weight_inputs_sum, W_b1 = Weights_bias_inputs_to_hiddens_sum,
                 W_b2 = Weights_bias_hiddens_to_inputs_sum, PARAMS = params, IDEAL = ideal_actions)

    else:

        file_name = 'Training_Data_Whole_DBN' + str('_') + str(Weight_inputs.shape[0]) + str('_') + str(
            Weight_inputs.shape[1]) + str('_') + str(Weight_inputs_other.shape[1]) + str('_Reinforced_') + str(Number_Single_Chunk_File)

        params = np.array([learning_rate, learning_rate_other, learning_rate_critic])

        R_ranges = np.vstack((R_range, R_range_interp))


        np.savez(path + file_name, Inputs_R = Reinforces, Inputs_S = Surprises, Inputs_Acc = Accuracies, Inputs_Rec = Reconstruction_errors,
                 Net_W_NOT_SUM = Weight_inputs, Net_W = Weight_inputs_sum, W_b1 = Weights_bias_inputs_to_hiddens_sum,
                 W_b2 = Weights_bias_hiddens_to_inputs_sum, Inputs_Rec_other = Reconstruction_errors_other,
                 Net_W_NOT_SUM_other = Weight_inputs_other, Net_W_other = Weight_inputs_sum_other, W_b1_other = Weights_bias_inputs_to_hiddens_sum_other,
                 W_b2_other = Weights_bias_hiddens_to_inputs_sum_other,
                 PARAMS = params, R_RANGES = R_ranges, IDEAL = ideal_actions)

def load_training_data_reinforced_CD_JOIN_SERVER(structure, load_path = ''):

    '''

        This function loads the training data of NET in case of reinforcement learning modification

        args:

            structure: vector with the topology of RBM/DBN (e.g. [2352, 150, 10])

            load_path: folder path of files



        return:

            in case of RBM:

            Reinforces_for_each_epoc: list of avg rewards (batch) for each epoc
            Reinforces_for_each_epoc_STD: standard deviation of previous variable
            Surprises_for_each_epoc: list of avg surprises (batch)  for each epoc
            Surprises_for_each_epoc_STD: standard deviation of previous variable
            Accuracies_for_each_epoc: list of avg accuracies (batch)  for each epoc
            Accuracies_for_each_epoc_STD: standard deviation of previous variable
            Reconstruction_errors_for_each_epoc: list of avg rec_errors (batch) for each epoc

            Weight_inputs = network matrix
            Sum_weights_network_for_each_epoc: sum of network matrix (I used the sum to have a gross measure of weights) for each epoc
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc: sum of bias from input to hidden weights matrix for each epoc
            Sum_bias_hiddens_to_inputs_weights_for_each_epoc: sum of bias from hidden to input weights matrix for each epoc

            Reinforces_for_each_input_for_each_epoc: list of rewards for each input (64) for each epoc
            Surprises_for_each_input_for_each_epoc: list of surprises for each input (64) for each epoc
            Accuracies_for_each_input_for_each_epoc: list of accuracies for each input (64) for each epoc


            learning_rate: learning rate of CD update
            learning_rate_critic: learning rate of critic update
            R_range: range of reward before interpolation
            R_range_interp: range of reward after interpolation
            ideal_actions: ideal actions that drive the reinforcement learning (labels)

            in case of DBN there are additional variables:

            learning_rate_Other: learning_rate for second RBM
            Reconstruction_errors_for_each_epoc_Other: ... for second RBM
            Weight_inputs_other = ... for second RBM
            Sum_weights_network_for_each_epoc_Other: ... for second RBM
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other: ... for second RBM
            Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other: ... for second RBM


        '''

    if load_path != '':

        path = load_path

    else:

        path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

    print('Training data processing: START.. \n')

    print('...\n')

    if len(structure) == 2:

        file_name = 'Training_Data_Single_RBM' + str('_') + str(structure[0]) + str('_') + str(structure[1]) + str('_Reinforced_*')

        Chunks_list = glob.glob(path + file_name)

        Chunks_list = natural_sort(Chunks_list)

        for (count, file) in enumerate(Chunks_list):

            #print('...')

            if count == 0:

                Training_data = np.load(file)

                Learning_rate = Training_data['PARAMS'][0]

                Learning_rate_critic = Training_data['PARAMS'][1]

                learning_rate_executor = Training_data['PARAMS'][2]

                CD_weight = Training_data['PARAMS'][3]

                R_range = [0, structure[1]]

                R_range_interp = [0, 1]

                ideal_actions = Training_data['IDEAL']

                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Acc'].T

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Rec'].T


                Sum_weights_network_for_each_input_for_each_epoc_JOIN  = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN  = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN  = Training_data['W_b2'].T

                # CALCULATED DATA

                Reinforces_for_each_epoc_JOIN = np.mean(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Reinforces_for_each_epoc_STD_JOIN = np.std(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_JOIN = np.mean(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_STD_JOIN = np.std(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_JOIN = np.mean(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_STD_JOIN = np.std(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Reconstruction_errors_for_each_epoc_JOIN = np.mean(Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                              axis=0)

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.std(Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                                 axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.mean(Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                axis=0)

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.std(Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                   axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                else:

                    Sum_weights_network_for_each_epoc_JOIN = copy.deepcopy(Sum_weights_network_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN)

            else:

                Training_data = np.load(file)


                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc = Training_data['Inputs_Acc'].T

                Reconstruction_errors_for_each_input_for_each_epoc = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc = Training_data['W_b2'].T

                # CALCULATED DATA

                Reinforces_for_each_epoc = np.mean(Reinforces_for_each_input_for_each_epoc, axis=0)

                Reinforces_for_each_epoc_STD = np.std(Reinforces_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc = np.mean(Surprises_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc_STD = np.std(Surprises_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc = np.mean(Accuracies_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc_STD = np.std(Accuracies_for_each_input_for_each_epoc, axis=0)

                Reconstruction_errors_for_each_epoc = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc,
                    axis=0)

                Reconstruction_errors_for_each_epoc_STD = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc,
                    axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc.ndim == 2:

                    Sum_weights_network_for_each_epoc = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_weights_network_for_each_epoc_STD = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc, axis=0)

                else:

                    Sum_weights_network_for_each_epoc = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc)

                # LEARNING VALUES

                Reinforces_for_each_input_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_input_for_each_epoc_JOIN,
                                                                          Reinforces_for_each_input_for_each_epoc))

                Surprises_for_each_input_for_each_epoc_JOIN = np.hstack((Surprises_for_each_input_for_each_epoc_JOIN,
                                                                         Surprises_for_each_input_for_each_epoc))

                Accuracies_for_each_input_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_input_for_each_epoc_JOIN,
                                                                          Accuracies_for_each_input_for_each_epoc))

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                                                                                     Reconstruction_errors_for_each_input_for_each_epoc))

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                                                                                   Sum_weights_network_for_each_input_for_each_epoc))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN,
                                                                                                  Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN,
                                                                                                  Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc))

                # CALCULATED DATA

                Reinforces_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_epoc_JOIN,
                                                          Reinforces_for_each_epoc))


                Reinforces_for_each_epoc_STD_JOIN = np.hstack((Reinforces_for_each_epoc_STD_JOIN,
                                                          Reinforces_for_each_epoc_STD))

                Surprises_for_each_epoc_JOIN = np.hstack((Surprises_for_each_epoc_JOIN,
                                                          Surprises_for_each_epoc))

                Surprises_for_each_epoc_STD_JOIN = np.hstack((Surprises_for_each_epoc_STD_JOIN,
                                                          Surprises_for_each_epoc_STD))

                Accuracies_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_epoc_JOIN,
                                                          Accuracies_for_each_epoc))

                Accuracies_for_each_epoc_STD_JOIN = np.hstack((Accuracies_for_each_epoc_STD_JOIN,
                                                          Accuracies_for_each_epoc_STD))

                Reconstruction_errors_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_JOIN,
                                                          Reconstruction_errors_for_each_epoc))

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_STD_JOIN,
                                                          Reconstruction_errors_for_each_epoc_STD))

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                          Sum_weights_network_for_each_epoc))

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_STD_JOIN,
                                                          Sum_weights_network_for_each_epoc_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD))

                else:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                          Sum_weights_network_for_each_epoc))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                                                          Sum_bias_inputs_to_hiddens_weights_for_each_epoc))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                                                          Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

        Weights_network = Training_data['Net_W_NOT_SUM']

        print('Training data processing: STOP \n')


        return learning_rate_executor, CD_weight, Learning_rate, Learning_rate_critic, R_range, R_range_interp, ideal_actions, Reinforces_for_each_epoc_JOIN, \
               Reinforces_for_each_epoc_STD_JOIN, Surprises_for_each_epoc_JOIN, Surprises_for_each_epoc_STD_JOIN, Accuracies_for_each_epoc_JOIN, \
               Accuracies_for_each_epoc_STD_JOIN,\
               Reconstruction_errors_for_each_epoc_JOIN, Weights_network, Sum_weights_network_for_each_epoc_JOIN, \
               Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN, Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN, \
               Reinforces_for_each_input_for_each_epoc_JOIN[:, -1], Surprises_for_each_input_for_each_epoc_JOIN[:, -1], Accuracies_for_each_input_for_each_epoc_JOIN[:, -1]






    else:

        file_name = '*Training_Data_Whole_DBN'  + str('_') + str(structure[0]) + str('_') + str(structure[1]) + str('_') + str(structure[2]) + str('_Reinforced_*')

        Chunks_list = glob.glob(path + file_name)

        Chunks_list = natural_sort(Chunks_list)

        for (count, file) in enumerate(Chunks_list):

            #print('...')

            if count == 0:

                Training_data = np.load(file)

                # LEARNING PARAMS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Learning_rate = Training_data['PARAMS'][0]

                Learning_rate_Other = Training_data['PARAMS'][1]

                Learning_rate_critic = Training_data['PARAMS'][2]

                R_range = Training_data['R_RANGES'][0]

                R_range_interp = Training_data['R_RANGES'][1]

                ideal_actions = Training_data['IDEAL']


                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Acc'].T

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = Training_data['W_b2'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc_JOIN = np.mean(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Reinforces_for_each_epoc_STD_JOIN = np.std(Reinforces_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_JOIN = np.mean(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Surprises_for_each_epoc_STD_JOIN = np.std(Surprises_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_JOIN = np.mean(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                Accuracies_for_each_epoc_STD_JOIN = np.std(Accuracies_for_each_input_for_each_epoc_JOIN, axis=0)

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc_JOIN = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                    axis=0)

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                    axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                        axis=0)

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN, axis=0)

                else:

                    Sum_weights_network_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN = Training_data['Inputs_Rec_other'].T

                Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN = Training_data['Net_W_other'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN = Training_data[
                    'W_b1_other'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN = Training_data[
                    'W_b2_other'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other_JOIN = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                Reconstruction_errors_for_each_epoc_Other_STD_JOIN = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                    Sum_weights_network_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                        axis=0)


                else:

                    Sum_weights_network_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN)


            else:

                Training_data = np.load(file)

                Weights_network = Training_data['Net_W_NOT_SUM']

                Weights_network_other = Training_data['Net_W_NOT_SUM_other'].T

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc = Training_data['Inputs_R'].T

                Surprises_for_each_input_for_each_epoc = Training_data['Inputs_S'].T

                Accuracies_for_each_input_for_each_epoc = Training_data['Inputs_Acc'].T

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc = Training_data['Inputs_Rec'].T

                Sum_weights_network_for_each_input_for_each_epoc = Training_data['Net_W'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc = Training_data['W_b1'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc = Training_data['W_b2'].T

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc = np.mean(Reinforces_for_each_input_for_each_epoc, axis=0)

                Reinforces_for_each_epoc_STD = np.std(Reinforces_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc = np.mean(Surprises_for_each_input_for_each_epoc, axis=0)

                Surprises_for_each_epoc_STD = np.std(Surprises_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc = np.mean(Accuracies_for_each_input_for_each_epoc, axis=0)

                Accuracies_for_each_epoc_STD = np.std(Accuracies_for_each_input_for_each_epoc, axis=0)

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc = np.mean(Reconstruction_errors_for_each_input_for_each_epoc,
                                                              axis=0)

                Reconstruction_errors_for_each_epoc_STD = np.std(Reconstruction_errors_for_each_input_for_each_epoc,
                                                                 axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc.ndim == 2:

                    Sum_weights_network_for_each_epoc = np.mean(Sum_weights_network_for_each_input_for_each_epoc,
                                                                axis=0)

                    Sum_weights_network_for_each_epoc_STD = np.std(Sum_weights_network_for_each_input_for_each_epoc,
                                                                   axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc,
                        axis=0)

                else:

                    Sum_weights_network_for_each_epoc = copy.deepcopy(Sum_weights_network_for_each_input_for_each_epoc)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other = Training_data['Inputs_Rec_other'].T

                Sum_weights_network_for_each_input_for_each_epoc_Other = Training_data['Net_W_other'].T

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other = Training_data['W_b1_other'].T

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other = Training_data['W_b2_other'].T

                # ----- SECOND RBM VARIABLES -----

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other = np.mean(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other, axis=0)

                Reconstruction_errors_for_each_epoc_Other_STD = np.std(
                    Reconstruction_errors_for_each_input_for_each_epoc_Other, axis=0)

                if Sum_weights_network_for_each_input_for_each_epoc_Other.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other = np.mean(
                        Sum_weights_network_for_each_input_for_each_epoc_Other, axis=0)

                    Sum_weights_network_for_each_epoc_Other_STD = np.std(
                        Sum_weights_network_for_each_input_for_each_epoc_Other, axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other = np.mean(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD = np.std(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other = np.mean(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD = np.std(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other,
                        axis=0)

                else:

                    Sum_weights_network_for_each_epoc_Other = copy.deepcopy(
                        Sum_weights_network_for_each_input_for_each_epoc_Other)
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other = copy.deepcopy(
                        Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other)
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other = copy.deepcopy(
                        Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other)

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_input_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_input_for_each_epoc_JOIN,
                                                                      Reinforces_for_each_input_for_each_epoc))

                Surprises_for_each_input_for_each_epoc_JOIN = np.hstack((Surprises_for_each_input_for_each_epoc_JOIN,
                                                                         Surprises_for_each_input_for_each_epoc))

                Accuracies_for_each_input_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_input_for_each_epoc_JOIN,
                                                                          Accuracies_for_each_input_for_each_epoc))

                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Reconstruction_errors_for_each_input_for_each_epoc_JOIN,
                     Reconstruction_errors_for_each_input_for_each_epoc))

                Sum_weights_network_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_weights_network_for_each_input_for_each_epoc_JOIN,
                     Sum_weights_network_for_each_input_for_each_epoc))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_JOIN,
                     Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN = np.hstack(
                    (Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_JOIN,
                     Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc))

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- DBN -----

                Reinforces_for_each_epoc_JOIN = np.hstack((Reinforces_for_each_epoc_JOIN,
                                                           Reinforces_for_each_epoc))

                Reinforces_for_each_epoc_STD_JOIN = np.hstack((Reinforces_for_each_epoc_STD_JOIN,
                                                               Reinforces_for_each_epoc_STD))

                Surprises_for_each_epoc_JOIN = np.hstack((Surprises_for_each_epoc_JOIN,
                                                          Surprises_for_each_epoc))

                Surprises_for_each_epoc_STD_JOIN = np.hstack((Surprises_for_each_epoc_STD_JOIN,
                                                              Surprises_for_each_epoc_STD))

                Accuracies_for_each_epoc_JOIN = np.hstack((Accuracies_for_each_epoc_JOIN,
                                                           Accuracies_for_each_epoc))

                Accuracies_for_each_epoc_STD_JOIN = np.hstack((Accuracies_for_each_epoc_STD_JOIN,
                                                               Accuracies_for_each_epoc_STD))
                # ----- FIRST RBM VARIABLES -----

                Reconstruction_errors_for_each_epoc_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_JOIN,
                                                                      Reconstruction_errors_for_each_epoc))

                Reconstruction_errors_for_each_epoc_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_STD_JOIN,
                                                                          Reconstruction_errors_for_each_epoc_STD))

                if Sum_weights_network_for_each_input_for_each_epoc_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                                        Sum_weights_network_for_each_epoc))

                    Sum_weights_network_for_each_epoc_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_STD_JOIN,
                                                                            Sum_weights_network_for_each_epoc_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc_STD))


                else:

                    Sum_weights_network_for_each_epoc_JOIN = np.hstack((Sum_weights_network_for_each_epoc_JOIN,
                                                                        Sum_weights_network_for_each_epoc))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_epoc))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_epoc))

                # LEARNING VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # ----- SECOND RBM VARIABLES -----

                Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Reconstruction_errors_for_each_input_for_each_epoc_Other_JOIN,
                         Reconstruction_errors_for_each_input_for_each_epoc_Other))

                Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_weights_network_for_each_input_for_each_epoc_Other))

                Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_bias_inputs_to_hiddens_weights_for_each_input_for_each_epoc_Other))

                Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN = np.hstack(
                        (Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other_JOIN,
                         Sum_bias_hiddens_to_inputs_weights_for_each_input_for_each_epoc_Other))

                # CALCULATED DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Reconstruction_errors_for_each_epoc_Other_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_Other_JOIN,
                                                                      Reconstruction_errors_for_each_epoc_Other))

                Reconstruction_errors_for_each_epoc_Other_STD_JOIN = np.hstack((Reconstruction_errors_for_each_epoc_Other_STD_JOIN,
                                                                          Reconstruction_errors_for_each_epoc_Other_STD))


                if Sum_weights_network_for_each_input_for_each_epoc_Other_JOIN.ndim == 2:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other))

                    Sum_weights_network_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other_STD))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other))

                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_STD))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other))

                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_STD))


                else:

                    Sum_weights_network_for_each_epoc_Other_JOIN = np.hstack((Sum_weights_network_for_each_epoc_Other_JOIN,
                                                                              Sum_weights_network_for_each_epoc_Other))
                    Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other))
                    Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN = np.hstack((Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN,
                                                                              Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other))

        print('Training data processing: STOP \n')


        return Learning_rate, Learning_rate_Other, Learning_rate_critic, R_range, R_range_interp, ideal_actions, \
               Reinforces_for_each_epoc_JOIN, Reinforces_for_each_epoc_STD_JOIN, Surprises_for_each_epoc_JOIN, Surprises_for_each_epoc_STD_JOIN, \
               Accuracies_for_each_epoc_JOIN, Accuracies_for_each_epoc_STD_JOIN, Reconstruction_errors_for_each_epoc_JOIN, Weights_network, \
               Sum_weights_network_for_each_epoc_JOIN, Sum_bias_inputs_to_hiddens_weights_for_each_epoc_JOIN, Sum_bias_hiddens_to_inputs_weights_for_each_epoc_JOIN, \
               Reconstruction_errors_for_each_epoc_Other_JOIN, Weights_network_other, Sum_weights_network_for_each_epoc_Other_JOIN, Sum_bias_inputs_to_hiddens_weights_for_each_epoc_Other_JOIN,\
               Sum_bias_hiddens_to_inputs_weights_for_each_epoc_Other_JOIN, Reinforces_for_each_input_for_each_epoc_JOIN[:, -1], Surprises_for_each_input_for_each_epoc_JOIN[:, -1],\
               Accuracies_for_each_input_for_each_epoc_JOIN[:, -1]


def save_training_data_joined_SERVER(params, Ideals,  training_data, performance_data, path, random_name_simulation):

    '''

    This function saves the joined data of each simulation.

    args:
        params: parameters of simulation
        training_data: training data of simulation
        performance_data: summary of performances (max r, min rec errors, et)
        path: save path
        random_name_simulation: name of simulation

    return: none (file in folder)

    '''

    np.savez(path + str('_Simulation_data_') + str(random_name_simulation), Params = params, Ideals_actions = Ideals, Training_data = training_data, Perform_data = performance_data)

def load_training_data_joined_SERVER(Saves_path):

    '''

    This function loads the joined data of each simulation, creating a matrix that stores the final results
    (max r, min rec. err...) for each simualtion, and many matrices for training data (one for rewards/epocs for each simulation, etc)

    args:

        path: saves path

    return:

    Final_results:
    Parameters:


    '''

    File_name = str('_Simulation_data_')
    Sub_folder_name = str('\\Training_data\\')

    Chunks_list = glob.glob(Saves_path + str('*'))

    Performances_achieved_JOINED = []
    Params_JOINED = []
    Ideals_JOINED = []


    for (count, sim) in enumerate(Chunks_list):

        Sim_data = np.load(sim + Sub_folder_name + File_name + str(sim[-32:]) + str('.npz'))

        Performances_achieved = Sim_data['Perform_data']

        Params = Sim_data['Params']

        Ideals = Sim_data['Ideals_actions']



        # Training_data = Sim_data['Training_data']

        Performances_achieved_JOINED.append(Performances_achieved)

        Params_JOINED.append(Params)

        Ideals_JOINED.append(Ideals)

    Performances_achieved_JOINED = np.vstack(Performances_achieved_JOINED)
    Params_JOINED = np.vstack(Params_JOINED)
    #Ideals_JOINED = np.vstack(Ideals_JOINED)

    return Performances_achieved_JOINED, Params_JOINED, Chunks_list

def statistics_server_data(Params, Performances_achieved, Filters, POI):

    '''

    This function executes many statistic computations on parameters and achieved results of simulations.

    args:


        Params: parameters of server simulations

        Performances_achieved: performances achieved of server simulations - [Max_R, Max_Acc, min(Errors_Epocs), Rec_error_batch_DBN]

        Filters: filters to apply to the simulations sample

       POI: "Parameter of Interest" of wich the function plots relation with achieved performances.

    return:

        Fig: figure with graphical analisys

    '''

    # MY PC SIMULATIONS FORMAT

    # PERF. ACHIEVED FORMAT = [Max_R, Max_Acc, min(Errors_Epocs), Rec_error_batch_DBN]
    # PARAMS FORMAT = [Executor_Input, learning_rate, critic_learning_rate, learning_rate_Executor, Binary_Reward_format, CD_weight, second_hidden]_units_RBM_2]

    # SERVER SIMULATIONS FORMAT

    # PERF. ACHIEVED FORMAT = [Max_R, Max_Acc, Min. Rec. error (hidden), Min. Rec. error (Visible)]
    # PARAMS FORMAT = [Executor_Input, critic_learning_rate, Binary_Reward_format, CD_weight, salient_feature,  Network_weights[0].shape[1]]





    # FILTERS ----------------

    for (count, cond_value) in enumerate(Filters):

        if cond_value[0] == 'input_layer':     # Executor input

            indices_layer_input_filter = np.where(Params[:, 0] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_layer_input_filter]
            Params = Params[indices_layer_input_filter]

        if cond_value[0] == 'critic_learning_rate':        # Critic learning rate

            indices_critic_learning_rate_filter = np.where(Params[:, 1] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_critic_learning_rate_filter]
            Params = Params[indices_critic_learning_rate_filter]

        if cond_value[0] == 'binary_reward_format':        # Binary_reward_format

            indices_binary_reward_format_filter = np.where(Params[:, 2] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_binary_reward_format_filter]
            Params = Params[indices_binary_reward_format_filter]

        if cond_value[0] == 'cd_weight ':  # Binary_reward_format

            indices_cd_weight_filter = np.where(Params[:, 3] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_cd_weight_filter]
            Params = Params[indices_cd_weight_filter]

        if cond_value[0] == 'salient_feature':

            indices_salient_feature_filter = np.where(Params[:, 4] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_salient_feature_filter]
            Params = Params[indices_salient_feature_filter]


        if cond_value[0]  == 'hidden_second_units':       # Second Hidden units

            indices_hidden_second_units_filter = np.where(Params[:, 5] == cond_value[1])
            Performances_achieved = Performances_achieved[indices_hidden_second_units_filter]
            Params = Params[indices_hidden_second_units_filter]

        if cond_value[0] == 'max reward':

            indices_max_r_filter = np.where(Performances_achieved[:, 0] > cond_value[1])
            Performances_achieved = Performances_achieved[indices_max_r_filter]
            Params = Params[indices_max_r_filter]



    Max_r = np.array(Performances_achieved[:, 0])
    Max_Acc = np.array(Performances_achieved[:, 1])
    Min_rec_err_hidden = np.array(Performances_achieved[:, 2])
    Min_rec_err_visible = np.array(Performances_achieved[:, 3])

    # MY PC PARAMETERS FORMAT

    # Executor_layer_as_input = np.array(Params[:, 0])
    # Learning_rate_CD = np.array(Params[:, 1])
    # Learning_rate_critic = np.array(Params[:, 2])
    # Learning_rate_REINFORCE = np.array(Params[:, 3])
    # Binary_reward_format = np.array(Params[:, 4])
    # CD_Weight = np.array(Params[:, 5])

    # MY PC PARAMETERS FORMAT

    Executor_layer_as_input = np.array(Params[:, 0])
    Learning_rate_critic = np.array(Params[:, 1])
    Binary_reward_format = np.array(Params[:, 2])
    CD_Weight = np.array(Params[:, 3])
    Salient_feature =  np.array(Params[:, 4])
    Hidden_second_units =  np.array(Params[:, 5])



    if POI == 'CD_weight':

        Parameter_to_valuate = copy.deepcopy(CD_Weight)

    elif POI == 'Executor_layer_as_input':

        Parameter_to_valuate = copy.deepcopy(Executor_layer_as_input)

    # elif POI == 'Hidden_second_units':
    #
    #     Parameter_to_valuate = copy.deepcopy(Hidden_second_units)

    elif POI == 'Binary_Reward_format':

        Parameter_to_valuate = copy.deepcopy(Binary_reward_format)

    elif POI == 'critic_learning_rate':

        Parameter_to_valuate = copy.deepcopy(Learning_rate_critic)

    elif POI == 'salient_feature':

        Parameter_to_valuate = copy.deepcopy(Salient_feature)


    elif POI == 'hidden_second_units':

        Parameter_to_valuate = copy.deepcopy(Hidden_second_units)


    Labels_achievements = ['Max r', 'Max acc', 'Min rec. error (Hidden)', 'Min rec. error (Visible)']

    # PLOTS

    Fig = plt.figure(figsize=(19.20, 10.80))

    plt.suptitle('Role of ' + str(POI) + str(' in performance achievements (Max reward, Max accuracy, Min. rec. error (on Hidden), '
                                             'Min. rec. error (on Visible))'))

    num_rows = 2
    num_columns = 2

    # PARAMETER (X-AXIS) VALUES

    indices = Parameter_to_valuate.argsort()

    X = Parameter_to_valuate[indices]

    X_categories = sorted(dict(zip(reversed(X), range(len(X) - 1, -1, -1))).values())

    X_values_for_category = X[X_categories]


    for single_perf_coeff in range(0, Performances_achieved.shape[1]):

        Specific_perfomance_value = np.array(Performances_achieved[:, single_perf_coeff])

        Y = Specific_perfomance_value[indices]

        Plot = plt.subplot(num_rows, num_columns, single_perf_coeff + 1)

        for point in range(0, len(Y)):

            plt.scatter(np.where(X_values_for_category == X[point]), Y[point], color='black')

        Plot.set_xticks(range(0, len(X_values_for_category)))

        means_values_categories = []
        standard_deviations_values_categories = []

        for count, cat_ind in enumerate(X_categories):

            repetitions = np.where(X == X[cat_ind])
            mean_cat_values = np.mean(Y[repetitions])
            std_cat_values = np.std(Y[repetitions])

            means_values_categories.append(mean_cat_values)
            standard_deviations_values_categories.append(std_cat_values)

        X_means = range(0, len(X_values_for_category))

        for point in range(0, len(X_means)):

            x_single_point = X_means[point]
            y_single_point = means_values_categories[point]
            sd_single_point = standard_deviations_values_categories[point]

            plt.errorbar(x_single_point, y_single_point, sd_single_point, color = 'blue', marker = 'o')

        plt.title(' Correlation ' + str(POI) + str(' - ') + str(Labels_achievements[single_perf_coeff]))
        plt.xlabel(POI)
        plt.ylabel(str(Labels_achievements[single_perf_coeff]))
        Plot.set_xticklabels(X_values_for_category)



    return Fig

def select_specific_simulation_to_analyze(simulations_paths, simulations_performances,
                                          simulations_parameters, simulations_conditions, salient_feature_for_selection):

    '''
    
    This function extracts a specific simulation, on the basis of a specific condition (e.g the simulation with highest reward),
     to be after analyzed by the "test enviroment" of library (training data, spread, reconstructions...etc)
    
    args
    
        simulations_paths: list of all simulations paths 
        simulations_performances: list of final performances of simulation (r,acc, rec error hidden, rec error visible)
        simulations_parameters: list of parameters for each simulation (learning_rate critic, cd weight, binary reward, etc)
        simulations_conditions: list of conditiosn to apply many filters (e.g. ['salient feature', 'color'] focus on sim.s that focus on colour)
        salient_feature_for_selection: performance coefficient to use to extract the "winning simulation"

    return:

        extracted_simulation_parameters:parameters of extracted simulation
        extracted_simulation_performances:performances coefficient of extracted simulation
        extracted_simulation_path: path of extracted simulation


    '''

    # MY PC SIMULATIONS FORMAT

    # PERF. ACHIEVED FORMAT = [Max_R, Max_Acc, min(Errors_Epocs), Rec_error_batch_DBN]
    # PARAMS FORMAT = [Executor_Input, learning_rate, critic_learning_rate, learning_rate_Executor, Binary_Reward_format, CD_weight, second_hidden]_units_RBM_2]

    # SERVER SIMULATIONS FORMAT

    # PERF. ACHIEVED FORMAT = [Max_R, Max_Acc, Min. Rec. error (hidden), Min. Rec. error (Visible)]
    # PARAMS FORMAT = [Executor_Input, critic_learning_rate, Binary_Reward_format, CD_weight, salient_feature,  Network_weights[0].shape[1]]

    for (count, cond_value) in enumerate(simulations_conditions):

        if cond_value[0] == 'input_layer':     # Executor input

            indices_layer_input_filter = np.where(simulations_parameters[:, 0] == cond_value[1])
            simulations_performances = simulations_performances[indices_layer_input_filter]
            simulations_parameters = simulations_parameters[indices_layer_input_filter]

            simulations_paths = np.array(simulations_paths)[indices_layer_input_filter]

        if cond_value[0] == 'critic_learning_rate':        # Critic learning rate

            indices_critic_learning_rate_filter = np.where(simulations_parameters[:, 1] == cond_value[1])
            simulations_performances = simulations_performances[indices_critic_learning_rate_filter]
            simulations_parameters = simulations_parameters[indices_critic_learning_rate_filter]
            simulations_paths = simulations_paths[indices_critic_learning_rate_filter]

        if cond_value[0] == 'binary_reward_format':        # Binary_reward_format

            indices_binary_reward_format_filter = np.where(simulations_parameters[:, 2] == cond_value[1])
            simulations_performances = simulations_performances[indices_binary_reward_format_filter]
            simulations_parameters = simulations_parameters[indices_binary_reward_format_filter]
            simulations_paths = simulations_paths[indices_binary_reward_format_filter]

        if cond_value[0] == 'cd_weight':  # Binary_reward_format

            indices_cd_weight_filter = np.where(simulations_parameters[:, 3] == cond_value[1])
            simulations_performances = simulations_performances[indices_cd_weight_filter]
            simulations_parameters = simulations_parameters[indices_cd_weight_filter]
            simulations_paths = simulations_paths[indices_cd_weight_filter]

        if cond_value[0] == 'salient_feature':

            indices_salient_feature_filter = np.where(simulations_parameters[:, 4] == cond_value[1])
            simulations_performances = simulations_performances[indices_salient_feature_filter]
            simulations_parameters = simulations_parameters[indices_salient_feature_filter]
            simulations_paths = simulations_paths[indices_salient_feature_filter]

        if cond_value[0]  == 'hidden_second_units':       # Second Hidden units

            indices_hidden_second_units_filter = np.where(simulations_parameters[:, 5] == cond_value[1])
            simulations_performances = simulations_performances[indices_hidden_second_units_filter]
            simulations_parameters = simulations_parameters[indices_hidden_second_units_filter]
            simulations_paths = simulations_paths[indices_hidden_second_units_filter]



    # SALIENT PERFOMANCE COEFFICIENT

    if salient_feature_for_selection == 'max_reward':

        extracted_simulation_index = np.argmax(simulations_performances[:, 0])


    elif salient_feature_for_selection == 'max_accuracy':

        extracted_simulation_index = np.argmax(simulations_performances[:, 1])


    elif salient_feature_for_selection == 'min_rec_error_hidden':

        extracted_simulation_index = np.argmin(simulations_performances[:, 2])


    elif salient_feature_for_selection == 'min_rec_error_visible':

        extracted_simulation_index = np.argmin(simulations_performances[:, 3])


    extracted_simulation_path = simulations_paths[extracted_simulation_index]
    extracted_simulation_parameters = simulations_parameters[extracted_simulation_index]
    extracted_simulation_performances = simulations_performances[extracted_simulation_index]

    return extracted_simulation_parameters, extracted_simulation_performances, extracted_simulation_path

# %%%%%%%%%%%%% FUNCTIONS FOR KUKA SIMULATION %%%%%%%%%%%%%












