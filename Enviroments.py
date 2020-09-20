from Basic_Functions import *
from System_Components_Functions import *
import datetime
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')


# %%%%%%%%%%%%% MACRO FUNCTIONS TO TRAIN THE SYSTEM %%%%%%%%%%%%%


# SINGLE RBM TRAINING

def RBM_Training(Network_weights,
                 dataset,
                 labels_expected_actions,

                 Batch_on,
                 number_of_resonances,
                 save_choice, Graphic_on, Prints_on,
                 Save_Folders_Paths,

                 Tot_epocs = 50000, learning_rate = 0.01, alfa = 0, target_sparsety = 1, target_min_error = 0.5,
                 Network_weights_other = (0, 0, 0)):


    '''

    This is a macro-function that executes a training of weights of an RBM. It shows specific input modality
    (batch mode of 64 or stocastic/online mode of 1 input), number of resonaces (variable int number).



    args:

        Network_weights: weight of RBM to train. It is composed by net weights and two biases
        dataset: whole dataset fro training
        labels_expected_actions: labels to recognize the input features (color, form and size)
        Batch_on: specific input modality (64 inputs or  1 input)
        number_of_resonances: number of steps into Gibbs sampling process
        save_choice: save choice (Boolean)
        Graphic_on: graphical computations choice (Boolean)
        Prints_on: prints on the command line choice (Boolean)
        Save_Folders_Paths: folders of saves
        Tot_epocs: number of maximum epocs to train
        learning_rate: learnin rate of CD
        alfa: momentum
        target_sparsety: sparsety required to net
        target_min_error : minimum reconstruction error to achieve for stopping the learning process
        Network_weights_other: Network_weights of first RBM

    return:

        Network_weights: trained weights of RBM. It is composed by net weights and two biases

    '''

    # ONLINE MODE / BATCH ASSIGNATION, STEPS_TO_SAVE_DATA, PATHS, AND OTHER INIT VARIABLES

    Total_inputs = 64

    if Batch_on == True:
        Batch_size = 64
    else:
        Batch_size = 1

    step_graphic_save_functions = 1000
    Number_Single_Chunk_File = 0
    epoc_NOT_CHUNKED = 0

    batch_single, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev, \
    Rec_Epocs, Hiddens_Activ_Epocs_each_input, Errors_Epocs, STDs_Errors_Epocs, Errors_Epocs_each_input, Weights_SUM_Epocs_each_input, \
    Weights_SUM_Epocs_bias_1_each_input, Weights_SUM_Epocs_bias_2_each_input, Weights_SUM_Epocs_input, Weights_SUM_Epocs_bias_1, \
    Weights_SUM_Epocs_bias_2, epoc = Initialize_variabiles_RBM()

    Ideal_actions = labels_expected_actions[0]
    Ideal_actions_batch = labels_expected_actions[1]
    Legend_lab = labels_expected_actions[2]

    # PATHS

    weights_save_path = Save_Folders_Paths[0]

    training_data_save_path = Save_Folders_Paths[1]

    Saves_path_training_visual_outputs = Save_Folders_Paths[2]

    # PATHS CLEANING

    clean_training_data_folder(training_data_save_path)

    clean_training_data_folder(Saves_path_training_visual_outputs)

    print('')
    print("Start RBM Training... \n")
    Starting_time = datetime.datetime.today()


    # HERE I TRY TO SHUFFLE DATASET

    #dataset, Ideal_actions = Unison_shuffle(copy.deepcopy(dataset), copy.deepcopy(Ideal_actions))

    while epoc_NOT_CHUNKED < Tot_epocs:    # START EPOC LOOP %%%%%%%%%%%%%%%%%%


        if not Batch_on:

            Errors_online = []
            Rec_online = []
            Hiddens_Activ_online = []

            Weights_online = []
            Weights_online_b_1 = []
            Weights_online_b_2 = []

            Weights_online_SUM = []
            Weights_online_b_1_SUM = []
            Weights_online_b_2_SUM = []

        Starting_time_epoc = datetime.datetime.today()

        # START BATCH/INPUTS LOOP %%%%%%%%%%%%%%%%%%

        while batch_single < Total_inputs / Batch_size:

            # IF SECOND RBM TRAINING...CREATE INPUT (I.E. FIRST RBM HIDDEN ACTIVATION)

            if dataset.shape[1] == (28 * 28 * 3) and Network_weights[0].shape[0] != (28 * 28 * 3):

                original_input, _ = Activation_Hidden_Layer(
                    Get_input(dataset[batch_single:(batch_single + Batch_size), :]),
                    Network_weights_other[0], Network_weights_other[1], True)

            else:

                original_input = Get_input(dataset[batch_single:(batch_single + Batch_size), :])

            #original_input += np.random.uniform(-0.1, 0.1, (original_input.shape[0], original_input.shape[1]))  # noise, case

            input = copy.deepcopy(original_input)

            # START RESONANCES/GIBBS SAMPLING %%%%%%%%%%%%%%%%%%

            for res in range(0, number_of_resonances):


                if res == 0:

                    Activation_hidden_first_spread_original, penalty = Activation_Hidden_Layer(input, Network_weights[0], Network_weights[1], True,
                                                            learning_rate, target_sparsety)

                    Activation_hidden_first_spread_original_Probabilities, _ = Activation_Hidden_Layer(input, Network_weights[0], Network_weights[1], False,
                                                            learning_rate, target_sparsety)

                    Activation_hidden_first_spread = copy.deepcopy(Activation_hidden_first_spread_original)

                    rec_input_original_Probabilities = Input_reconstruction(Activation_hidden_first_spread_original_Probabilities, Network_weights[0], Network_weights[2])


                    rec_input_original = Input_reconstruction(Activation_hidden_first_spread, Network_weights[0], Network_weights[2])


                    rec_input = copy.deepcopy(rec_input_original)





                else:

                    Activation_hidden_first_spread, _ = Activation_Hidden_Layer(input, Network_weights[0], Network_weights[1], False,
                                                                               learning_rate, target_sparsety)



                    rec_input = Input_reconstruction(Activation_hidden_first_spread, Network_weights[0], Network_weights[2])



                Rec_error_for_inputs, Rec_error_batch, St_dev_error_batch = Reconstruction_errors(original_input, rec_input_original)

                input = copy.deepcopy(rec_input)

            # STOP RESONANCES/GIBBS SAMPLING %%%%%%%%%%%%%%%%%%

            Activation_hidden_second_spread, _ = Activation_Hidden_Layer(rec_input, Network_weights[0], Network_weights[1])

            Weight_inputs_update, Weights_bias_inputs_to_hiddens_update, Weights_bias_hiddens_to_inputs_update = Potential_update_CD(original_input,
                                                                             Activation_hidden_first_spread_original,
                                                                             rec_input, Activation_hidden_second_spread, learning_rate,
                                                                             alfa, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev,
                                                                             Weights_bias_hiddens_to_inputs_update_prev, penalty)

            Network_weights[0], Network_weights[1], Network_weights[2], \
            Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, \
            Weights_bias_hiddens_to_inputs_update_prev = Effective_update(Network_weights[0], Network_weights[1], Network_weights[2], # remeber to exclude the .T from b1 and b2
                                                                          Weight_inputs_update, Weights_bias_inputs_to_hiddens_update.T,
                                                                          Weights_bias_hiddens_to_inputs_update.T)


            # STOCASTIC/ONLINE DATA COLLECTION

            if not Batch_on:


                Errors_online.append(Rec_error_batch)
                Rec_online.append(rec_input_original_Probabilities)
                Hiddens_Activ_online.append(Activation_hidden_first_spread_original)

                Weights_online.append(Network_weights[0])
                Weights_online_b_1.append(Network_weights[1])
                Weights_online_b_2.append(Network_weights[2])

                Weights_online_SUM.append(np.sum(Network_weights[0]))
                Weights_online_b_1_SUM.append(np.sum(Network_weights[1]))
                Weights_online_b_2_SUM.append(np.sum(Network_weights[2]))


            batch_single += 1

        # STOP BATCH/INPUTS LOOP %%%%%%%%%%%%%%%%%%

        batch_single = 0


        # DATA COLLECTION FOR EACH EPOC%%%%%%%%%%%%%%%%%%

        # BATCH LEARNING


        if Batch_on:


            Rec_Epocs.append(rec_input_original)
            Hiddens_Activ_Epocs_each_input.append(Activation_hidden_first_spread_original)
            Errors_Epocs_each_input.append(Rec_error_for_inputs)
            Errors_Epocs.append(Rec_error_batch)
            STDs_Errors_Epocs.append(St_dev_error_batch)

        # STOCASTIC/ONLINE LEARNING

        if not Batch_on:

            Rec_online = np.vstack(Rec_online)
            Rec_Epocs.append(Rec_online)
            Hiddens_Activ_Epocs_each_input.append(Hiddens_Activ_online)

            Mean_Err = np.mean(Errors_online)
            STD_Err = np.std(Errors_online)

            Errors_Epocs.append(Mean_Err)
            STDs_Errors_Epocs.append(STD_Err)
            Errors_Epocs_each_input.append(Errors_online)

            Mean_W = np.mean(Weights_online_SUM)
            Mean_W_b_1 = np.mean(Weights_online_b_1_SUM)
            Mean_W_b_2 = np.mean(Weights_online_b_2_SUM)

            Weights_SUM_Epocs_input.append(Mean_W)
            Weights_SUM_Epocs_bias_1.append(Mean_W_b_1)
            Weights_SUM_Epocs_bias_2.append(Mean_W_b_2)

            Weights_SUM_Epocs_each_input.append(Weights_online_SUM)
            Weights_SUM_Epocs_bias_1_each_input.append(Weights_online_b_1_SUM)
            Weights_SUM_Epocs_bias_2_each_input.append(Weights_online_b_2_SUM)

        # PRINTS AND LEARNING RESULTS CALCULATION FOR EACH EPOC%%%%%%%%%%%%%%%%%%

        if epoc == 0:

            Min_error = np.around(Errors_Epocs[epoc], decimals=5)
            sd_min_error = np.around(STDs_Errors_Epocs[epoc], decimals=5)

            min_error_prev = copy.deepcopy(Errors_Epocs[epoc])

            # COMPUTATION OF PERFORMANCE LEVEL FOR RECONSTRUCTION

            Span_Rec_error =  1 - target_min_error
            min_step_error_to_100 = Span_Rec_error / 100
            learning_percentual_progress = (1 - Min_error)  / min_step_error_to_100


            found_min_error = 0

        else:

            Min_error = min(np.around(Errors_Epocs[epoc], decimals=5), Min_error)

            learning_percentual_progress = (1 - Min_error)  / min_step_error_to_100

            if Min_error < min_error_prev:

                found_min_error = epoc
                sd_min_error= np.around(STDs_Errors_Epocs[epoc], decimals=5)

            min_error_prev = copy.deepcopy(Min_error)

        End_time_epoc = datetime.datetime.today()

        if Prints_on:

            print('')
            print(str(" Epoc ") + str(epoc_NOT_CHUNKED) + str(' / ') + str(Tot_epocs) + str(' (') + str(
                np.around((epoc_NOT_CHUNKED / Tot_epocs) * 100, decimals=2)) + str(' %, stimated max time to end: ')
                  + str((End_time_epoc - Starting_time_epoc) * (Tot_epocs - epoc_NOT_CHUNKED)) + str(')') + str(
                " ------------------------------ "))
            print('')

            print(
                str(' *** Performance achieved ') + str('(Rec. Error Target: ') + str(
                    target_min_error)
                + str(') : '))

            print('')
            print(str('        ') + str('+++ Rec. Error progress: ') + str(
                np.around(learning_percentual_progress, decimals=2))
                  + str(' % ') + str(' ///  Min error achieved: ') + str(np.around(Min_error, decimals=4)) + str(
                ' +- ') + str(sd_min_error)
                  + str(' at epoc ') + str(found_min_error) + str(' (Epocs without changes: ') + str(epoc - found_min_error)
                  + str(')'))
            print('')
            print(str(" *** Epoc Error = ") + str(np.around(Errors_Epocs[epoc], decimals=4)) + str(" +- ") +
                        str(np.around(STDs_Errors_Epocs[epoc], decimals=4)))

        epoc += 1
        epoc_NOT_CHUNKED += 1

        # GRAPHICAL FUNCTIONS%%%%%%%%%%%%%%%%%%

        if Graphic_on and (epoc_NOT_CHUNKED != 0) and ((epoc_NOT_CHUNKED % step_graphic_save_functions) == 0):

            # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

            Visible_CD_Pure, Explained_Var_Visible, Explained_Var_ratio_Visible = dim_reduction(dataset, 3)

            if Network_weights_other != (0, 0, 0):


                # INPUTS RECONSTRUCTIONS%%%%%

                Rec_inputs_img, Rec_inputs = Window_builder_all_inputs_recostructions(np.array(Rec_Epocs[-1]),
                                                                                      Network_weights_other[0],
                                                                                      Network_weights_other[2])

                Rec_error_for_inputs_DBN, Rec_error_batch_DBN, St_dev_error_batch_DBN = Reconstruction_errors(dataset, Rec_inputs)
                print('')
                print(str(" *** Error_Whole DBN = ") + str(Rec_error_batch_DBN) + str(" +- ") +
                      str(St_dev_error_batch_DBN))
                print('')

                # WEIGHTS DISTRIBUTIONS PLOTS

                Weights_FIRST_img = Windows_builder_network_weights(Network_weights_other[0],
                                                                    Network_weights_other[1],
                                                                    Network_weights_other[2])

                Weights_SECOND_img = Windows_builder_network_weights(Network_weights[0],
                                                                     Network_weights[1],
                                                                     Network_weights[2])

                Weights_FIRST_img.savefig(
                    Saves_path_training_visual_outputs + str('Weights_FIRST_RBM.png'))

                plt.close()

                Weights_SECOND_img.savefig(
                    Saves_path_training_visual_outputs + str('Weights_SECOND_RBM.png'))

                plt.close()

                Rec_inputs_img.savefig(
                    Saves_path_training_visual_outputs + str('DBN_RECONSTRUCTED_INPUTS.png'))

                plt.close()

                # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                Visible_CD_Biased, Explained_Var_Visible_Biased, Explained_Var_ratio_Visible_Biased = dim_reduction(Rec_inputs, 3)

                # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                FIRST_HIDDEN_CD_Biased, Explained_Var_FIRST_HIDDEN_Biased, Explained_Var_ratio_FIRST_HIDDEN_Biased = dim_reduction(
                    Rec_Epocs[-1], 3)

                # DIMENSIONAL REDUCTIONS (SECOND HIDDEN LAYER)%%%%%

                SECOND_HIDDEN_CD_Biased, Explained_Var_SECOND_HIDDEN_Biased, Explained_Var_ratio_SECOND_HIDDEN_Biased = dim_reduction(
                    Hiddens_Activ_Epocs_each_input[-1], 3)



            else:

                # INPUTS RECONSTRUCTIONS%%%%%

                Rec_inputs_img, Rec_inputs = Window_builder_all_inputs_recostructions(np.array(Rec_Epocs[-1]))


                Rec_inputs_img.savefig('.\\Training_Visual_Outputs\\RBM_RECONSTRUCTED_INPUTS.png')

                plt.close()

                # WEIGHTS DISTRIBUTIONS PLOTS

                Weights_img = Windows_builder_network_weights(Network_weights[0],
                                                                     Network_weights[1],
                                                                     Network_weights[2])

                Weights_img.savefig(
                    Saves_path_training_visual_outputs + str('Weights_FIRST_RBM.png'))

                plt.close()

                # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                Visible_CD_Biased, Explained_Var_Visible_Biased, Explained_Var_ratio_Visible_Biased = dim_reduction(
                    Rec_inputs, 3)

                # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                FIRST_HIDDEN_CD_Biased, Explained_Var_FIRST_HIDDEN_Biased, Explained_Var_ratio_FIRST_HIDDEN_Biased = dim_reduction(
                    Hiddens_Activ_Epocs_each_input[-1], 3)

            # 3D PLOTS OF LAYER ACTIVATIONS%%%%%


            Visible_activation_disentanglement_img = Window_builder_layers_activations(
                Visible_CD_Biased,
                Ideal_actions,
                Ideal_actions_batch,
                Legend_lab,
                'input',
                2,
            )

            Visible_activation_disentanglement_img.savefig(
                Saves_path_training_visual_outputs + str('DISENTANGLEMENT_VISIBLE.png'))

            plt.close()

            First_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                FIRST_HIDDEN_CD_Biased,
                Ideal_actions,
                Ideal_actions_batch,
                Legend_lab,
                'FHL',
                2,
            )

            First_Hidden_activation_disentanglement_img.savefig(
                Saves_path_training_visual_outputs + str('DISENTANGLEMENT_FIRST_HIDDEN.png'))

            plt.close()


            if Network_weights_other != (0, 0, 0):

                Second_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                    SECOND_HIDDEN_CD_Biased,
                    Ideal_actions,
                    Ideal_actions_batch,
                    Legend_lab,
                    'SHL',
                    2,
                )

                Second_Hidden_activation_disentanglement_img.savefig(
                    Saves_path_training_visual_outputs + str('DISENTANGLEMENT_SECOND_HIDDEN.png'))

                plt.close()

        # SAVING FUNCTIONS%%%%%%%%%%%%%%%%%%

        if save_choice and epoc_NOT_CHUNKED != 0 and ((epoc_NOT_CHUNKED % step_graphic_save_functions) == 0):

            save_weights(Network_weights[0], Network_weights[1], Network_weights[2], Hiddens_Activ_Epocs_each_input[-1],
                         weights_save_path)
            save_weights(Network_weights[0], Network_weights[1], Network_weights[2], Hiddens_Activ_Epocs_each_input[-1],
                         weights_save_path)

            save_training_data_CHUNKED(Number_Single_Chunk_File, [Network_weights[0].shape[0], Network_weights[0].shape[1]], Errors_Epocs,
                               STDs_Errors_Epocs, save_path = training_data_save_path)



            Number_Single_Chunk_File += 1

            print('')
            print("...TRAINING DATA AND WEIGHTS SAVED!")
            print('')


        # VARIABLES RESET/RE-INITIALIZATION%%%%%%%%%%%%%%%%%%

        if epoc_NOT_CHUNKED != 0 and (epoc_NOT_CHUNKED % step_graphic_save_functions) == 0:

            print('..RESET VARIABLES...')
            print('')

            # INIT VARIABLES

            batch_single, Weight_inputs_update_prev, Weights_bias_inputs_to_hiddens_update_prev, Weights_bias_hiddens_to_inputs_update_prev, \
            Rec_Epocs, Hiddens_Activ_Epocs_each_input, Errors_Epocs, STDs_Errors_Epocs, Errors_Epocs_each_input, Weights_SUM_Epocs_each_input, \
            Weights_SUM_Epocs_bias_1_each_input, Weights_SUM_Epocs_bias_2_each_input, Weights_SUM_Epocs_input, Weights_SUM_Epocs_bias_1, \
            Weights_SUM_Epocs_bias_2, epoc = Initialize_variabiles_RBM()



        # STOP TRAINING CONDITIONS %%%%%%%%%%%%%%%%%%

        if Min_error < target_min_error:

            break

    End_time = datetime.datetime.today()

    print('')
    print(' ------------------------------ ')
    print('')
    print(' ...End RBM Training (session_time =  ' + str(End_time - Starting_time) + str(')'))
    print('')
    print('There is a difference between knowing the path and walking the path.')
    print('')

    return Network_weights


# WHOLE ARCHITECTURE_TRAINING

def Robotic_Agent_Training(
                                 Sensory_Topology,
                                 N_res_1,
                                 N_res_2,
                                 N_res_3,
                                 learning_rate_CD,
                                 learning_rate_RL,
                                 CD_weight,

                                 Controller_Topology,
                                 Controller_learning_rate,

                                 Evaluator_Topology,
                                 Evaluator_learning_rate,


                                 dataset,
                                 save_choice,
                                 Graphic_on,
                                 Prints_on,
                                 Save_Folders_Paths,
                                 salient_feature = 'none',
                                 Tot_epocs = 50000,
                                 ):


    '''

    This function allows the execution of an enviroment in which I train a neuro-robotic architecture.

    args:

         Sensory_Topology: weight of RBM to train. It is composed by net weights and two biases
         N_res_1: number of steps into Gibbs sampling process (first RBM),
         N_res_2: number of steps into Gibbs sampling process (second RBM),
         N_res_3: number of steps into Gibbs sampling process (whole DBN),

         learning_rate_CD: learnin rate of CD,
         learning_rate_RL: learnin rate of REINFORCE,
         CD_weight: weight of CD update (REINFORCE update -> (1 - CD_weight),

         Controller_Topology: weights of controller
         Controller_learning_rate: learning rate of perceptron,

         Evaluator_Topology: weights of evaluator
         Evaluator_learning_rate: learning rate of evaluator (critic)


        dataset: whole dataset fro training
        save_choice: save choice (Boolean)

        Graphic_on: graphical computations choice (Boolean)
        Prints_on: prints on the command line choice (Boolean)
        Save_Folders_Paths: folders for saves
        salient_feature = feature to focus (color, form, size or none),
        Tot_epocs: number of maximum epocs to train,



    return:

        Network_weights: trained weights of RBM. It is composed by net weights and two biases
        Executor_weights: trained weight of Perceptron. It is composed by net weights and the biases


    '''

    #Graphic_on = False

    # SAVE PATHS SETTING AND REMOVAL OF PREVIOUS SAVED FILES


    save_path_weights = Save_Folders_Paths[0]

    Saves_path = Save_Folders_Paths[1]

    Saves_path_training_visual_outputs = Save_Folders_Paths[2]


    # INIT VARIABLES

    Number_Single_Chunk_File = 0
    epoc_NOT_CHUNKED = 0
    epoc = 0
    step_graphic_save_functions = 1000

    Training_variables_Whole_System, \
    Network_weights_Training_variables_Sensory, \
    Network_weights_Controller, \
    Network_weights_Evaluator = WHOLE_SYSTEM_INITIALIZATION(

                                                                    Sensory_Topology,
                                                                    Controller_Topology,
                                                                    Evaluator_Topology

                                                            )

    Network_weights_Sensory = copy.deepcopy(Network_weights_Training_variables_Sensory[0:2])
    Training_variables_Sensory = copy.deepcopy(Network_weights_Training_variables_Sensory[2])

    Ideals_executor, \
    Ideals_executor_batch, \
    Legend_lab = Ideal_actions_initialization(

                                                Network_weights_Controller[0].shape[1],
                                                salient_feature
                                            )

    print('')
    print("Start Simulation... \n")
    Starting_time = datetime.datetime.today()


    # HERE I TRY TO SHUFFLE DATASET

    #dataset, Ideals_executor = Unison_shuffle(copy.deepcopy(dataset), copy.deepcopy(Ideals_executor))

    Rec_inputs_On_Hidden_Epocs = []
    Rec_inputs_On_Visible_Epocs = []
    Hidden_Activations_Epocs = []

    Rec_Errors_On_Hidden_Epocs_each_input = []
    Rec_Errors_On_Visible_Epocs_each_input = []

    Rec_Errors_On_Hidden_Epocs = []
    Rec_Errors_On_Visible_Epocs = []

    STD_Rec_Errors_On_Hidden_Epocs = []
    STD_Rec_Errors_On_Visible_Epocs = []

    Mean_W_Epocs = []
    Mean_W_b_1_Epocs = []
    Mean_W_b_2_Epocs = []

    Mean_W_Epocs_each_input = []
    Mean_W_b_1_Epocs_each_input = []
    Mean_W_b_2_Epocs_each_input = []

    Rewards_Epocs = []
    STD_Rewards_Epocs = []
    Rewards_Epocs_each_input = []

    Surprises_Epocs = []
    STD_Surprises_Epocs = []
    Surprises_Epocs_each_input = []

    Accuracies_Epocs = []
    STD_Accuracies_Epocs = []
    Accuracies_Epocs_each_input = []

    while epoc_NOT_CHUNKED < Tot_epocs:    # START EPOC LOOP %%%%%%%%%%%%%%%%%%

        Rec_Errors_steps = [[],[]]
        Rec_inputs_steps_On_Hidden = []
        Rec_inputs_steps_On_Visible = []

        Hidden_Activations_steps = []

        Weights_steps = []
        Weights_b_1_steps = []
        Weights_b_2_steps = []

        Weights_AVG_steps = []
        Weights_b_1_AVG_steps = []
        Weights_b_2_AVG_steps = []


        Rewards_steps = []
        Surprises_steps = []
        Accuracies_steps = []

        step = 0

        # START BATCH/INPUTS LOOP %%%%%%%%%%%%%%%%%%

        while step < dataset.shape[0]:

            Sensory_Component_output, \
            Controller_Component_output, \
            Evaluator_Component_output = WHOLE_SYSTEM_ACTIVATION(
                                                                    dataset[step],
                                                                    [Network_weights_Sensory, N_res_1, N_res_2, N_res_3],
                                                                    Network_weights_Controller,
                                                                    Network_weights_Evaluator

                                                                )

            Whole_system_performances, \
            Network_weights_Evaluator, \
            Network_weights_Sensory, \
            Network_weights_Controller = WHOLE_SYSTEM_TRAINING(
                                                                  Network_weights_Sensory,
                                                                  Sensory_Component_output,
                                                                  [CD_weight, learning_rate_CD, learning_rate_RL],

                                                                  Network_weights_Controller,
                                                                  Controller_Component_output,
                                                                  Controller_learning_rate,

                                                                  Network_weights_Evaluator,
                                                                  Evaluator_Component_output,
                                                                  Evaluator_learning_rate,

                                                                  Ideals_executor[step]
                                                             )





            # STOCASTIC/ONLINE DATA COLLECTION


            Training_variables_Sensory, \
            Training_variables_Whole_System = WHOLE_SYSTEM_TRAINING_VARIABLES_COMPUTATION_AND_COLLECTION(
                                                                                                         Training_variables_Sensory,
                                                                                                         Training_variables_Whole_System,
                                                                                                         Network_weights_Sensory,
                                                                                                         Sensory_Component_output,
                                                                                                         Whole_system_performances
                                                                                                         )

            # [Rec_inputs_on_hidden, Rec_inputs_on_visible,
            #  Hidden_Activations,
            #  Rec_Errors_on_hidden, Rec_Errors_on_visible,
            #  Weights_AVG, Weights_AVG_bias_to_hidden, Weights_AVG_bias_to_visible]
            #
            Rec_Errors_steps[0].append(Training_variables_Sensory[3][-1])
            Rec_Errors_steps[1].append(Training_variables_Sensory[4][-1])
            Rec_inputs_steps_On_Hidden.append(Training_variables_Sensory[0][-1])
            Rec_inputs_steps_On_Visible.append(Training_variables_Sensory[1][-1])

            Hidden_Activations_steps.append(Training_variables_Sensory[2][-1])

            # Weights_steps.append(Network_weights_Sensory[1][0])
            # Weights_b_1_steps.append(Network_weights_Sensory[1][1])
            # Weights_b_2_steps.append(Network_weights_Sensory[1][2])

            Weights_AVG_steps.append(np.mean(Network_weights_Sensory[1][0]))
            Weights_b_1_AVG_steps.append(np.mean(Network_weights_Sensory[1][1]))
            Weights_b_2_AVG_steps.append(np.mean(Network_weights_Sensory[1][2]))

            Rewards_steps.append(Whole_system_performances[0])
            Surprises_steps.append(Whole_system_performances[1])
            Accuracies_steps.append(Whole_system_performances[2])



            step += 1

        # STOP BATCH/INPUTS LOOP %%%%%%%%%%%%%%%%%%



        # DATA COLLECTION FOR EACH EPOC%%%%%%%%%%%%%%%%%%

        # STOCASTIC/ONLINE LEARNING

        # Rec_inputs_On_Hidden_Epocs.append(np.vstack(Rec_inputs_steps[0]))
        # Rec_inputs_On_Visible_Epocs.append(np.vstack(Rec_inputs_steps[1]))

        Hidden_Activations_steps_single_epoc = np.stack(Hidden_Activations_steps, axis= 1)[0]

        Hidden_Activations_Epocs.append(Hidden_Activations_steps_single_epoc)

        Rec_Errors_On_Hidden_Epocs_each_input.append(Rec_Errors_steps[0])
        Rec_Errors_On_Visible_Epocs_each_input.append(Rec_Errors_steps[1])

        Rec_Errors_On_Hidden_Epocs.append(np.mean(Rec_Errors_steps[0]))
        Rec_Errors_On_Visible_Epocs.append(np.mean(Rec_Errors_steps[1]))

        STD_Rec_Errors_On_Hidden_Epocs.append(np.std(Rec_Errors_steps[0]))
        STD_Rec_Errors_On_Visible_Epocs.append(np.std(Rec_Errors_steps[1]))


        Mean_W_Epocs.append(np.mean(Weights_AVG_steps))
        Mean_W_b_1_Epocs.append(np.mean(Weights_b_1_AVG_steps))
        Mean_W_b_2_Epocs.append(np.mean(Weights_b_2_AVG_steps))

        Mean_W_Epocs_each_input.append(Weights_AVG_steps)
        Mean_W_b_1_Epocs_each_input.append(Weights_b_1_AVG_steps)
        Mean_W_b_2_Epocs_each_input.append(Weights_b_2_AVG_steps)

        Rewards_Epocs.append(np.mean(Rewards_steps))
        STD_Rewards_Epocs.append(np.std(Rewards_steps))
        Rewards_Epocs_each_input.append(Rewards_steps)

        Surprises_Epocs.append(np.mean(Surprises_steps))
        STD_Surprises_Epocs.append(np.std(Surprises_steps))
        Surprises_Epocs_each_input.append(Surprises_steps)

        Accuracies_Epocs.append(np.mean(Accuracies_steps))
        STD_Accuracies_Epocs.append(np.std(Accuracies_steps))
        Accuracies_Epocs_each_input.append(Accuracies_steps)




        # PRINTS AND LEARNING RESULTS CALCULATION FOR EACH EPOC%%%%%%%%%%%%%%%%%%

        if epoc == 0:

                Max_R = np.around(Rewards_Epocs[epoc], decimals=5)
                sd_max_R = np.around(STD_Rewards_Epocs[epoc], decimals=5)
                Max_R_prev = copy.deepcopy(Rewards_Epocs[epoc])
                found_max_R = 0

                # COMPUTATION OF PERFORMANCE LEVEL FOR RENFORCEMENT LEARNING
                Span_Rec_error_R = 1
                min_step_error_to_100_R = Span_Rec_error_R / 100
                learning_percentual_progress_R = Max_R / min_step_error_to_100_R

                Min_error = np.around(Rec_Errors_On_Hidden_Epocs[epoc], decimals=5)
                sd_min_error = np.around(STD_Rec_Errors_On_Hidden_Epocs[epoc], decimals=5)
                min_error_prev = copy.deepcopy(Rec_Errors_On_Hidden_Epocs[epoc])
                found_min_error = 0

                Max_Acc = np.around(Accuracies_Epocs[epoc], decimals=5)
                sd_max_Acc = np.around(STD_Accuracies_Epocs[epoc], decimals=5)
                Max_Acc_prev = copy.deepcopy(Accuracies_Epocs[epoc])
                found_max_Acc = 0
                stability = 0

                # COMPUTATION OF PERFORMANCE LEVEL FOR ACCURACY
                Span_Rec_error_ACC = 1
                min_step_error_to_100_ACC = Span_Rec_error_ACC / 100
                learning_percentual_progress_ACC = Max_Acc / min_step_error_to_100_ACC



                if Max_Acc == 1:

                    stability += 1

                else:

                    stability = 0


                Max_stability = np.around(stability, decimals=5)


        else:


            Max_R = max(np.around(Rewards_Epocs[epoc], decimals=5), Max_R)

            if Max_R > Max_R_prev:

                found_max_R = epoc
                sd_max_R = np.around(STD_Rewards_Epocs[epoc], decimals=5)

            Max_R_prev = copy.deepcopy(Max_R)

            learning_percentual_progress_R = Max_R  / min_step_error_to_100_R



            Min_error = min(np.around(Rec_Errors_On_Hidden_Epocs[epoc], decimals=5), Min_error)

            if Min_error < min_error_prev:

                found_min_error = epoc
                sd_min_error= np.around(STD_Rec_Errors_On_Hidden_Epocs[epoc], decimals=5)

            min_error_prev = copy.deepcopy(Min_error)

            Max_Acc = max(np.around(Accuracies_Epocs[epoc], decimals=5), Max_Acc)

            learning_percentual_progress_ACC = Max_Acc  / min_step_error_to_100_ACC

            if Max_Acc > Max_Acc_prev:

                found_max_Acc = epoc
                sd_max_Acc = np.around(STD_Accuracies_Epocs[epoc], decimals=5)

            Max_Acc_prev = copy.deepcopy(Max_Acc)


            if Accuracies_Epocs[epoc] == 1:

                stability += 1

            else:

                stability = 0

            Max_stability = max(stability, Max_stability)


        if Prints_on:


            print('')
            print(str(" Epoc ") + str(epoc_NOT_CHUNKED) + str(' / ') + str(Tot_epocs) + str(' (') + str(np.around((epoc_NOT_CHUNKED/Tot_epocs) * 100, decimals = 2)) + str(' %)') + str(" ------------------------------ "))
            print('')

            print(str(' *** Performance achieved ') + str('(Acccuracy Target: 1, Reinf. target: 1): '))

            print('')

            print(str('        ') + str('+++ Reinforcement progress: ') + str(np.around(learning_percentual_progress_R, decimals = 2))
                + str(' % ') + str(' ///  Max Reinf. achieved: ') + str(np.around(Max_R, decimals = 4)) + str(' +- ') + str(sd_max_R)
                  + str(' at epoc ') + str(found_max_R) + str(' (Epocs without changes: ') + str(epoc - found_max_R)
                  + str(')'))
            print('')
            print(str('        ') + str('+++ Accuracy progress: ') + str(np.around(learning_percentual_progress_ACC, decimals=2)) + str(' % ')
                + str(' ///  Max Accuracy achieved: ') + str(np.around(Max_Acc, decimals=4)) + str(' +- ') + str(
                    sd_max_Acc)
                + str(' at epoc ') + str(found_max_Acc) + str(' (Epocs without changes: ') + str(epoc - found_max_Acc)
                  + str(')'))

            print('')
            print(str(" *** Epoc Reward = ") + str(np.around(Rewards_Epocs[epoc], decimals=4)) + str(" +- ") +
                  str(np.around(STD_Rewards_Epocs[epoc], decimals=4)))
            print('')
            print(str(' *** Epoc Surprise = ') + str(Surprises_Epocs[epoc]) + str(' +/- ') + str(
                STD_Surprises_Epocs[epoc]))
            print('')
            print(str(' *** Epoc Accuracy = ') + str(Accuracies_Epocs[epoc]) + str(' +/- ') + str(STD_Accuracies_Epocs[epoc]))
            print('')
            print(str(" *** Epoc Rec. Error (on hidden) = ") + str(np.around(Rec_Errors_On_Hidden_Epocs[epoc], decimals=4)) + str(" +- ") +
                  str(np.around(STD_Rec_Errors_On_Hidden_Epocs[epoc], decimals=4)))
            # print(str(' *** Stability = ') + str(stability) + str(' (Max achieved: ') + str(Max_stability) + str(')'))
            print('')


        epoc += 1
        epoc_NOT_CHUNKED += 1


        # SAVING FUNCTIONS%%%%%%%%%%%%%%%%%%

        if save_choice and epoc_NOT_CHUNKED != 0 and ((epoc_NOT_CHUNKED % step_graphic_save_functions) == 0):


            save_weights(Network_weights_Sensory[1][0], Network_weights_Sensory[1][1], Network_weights_Sensory[1][2], Hidden_Activations_Epocs[-1], save_path_weights)

            save_weights(Network_weights_Sensory[0][0], Network_weights_Sensory[0][1], Network_weights_Sensory[0][2], Hidden_Activations_Epocs[-1], save_path_weights)


            save_training_data_reinforced_CD_CHUNKED(Number_Single_Chunk_File, np.array(Rewards_Epocs_each_input),
                                                     np.array(Surprises_Epocs_each_input),
                                                     np.array(Accuracies_Epocs_each_input),
                                                     np.array(Rec_Errors_On_Hidden_Epocs_each_input),
                                                     Network_weights_Sensory[1][0],
                                                     np.array(Mean_W_Epocs_each_input),
                                                     np.array(Mean_W_b_1_Epocs_each_input),
                                                     np.array(Mean_W_b_2_Epocs_each_input),
                                                     learning_rate_CD,
                                                     Evaluator_learning_rate,
                                                     [0, Controller_Topology[1]],
                                                     [0,1],
                                                     Ideals_executor,
                                                     save_path=Saves_path)
            print('')
            print("...TRAINING DATA AND WEIGHTS SAVED!")
            print('')

            Number_Single_Chunk_File += 1


        # GRAPHICAL FUNCTIONS%%%%%%%%%%%%%%%%%%

        if Graphic_on and (epoc_NOT_CHUNKED != 0) and ((epoc_NOT_CHUNKED % step_graphic_save_functions) == 0):

            print('')
            print("...SAVING VISUAL OUTPUTS!")
            print('')

            # WEIGHTS DISTRIBUTIONS PLOTS

            Weights_FIRST_img = Windows_builder_network_weights(Network_weights_Sensory[0][0],
                                                                Network_weights_Sensory[0][1],
                                                                Network_weights_Sensory[0][2])

            Weights_SECOND_img = Windows_builder_network_weights(Network_weights_Sensory[1][0],
                                                                 Network_weights_Sensory[1][1],
                                                                 Network_weights_Sensory[1][2])

            Weights_FIRST_img.savefig(
                Saves_path_training_visual_outputs + str('Weights_FIRST_RBM.png'))

            plt.close()

            Weights_SECOND_img.savefig(
                Saves_path_training_visual_outputs + str('Weights_SECOND_RBM.png'))

            plt.close()

            # INPUTS RECONSTRUCTIONS%%%%%

            Rec_inputs_img, Rec_inputs = Window_builder_all_inputs_recostructions(np.array(Rec_inputs_steps_On_Visible))

                # np.array(Rec_inputs_steps_On_Hidden),
                #                                                                   Network_weights_Sensory[0][0],
                #                                                                   Network_weights_Sensory[0][2])

            Rec_inputs_img.savefig(
                Saves_path_training_visual_outputs + str('DBN_RECONSTRUCTED_INPUTS_NOT_FINISHED.png'))

            plt.close()

            Rec_error_for_inputs_DBN, Rec_error_batch_DBN, St_dev_error_batch_DBN = Reconstruction_errors(dataset,
                                                                                                          Rec_inputs)
            print('')
            print(str(" *** Error_Whole DBN = ") + str(Rec_error_batch_DBN) + str(" +- ") +
                  str(St_dev_error_batch_DBN))
            print('')

            # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

            Visible_CD_Pure, Explained_Var_Visible, Explained_Var_ratio_Visible = dim_reduction(dataset, 3)

            Visible_CD_Biased, Explained_Var_Visible_Biased, Explained_Var_ratio_Visible_Biased = dim_reduction(
                Rec_inputs, 3)

            # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

            # FIRST_HIDDEN_CD_Pure, Explained_Var_FIRST_HIDDEN_Pure, \
            # Explained_Var_ratio_FIRST_HIDDEN_Pure = dim_reduction(Rec_Epocs[-1], 3)

            FIRST_HIDDEN_CD_Biased, Explained_Var_FIRST_HIDDEN_Biased, Explained_Var_ratio_FIRST_HIDDEN_Biased = dim_reduction(
                Rec_inputs_steps_On_Hidden, 3)

            # DIMENSIONAL REDUCTIONS (SECOND HIDDEN LAYER)%%%%%

            # SECOND_HIDDEN_CD_Pure, Explained_Var_SECOND_HIDDEN_Pure, \
            # Explained_Var_ratio_SECOND_HIDDEN_Pure = dim_reduction(Hiddens_Activ_Epocs_each_input[-1], 3)

            SECOND_HIDDEN_CD_Biased, Explained_Var_SECOND_HIDDEN_Biased, Explained_Var_ratio_SECOND_HIDDEN_Biased = dim_reduction(
                Hidden_Activations_Epocs[-1], 3)

            # 3D PLOTS OF LAYER ACTIVATIONS%%%%%

            Visible_activation_disentanglement_img = Window_builder_layers_activations(
                Visible_CD_Biased,
                Ideals_executor,
                Ideals_executor_batch,
                Legend_lab,
                'input',
                2,
            )

            Visible_activation_disentanglement_img.savefig(
                Saves_path_training_visual_outputs + str('DISENTANGLEMENT_VISIBLE.png'))

            plt.close()

            First_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                FIRST_HIDDEN_CD_Biased,
                Ideals_executor,
                Ideals_executor_batch,
                Legend_lab,
                'FHL',
                2,
            )

            First_Hidden_activation_disentanglement_img.savefig(
                Saves_path_training_visual_outputs + str('DISENTANGLEMENT_FIRST_HIDDEN.png'))

            plt.close()

            Second_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                SECOND_HIDDEN_CD_Biased,
                Ideals_executor,
                Ideals_executor_batch,
                Legend_lab,
                'SHL',
                2,
            )

            Second_Hidden_activation_disentanglement_img.savefig(
                Saves_path_training_visual_outputs + str('DISENTANGLEMENT_SECOND_HIDDEN.png'))

            plt.close()



            Learning_rate, Learning_rate_critic, R_range, R_range_interp, ideal_actions, Reinforces_for_each_epoc, \
            Reinforces_for_each_epoc_STD, Surprises_for_each_epoc, Surprises_for_each_epoc_STD, Accuracies_for_each_epoc, Accuracies_for_each_epoc_STD, \
            Reconstruction_errors_for_each_epoc, Weights_network, Sum_weights_network_for_each_epoc, \
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc, Sum_bias_hiddens_to_inputs_weights_for_each_epoc, \
            Reinforces_for_each_input, Surprises_for_each_input, \
            Accuracies_for_each_input = load_training_data_reinforced_CD_JOIN([Network_weights_Sensory[1][0].shape[0],
                                                       Network_weights_Sensory[1][0].shape[1]],
                                                      Saves_path)

            Check_panel_img = Window_builder_check_panel_RBM(Reinforces_for_each_epoc, Reinforces_for_each_epoc_STD,
                                                                        Surprises_for_each_epoc,
                                                                        Surprises_for_each_epoc_STD,
                                                                        Reinforces_for_each_input,
                                                                        Surprises_for_each_input,
                                                                        Network_weights_Sensory[1][0],
                                                                        Sum_weights_network_for_each_epoc,
                                                                        Sum_bias_inputs_to_hiddens_weights_for_each_epoc,
                                                                        Sum_bias_hiddens_to_inputs_weights_for_each_epoc,
                                                                        Accuracies_for_each_epoc,
                                                                        Accuracies_for_each_epoc_STD,
                                                                        Accuracies_for_each_input,
                                                                        Reconstruction_errors_for_each_epoc,
                                                                        Learning_rate,
                                                                        Learning_rate_critic, R_range,
                                                                        R_range_interp,
                                                                        ideal_actions,
                                                                        CD_weight
                                                                        )




            Check_panel_img.savefig(Saves_path_training_visual_outputs + str('CHECK_PANEL.png'))

            plt.close()

            print('')
            print("...VISUAL OUTPUTS SAVED!")
            print('')
        # VARIABLES RESET/RE-INITIALIZATION%%%%%%%%%%%%%%%%%%

        if epoc_NOT_CHUNKED != 0 and (epoc_NOT_CHUNKED % step_graphic_save_functions) == 0:

            print('..RESET VARIABLES...')
            print('')




            # INIT VARIABLES

            _, _, Training_variables_Sensory = SENSORY_COMPONENT_INITIALIZATION(

                Sensory_Topology[0],
                Sensory_Topology[1],
            )

            Rec_inputs_On_Hidden_Epocs = []
            Rec_inputs_On_Visible_Epocs = []
            Hidden_Activations_Epocs = []

            Rec_Errors_On_Hidden_Epocs_each_input = []
            Rec_Errors_On_Visible_Epocs_each_input = []

            Rec_Errors_On_Hidden_Epocs = []
            Rec_Errors_On_Visible_Epocs = []

            STD_Rec_Errors_On_Hidden_Epocs = []
            STD_Rec_Errors_On_Visible_Epocs = []

            Mean_W_Epocs = []
            Mean_W_b_1_Epocs = []
            Mean_W_b_2_Epocs = []

            Mean_W_Epocs_each_input = []
            Mean_W_b_1_Epocs_each_input = []
            Mean_W_b_2_Epocs_each_input = []

            Rewards_Epocs = []
            STD_Rewards_Epocs = []
            Rewards_Epocs_each_input = []

            Surprises_Epocs = []
            STD_Surprises_Epocs = []
            Surprises_Epocs_each_input = []

            Accuracies_Epocs = []
            STD_Accuracies_Epocs = []
            Accuracies_Epocs_each_input = []

            epoc = 0


        # STOP LEARNING CONDITIONS %%%%%%%%%%%%%%%%%%


        #if stability == 10:

         #   break

    End_time = datetime.datetime.today()

    print('')
    print(' ------------------------------ ')
    print('')
    print(' ...End simulation (session_time =  ' + str(End_time - Starting_time) + str(')'))
    print('')
    print('There is a difference between knowing the path and walking the path.')
    print('')


    return Network_weights_Sensory[1], Network_weights_Controller

# UTILITY TEST DF DBN ACTIVATIONS


def Utility_test(Inputs, Tester_Learning_Rule, layer_label, salient_feature_tester, size_hidden_tester,
                 L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format,  Tester_data_saves_path, adversary=False):
    '''

        This macro-function executes a test of internal representations produced by a spread of RBM/DBN. These representations
        correspond to layers (visible, first hidden or second hidden) recostructions of whole DBN. The test consists of a perceptron
        that receives as inputs the representations and produces as output a localistic/sparse classification dependind on a specific
        feature that guided the training of RBM that produce the inputs. The tester (perceptron) is trained with Backpropagation (SL)
        or REINFORCE (RL, Williams, 1992) and the teach in input / winner vector (for RL) correspond to 4 "ideals vectors",
        one for each specific attribute of feature. E.g. in case of salient feature =  color and format = localistic we have four vectors:
        [1,0, 0, 0] for green, [0, 1, 0, 0] for red...

    args:

        - Inputs: dataset (64 inputs) of representations tested by this function.
        - Tester_Learning_Rule: leearning rule that i use to update the wieights of tester (REINFORCE OR BACKPROPAGATION)
        - Binary_Reward_format: format of reward computation (0/1 or continuois between 0 and 1)
        - salient_feature_tester: feature that guides the training of RBM that produces the inputs of this function.
        - layer_label: layer from which i take the specific internal representations (visible, first hidden or second hidden)
        - size_hidden_tester: output size of test
        - L_rate_tester: learning rate of tester
        - L_rate_tester_critic: learning rate of critic
        - tot_epocs_test: training epochs
        - Tester_data_saves_path: saves folder
        - adversary: ad hoc variable that I use to distinguish the results of tester that receives by a specific "adversary net"
                    (in this case a standard DBN) and the results of another tester that receive by a DBN that is trained wth
                    my modified R-CD (contrastive divergence influenced by reinforcement learning)

    return:

            none (saved file in folder)

    '''

    if adversary:


        net_label = 'NOT BIASED'


    else:

        net_label = 'BIASED'

    # Previously_saved_chunks_to_delete = glob.glob(Tester_data_saves_path + str('*'))
    #
    # for file in Previously_saved_chunks_to_delete:
    #
    #     os.remove(file)

    print('Start perceptrons match (' + str(layer_label) + str(', ') + str(net_label) + str(' DBN RECONSTRUCTIONS)'))

    if Tester_Learning_Rule == 'BACKPROP':

        Tester_weights = tester_weights_init(Inputs.shape[1], size_hidden_tester)

        Ideals_tester, Ideals_tester_batch, Legend_Lab = Ideal_actions_initialization(size_hidden_tester,
                                                                                      salient_feature_tester)

        Tester_Errors_Epocs = []

        for epoc_t in range(0, tot_epocs_test):
            Output_tester, Tester_weights, ERR = tester_spread_update(
                Inputs,
                Ideals_tester,
                Tester_weights,
                L_rate_tester)

            Tester_Errors_Epocs.append([ERR])

            print('Training tester (' + str(layer_label) + str(', ') + str(net_label) + str(')') + str('- Epoc'),
                  epoc_t,
                  str(' // - Err_ =')
                  + str(np.around(Tester_Errors_Epocs[epoc_t], decimals=3)))

        Tester_Errors_Epocs = np.array(Tester_Errors_Epocs)

        tester_save_data([salient_feature_tester, Inputs.shape[1], size_hidden_tester, L_rate_tester],
                         Tester_Errors_Epocs, False, False, adversary, Tester_data_saves_path)


    else:

        Tester_weights = Executor_init(Inputs.shape[1], size_hidden_tester)

        Critic_weights = Critic_init_MLP(Inputs.shape[1])

        Ideals_tester, Ideals_tester_batch, Legend_lab = Ideal_actions_initialization(size_hidden_tester,
                                                                                      salient_feature_tester)

        Tester_Rewards_Epocs = []
        Tester_Accuracies_Epocs = []

        for epoc_t in range(0, tot_epocs_test):

            batch_testers_Rewards = []
            batch_testers_Accuracies = []

            for obj in range(0, Inputs.shape[0]):
                ideal = Ideals_tester[obj, :]
                ideal = ideal.reshape((1, ideal.shape[0]))

                input = Inputs[obj, :]
                input = input.reshape((input.shape[0], 1))

                # input += np.random.uniform(-0.2, 0.2, (input.shape[0], input.shape[1]))  # noise, case

                # rec_test = input.reshape([28, 28, 3], order='F')
                #
                # plt.imshow(rec_test)
                #
                # plt.show()

                Output_sigm, Output_bin = Executor_spread(input, Tester_weights)

                R_Pred, Hidden_Activation_critic = Critic_spread_MLP(input, Critic_weights)

                executed, ideal, R_Real, R_Real_batch, surprise, surprise_batch, acc_single, acc_batch, \
                R_range, R_range_interp = Reinforce_processing(Output_bin, ideal, R_Pred, size_hidden_tester,
                                                               Binary_Reward_format)

                Critic_weights = Critic_update_MLP(input.T, Hidden_Activation_critic,
                                                   Critic_weights,
                                                   R_Pred, R_Real_batch, L_rate_tester_critic)

                Gradient, Weights_executor_update, Weights_bias_executor_update = Executor_Potential_update(input,
                                                                                                            Output_sigm,
                                                                                                            Output_bin,
                                                                                                            L_rate_tester,
                                                                                                            surprise)

                Tester_weights = Executor_Effective_update(Tester_weights[0],
                                                           Tester_weights[1],
                                                           Weights_executor_update,
                                                           Weights_bias_executor_update)

                batch_testers_Rewards.append([R_Real])
                batch_testers_Accuracies.append([acc_single])

            batch_testers_Rewards = np.array(batch_testers_Rewards)

            batch_testers_Rewards = np.mean(batch_testers_Rewards)

            Tester_Rewards_Epocs.append(batch_testers_Rewards)

            batch_testers_Accuracies = np.array(batch_testers_Accuracies)

            batch_testers_Accuracies = np.mean(batch_testers_Accuracies)

            Tester_Accuracies_Epocs.append(batch_testers_Accuracies)

            print('Training tester (' + str(layer_label) + str(', ') + str(net_label) + str(')') + str(' - Epoc'),
                  epoc_t,

                  str(' - R = ')
                  + str(np.around(Tester_Rewards_Epocs[epoc_t], decimals=3)) + str(
                      ' , Acc = ')
                  + str(np.around(Tester_Accuracies_Epocs[epoc_t], decimals=3)))

        Tester_Rewards_Epocs = np.array(Tester_Rewards_Epocs)

        Tester_Accuracies_Epocs = np.array(Tester_Accuracies_Epocs)

        tester_save_data(
            [salient_feature_tester, Inputs.shape[1], size_hidden_tester, L_rate_tester,
             L_rate_tester_critic],
            False, Tester_Rewards_Epocs, Tester_Accuracies_Epocs, adversary, Tester_data_saves_path)

        # Tester_img_VISIBLE = Window_builder_tester_performances_REINFORCE('visible', salient_feature_DBN,
        #                                                                   Tester_Rewards_Epocs_VISIBLE, Tester_Accuracies_Epocs_VISIBLE)

        # plt.show()

    print('End perceptrons match (' + str(layer_label) + str(', ') + str(net_label) + str(' DBN RECONSTRUCTIONS)'))

    print('Tester data saved (' + str(layer_label) + str(', ') + str(net_label) + str(')'))

