from Basic_Functions import *
import datetime
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')



# %%%%%%%%%%%%% MACRO FUNCTIONS TO ACTIVATE THE SYSTEM %%%%%%%%%%%%%

# ACTIVATION FUNCTION (RBM/DBN)

def RBM_DBN_Activation(Input_, number_of_resonances, Network_weights_First_RBM, salient_feature = 'none', number_of_resonances_2 = 0, Network_weights_Second_RBM = 0,
                       number_of_resonances_3 = 0):

    '''

    This is a macro-function that executes a spread and reconstruction of an input. Depending on arg 'Network_weights_Second_RBM'
    it actives the first RBM or the whole DBN (first RBM and second RBM).


    args:

    Input_: input of network
    number_of_resonances: number of Gibbs sampling steps ('resonances') from input to firs hidden
    number_of_resonances_2: number of Gibbs sampling steps ('resonances') fro first hidden to second hidden
    number_of_resonances_3: number of Gibbs sampling steps ('resonances') of whole DBN (input -> first hidden -> second hidden
                            -> first hidden -> input)
    Network_weights_First_RBM: topology of first RBM (weights)
    Network_weights_Second_RBM: topology of second RBM (weights)

    return:

        original_input_first_RBM: original input given to the net
        Hidden_activations_first_RBM: first activation of first hidden
        Inputs_reconstructions_first_RBM: reconstruction of input
        Rec_errors_first_RBM: reconstruction error of first RBM

        Hidden_activations_second_RBM (only in case of DBN activation): first activation of second hidden
        Inputs_reconstructions_second_RBM (only in case of DBN activation): reconstruction of first hidden from second hidden
        Rec_errors_second_RBM (only in case of DBN activation): reconstruction error of second RBM


    N.B. number_of_resonances and number_of_resonances_2 regulate the number of steps into a Gibbs sampling process.
         number_of_resonances refers to an activation from input to first hidden while number_of_resonances_2 refers to an
         activation from first hidden to second hidden.

    '''

    Hidden_activations_first_RBM = []
    Hidden_activations_second_RBM = []

    Inputs_reconstructions_first_RBM = []
    Inputs_reconstructions_second_RBM = []

    Rec_errors_first_RBM = []
    Rec_errors_second_RBM = []

    if number_of_resonances_3 == 0:

        # START RESONANCES/GIBBS SAMPLING OF FIRST RBM %%%%%%%%%%%%%%%%%%

        for res in range(0, number_of_resonances):

            if res == 0:

                if number_of_resonances > 1:

                    print('Start Gibbs sampling/resonances (RBM 1) ...')

                original_input_first_RBM = Get_input(Input_)

                input_first_RBM = copy.deepcopy(original_input_first_RBM)

                original_activation_hidden_first_RBM, _ = Activation_Hidden_Layer(input_first_RBM, Network_weights_First_RBM[0],
                                                                      Network_weights_First_RBM[1], stochasticity = False)


                Activation_hidden_first_RBM = copy.deepcopy(original_activation_hidden_first_RBM)

                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))


            else:

                Activation_hidden_first_RBM, _ = Activation_Hidden_Layer(input_first_RBM, Network_weights_First_RBM[0],
                                                                            Network_weights_First_RBM[1], stochasticity = False)



                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))

            if number_of_resonances > 1:

                print('RBM_1 - Resonance n. ', res)


            rec_input_first_RBM = Input_reconstruction(Activation_hidden_first_RBM, Network_weights_First_RBM[0],
                                                       Network_weights_First_RBM[2])

            Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM = Reconstruction_errors(
                original_input_first_RBM, rec_input_first_RBM)

            Rec_errors_first_RBM.append([Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM])


            Inputs_reconstructions_first_RBM.append(copy.deepcopy(rec_input_first_RBM))

            input_first_RBM = rec_input_first_RBM

        if number_of_resonances > 1:

            print('...Stop Gibbs sampling/resonances (RBM 1)')

        print(' Wake up Neo The Matrix has you… ')

        # STOP RESONANCES/GIBBS SAMPLING OF FIRST RBM %%%%%%%%%%%%%%%%%%

        if Network_weights_Second_RBM != 0:

            # START RESONANCES/GIBBS SAMPLING OF SECOND RBM %%%%%%%%%%%%%%%%%%

            for res_2 in range(0, number_of_resonances_2):


                if res_2 == 0:

                    if number_of_resonances_2 > 1:

                        print('Start Gibbs sampling/resonances (RBM 2) ...')

                    original_input_second_RBM = Get_input(Activation_hidden_first_RBM)

                    original_activation_hidden_second_RBM, _ = Activation_Hidden_Layer(original_input_second_RBM, Network_weights_Second_RBM[0],
                                                                           Network_weights_Second_RBM[1],
                                                                           stochasticity=False)


                    Activation_hidden_second_RBM = copy.deepcopy(original_activation_hidden_second_RBM)

                    Hidden_activations_second_RBM.append(copy.deepcopy(Activation_hidden_second_RBM))

                else:

                    input_second_RBM = Get_input(Activation_hidden_first_RBM)

                    Activation_hidden_second_RBM, _ = Activation_Hidden_Layer(input_second_RBM, Network_weights_Second_RBM[0],
                                                                                 Network_weights_Second_RBM[1], stochasticity = False)



                    Hidden_activations_second_RBM.append(copy.deepcopy(Activation_hidden_second_RBM))

                if number_of_resonances_2 > 1:

                    print('RBM_2 - Resonance n. ', res_2)

                rec_input_second_RBM = Input_reconstruction(Activation_hidden_second_RBM, Network_weights_Second_RBM[0], Network_weights_Second_RBM[2])

                Rec_error_for_inputs_second_RBM, Rec_error_batch_second_RBM, St_dev_error_batch_second_RBM = Reconstruction_errors(
                    original_input_second_RBM, rec_input_second_RBM)

                Rec_errors_second_RBM.append([Rec_error_for_inputs_second_RBM, Rec_error_batch_second_RBM, St_dev_error_batch_second_RBM])

                Inputs_reconstructions_second_RBM.append(copy.deepcopy(rec_input_second_RBM))

                Activation_hidden_first_RBM = rec_input_second_RBM

            if number_of_resonances_2 > 1:

                print('...Stop Gibbs sampling/resonances (RBM 2)')

            rec_input_first_RBM = Input_reconstruction(Activation_hidden_first_RBM, Network_weights_First_RBM[0], Network_weights_First_RBM[2])

            Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM = Reconstruction_errors(
                original_input_first_RBM, rec_input_first_RBM)

            Rec_errors_first_RBM.append([Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM])

            Inputs_reconstructions_first_RBM.append(copy.deepcopy(rec_input_first_RBM))

            input_first_RBM = rec_input_first_RBM

        # STOP RESONANCES/GIBBS SAMPLING OF SECOND RBM %%%%%%%%%%%%%%%%%%
    else:

        # START RESONANCES/GIBBS SAMPLING OF WHOLE DBN %%%%%%%%%%%%%%%%%%

        print('Start Gibbs sampling/resonances (WHOLE DBN) ...')

        for res_3 in range(0, number_of_resonances_3):

            print('Whole_DBN - Resonance n. ', res_3)

            if res_3 == 0:

                original_input_first_RBM = Get_input(Input_)

                input_first_RBM = copy.deepcopy(original_input_first_RBM)


                original_activation_hidden_first_RBM, _ = Activation_Hidden_Layer(input_first_RBM,
                                                                               Network_weights_First_RBM[0],
                                                                               Network_weights_First_RBM[1],
                                                                               stochasticity=False)

                Activation_hidden_first_RBM = copy.deepcopy(original_activation_hidden_first_RBM)

                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))

                original_input_second_RBM = Get_input(Activation_hidden_first_RBM)

                original_activation_hidden_second_RBM, _ = Activation_Hidden_Layer(original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity=False)
                if salient_feature != 'none':

                    if original_activation_hidden_second_RBM.shape[1] == 12 or \
                            original_activation_hidden_second_RBM.shape[
                                1] == 4:

                        Activation_hidden_second_RBM = Top_Down_Manipulation(original_activation_hidden_second_RBM,
                                                                             salient_feature)

                Activation_hidden_second_RBM = copy.deepcopy(original_activation_hidden_second_RBM)

                Hidden_activations_second_RBM.append(copy.deepcopy(Activation_hidden_second_RBM))



            else:

                Activation_hidden_first_RBM, _ = Activation_Hidden_Layer(input_first_RBM,
                                                                               Network_weights_First_RBM[0],
                                                                               Network_weights_First_RBM[1],
                                                                               stochasticity=False)

                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))

                original_input_second_RBM = Get_input(Activation_hidden_first_RBM)

                original_activation_hidden_second_RBM, _ = Activation_Hidden_Layer(original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity=False)


                if salient_feature != 'none':

                    if Activation_hidden_second_RBM.shape[1] == 12 or \
                            Activation_hidden_second_RBM.shape[
                                1] == 4:


                        Activation_hidden_second_RBM = Top_Down_Manipulation(Activation_hidden_second_RBM,
                                                                                  salient_feature)

                Hidden_activations_second_RBM.append(Activation_hidden_second_RBM)

            rec_input_second_RBM = Input_reconstruction(Activation_hidden_second_RBM, Network_weights_Second_RBM[0],
                                                        Network_weights_Second_RBM[2])

            Rec_error_for_inputs_second_RBM, Rec_error_batch_second_RBM, St_dev_error_batch_second_RBM = Reconstruction_errors(
                original_input_second_RBM, rec_input_second_RBM)

            Rec_errors_second_RBM.append([Rec_error_for_inputs_second_RBM, Rec_error_batch_second_RBM, St_dev_error_batch_second_RBM])

            Inputs_reconstructions_second_RBM.append(rec_input_second_RBM)

            Activation_hidden_first_RBM = rec_input_second_RBM

            rec_input_first_RBM = Input_reconstruction(Activation_hidden_first_RBM, Network_weights_First_RBM[0],
                                                       Network_weights_First_RBM[2])

            Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM = Reconstruction_errors(
                original_input_first_RBM, rec_input_first_RBM)

            Rec_errors_first_RBM.append([Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM])

            Inputs_reconstructions_first_RBM.append(rec_input_first_RBM)

            input_first_RBM = rec_input_first_RBM

        print('...Stop Gibbs sampling/resonances (WHOLE DBN)')

        # STOP RESONANCES/GIBBS SAMPLING OF WHOLE DBN %%%%%%%%%%%%%%%%%%



    # RETURNS

    if Network_weights_Second_RBM == 0:

        return original_input_first_RBM, Hidden_activations_first_RBM, Inputs_reconstructions_first_RBM, Rec_errors_first_RBM

    else:

        return original_input_first_RBM, Hidden_activations_first_RBM, Hidden_activations_second_RBM, \
               Inputs_reconstructions_first_RBM, Inputs_reconstructions_second_RBM, Rec_errors_first_RBM, Rec_errors_second_RBM


# %%%%%%%%%%%%% MACRO FUNCTIONS FOR ROBOTIC SIMULATION %%%%%%%%%%%%%

# SENSORY COMPONENT -----------------

# INITIALIZATION

def SENSORY_COMPONENT_INITIALIZATION(

                                        Topology_RBM_1,
                                        Topology_RBM_2,

                                    ):

    '''

    This function initializes the DBN, loading the pretrained weights of first RBM (the one that receives the kuka camera input), and
    randomly inizializing the weights of second RBM. Moreover, it initializes the "conteitors variables" and counts for the training.

    args:

        Topology_RBM_1: number of visible and first hidden units of DBN (visible/hidden of RBM 1)
        Topology_RBM_2: number of first hidden and second hidden units of DBN (visible/hidden of RBM 2)



    return: Weights_RBM_1: weights of RBM 1
            Weights_bias_to_hidden_RBM_1: weight of bias (to hidden) of RBM 1
            Weights_bias_to_visible_RBM_1: weight of bias (to visible) of RBM 1
            Weights_RBM_2: weights of RBM 2
            Weights_bias_to_hidden_RBM_2: weight of bias (to hidden) of RBM 2
            Weights_bias_to_visible_RBM_2:weight of bias (to visible) of RBM 2
            Training_Check_Performance_variables: many training variables (e.g. contenitors, counts etc)

    '''

    # INIZIALIZATION OF DBN WEIGHTS

    Weights_RBM_1, \
    Weights_bias_to_hidden_RBM_1, \
    Weights_bias_to_visible_RBM_1 = Load_weights(
                                                        Topology_RBM_1[0],
                                                        Topology_RBM_1[1],
                                                        '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'
                                                )

    Weights_RBM_2, \
    Weights_bias_to_hidden_RBM_2, \
    Weights_bias_to_visible_RBM_2 = Random_initialization_weights(
                                                                    Topology_RBM_2[0],
                                                                    Topology_RBM_2[1]
                                                                 )

    # INIT TRAINING/PERFORMANCES VARIABLES

    Training_Check_Performance_variables = Initialize_variabiles_RBM_K()


    return [Weights_RBM_1, Weights_bias_to_hidden_RBM_1, Weights_bias_to_visible_RBM_1], \
            [Weights_RBM_2, Weights_bias_to_hidden_RBM_2, Weights_bias_to_visible_RBM_2], \
            Training_Check_Performance_variables



# ACTIVATION

def SENSORY_COMPONENT_ACTIVATION(
                                 Input_,
                                 Network_weights_First_RBM, Network_weights_Second_RBM,
                                 number_of_resonances_1 = 2, number_of_resonances_2 = 2, number_of_resonances_3  = 0,
                                ):

    '''

    This is a macro-function that allows a Deep Belief Network (DBN)/Deep Restricted Boltmann Machine (DRBM) to execute a spread and a reconstruction of an input.


    args:

    Input_: input of network
    number_of_resonances_1: number of Gibbs sampling steps ('resonances') from input to firs hidden
    number_of_resonances_2: number of Gibbs sampling steps ('resonances') fro first hidden to second hidden
    number_of_resonances_3: number of Gibbs sampling steps ('resonances') of whole DBN (input -> first hidden -> second hidden
                            -> first hidden -> input)
    Network_weights_First_RBM: topology of first RBM (weights)
    Network_weights_Second_RBM: topology of second RBM (weights)

    return:

        original_input_first_RBM: original input given to the net
        Hidden_activations_first_RBM: first activation of first hidden
        Inputs_reconstructions_first_RBM: reconstruction of input
        Rec_errors_first_RBM: reconstruction error of first RBM

        Hidden_activations_second_RBM (only in case of DBN activation): first activation of second hidden
        Inputs_reconstructions_second_RBM (only in case of DBN activation): reconstruction of first hidden from second hidden
        Rec_errors_second_RBM (only in case of DBN activation): reconstruction error of second RBM


    '''

    Hidden_activations_first_RBM = []
    Hidden_activations_second_RBM = []

    Inputs_reconstructions_first_RBM = []
    Inputs_reconstructions_second_RBM = []

    Rec_errors_first_RBM = []
    Rec_errors_second_RBM = []

    if number_of_resonances_3 == 0:

        # START RESONANCES/GIBBS SAMPLING OF FIRST RBM %%%%%%%%%%%%%%%%%%

        for res in range(0, number_of_resonances_1):

            if res == 0:

                # if number_of_resonances_1 > 1:

                    #print('Start Gibbs sampling/resonances (RBM 1) ...')

                original_input_first_RBM = Get_input(Input_)

                input_first_RBM = copy.deepcopy(original_input_first_RBM)

                original_activation_hidden_first_RBM_Probabilities, _ = Activation_Hidden_Layer(
                                                                                  input_first_RBM,
                                                                                  Network_weights_First_RBM[0],
                                                                                  Network_weights_First_RBM[1],
                                                                                  stochasticity = False
                                                                                  )

                original_activation_hidden_first_RBM, penalty = Activation_Hidden_Layer(
                                                                                  input_first_RBM,
                                                                                  Network_weights_First_RBM[0],
                                                                                  Network_weights_First_RBM[1],
                                                                                  stochasticity = True
                                                                                  )


                Activation_hidden_first_RBM = copy.deepcopy(original_activation_hidden_first_RBM)

                Hidden_activations_first_RBM.append([original_activation_hidden_first_RBM_Probabilities, copy.deepcopy(Activation_hidden_first_RBM)])


            else:

                Activation_hidden_first_RBM, _ = Activation_Hidden_Layer(
                                                                         input_first_RBM,
                                                                         Network_weights_First_RBM[0],
                                                                         Network_weights_First_RBM[1],
                                                                         stochasticity = False
                                                                         )


                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))

            # if number_of_resonances_1 > 1:
            #
            #     print('RBM_1 - Resonance n. ', res)


            rec_input_first_RBM = Input_reconstruction(
                                                       Activation_hidden_first_RBM,
                                                       Network_weights_First_RBM[0],
                                                       Network_weights_First_RBM[2]
                                                      )

            Inputs_reconstructions_first_RBM.append(copy.deepcopy(rec_input_first_RBM))

            input_first_RBM = rec_input_first_RBM


            Activation_hidden_first_RBM, _ = Activation_Hidden_Layer(
                                                                        input_first_RBM,
                                                                        Network_weights_First_RBM[0],
                                                                        Network_weights_First_RBM[1],
                                                                        stochasticity=False
                                                                    )

            Rec_error_absolute_first_RBM, \
            Rec_error_percent_first_RBM = Reconstruction_errors_K(
                                                       original_input_first_RBM,
                                                       rec_input_first_RBM
                                                      )

            Rec_errors_first_RBM.append([Rec_error_absolute_first_RBM, Rec_error_percent_first_RBM])
            Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))


        # if number_of_resonances_1 > 1:
        #
        #     print('...Stop Gibbs sampling/resonances (RBM 1)')

        #print(' Wake up Neo The Matrix has you… ')

        # STOP RESONANCES/GIBBS SAMPLING OF FIRST RBM %%%%%%%%%%%%%%%%%%

        # START RESONANCES/GIBBS SAMPLING OF SECOND RBM %%%%%%%%%%%%%%%%%%

        for res_2 in range(0, number_of_resonances_2):


            if res_2 == 0:

                # if number_of_resonances_2 > 1:
                #
                #     print('Start Gibbs sampling/resonances (RBM 2) ...')

                original_input_second_RBM = Get_input(original_activation_hidden_first_RBM)

                original_activation_hidden_second_RBM_Probabilities, _ = Activation_Hidden_Layer(
                                                                                   original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity = False
                                                                                   )

                original_activation_hidden_second_RBM, penalty_2 = Activation_Hidden_Layer(
                                                                                   original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity = True
                                                                                   )


                Activation_hidden_second_RBM = copy.deepcopy(original_activation_hidden_second_RBM)

                Hidden_activations_second_RBM.append([original_activation_hidden_second_RBM_Probabilities, copy.deepcopy(Activation_hidden_second_RBM)])

            else:

                input_second_RBM = Get_input(Activation_hidden_first_RBM)

                Activation_hidden_second_RBM, _ = Activation_Hidden_Layer(
                                                                            input_second_RBM, Network_weights_Second_RBM[0],
                                                                            Network_weights_Second_RBM[1],
                                                                            stochasticity = False
                                                                          )

                Hidden_activations_second_RBM.append(copy.deepcopy(Activation_hidden_second_RBM))

            # if number_of_resonances_2 > 1:
            #
            #     print('RBM_2 - Resonance n. ', res_2)

            rec_input_second_RBM = Input_reconstruction(
                                                        Activation_hidden_second_RBM,
                                                        Network_weights_Second_RBM[0],
                                                        Network_weights_Second_RBM[2]
                                                        )

            Activation_hidden_first_RBM = rec_input_second_RBM

            input_second_RBM = Get_input(Activation_hidden_first_RBM)

            Activation_hidden_second_RBM, _ = Activation_Hidden_Layer(
                                                                        input_second_RBM,
                                                                        Network_weights_Second_RBM[0],
                                                                        Network_weights_Second_RBM[1],
                                                                        stochasticity=False
                                                                    )


            Rec_error_absolute_second_RBM, \
            Rec_error_percent_second_RBM  = Reconstruction_errors_K(
                                                                        original_input_second_RBM,
                                                                        rec_input_second_RBM
                                                                     )

            Rec_errors_second_RBM.append([Rec_error_absolute_second_RBM, Rec_error_percent_second_RBM])

            Inputs_reconstructions_second_RBM.append(copy.deepcopy(rec_input_second_RBM))

            Hidden_activations_second_RBM.append(copy.deepcopy(Activation_hidden_second_RBM))



        # if number_of_resonances_2 > 1:
        #
        #     print('...Stop Gibbs sampling/resonances (RBM 2)')

        rec_input_first_RBM = Input_reconstruction(
                                                   Activation_hidden_first_RBM,
                                                   Network_weights_First_RBM[0],
                                                   Network_weights_First_RBM[2]
                                                  )

        Rec_error_absolute_first_RBM, \
        Rec_error_percent_first_RBM = Reconstruction_errors_K(
                                                             original_input_first_RBM,
                                                             rec_input_first_RBM
                                                            )

        Rec_errors_first_RBM.append([Rec_error_absolute_first_RBM, Rec_error_percent_first_RBM])

        Inputs_reconstructions_first_RBM.append(copy.deepcopy(rec_input_first_RBM))

        input_first_RBM = rec_input_first_RBM

        # STOP RESONANCES/GIBBS SAMPLING OF SECOND RBM %%%%%%%%%%%%%%%%%%

    else:

        # START RESONANCES/GIBBS SAMPLING OF WHOLE DBN %%%%%%%%%%%%%%%%%%

        #print('Start Gibbs sampling/resonances (WHOLE DBN) ...')

        for res_3 in range(0, number_of_resonances_3):

            #print('Whole_DBN - Resonance n. ', res_3)

            if res_3 == 0:

                original_input_first_RBM = Get_input(Input_)

                input_first_RBM = copy.deepcopy(original_input_first_RBM)

                original_activation_hidden_first_RBM_Probabilities, _ = Activation_Hidden_Layer(
                                                                                    input_first_RBM,
                                                                                    Network_weights_First_RBM[0],
                                                                                    Network_weights_First_RBM[1],
                                                                                    stochasticity=False
                                                                                )


                original_activation_hidden_first_RBM, penalty = Activation_Hidden_Layer(
                                                                               input_first_RBM,
                                                                               Network_weights_First_RBM[0],
                                                                               Network_weights_First_RBM[1],
                                                                               stochasticity=True
                                                                                )

                Activation_hidden_first_RBM = copy.deepcopy(original_activation_hidden_first_RBM)

                Hidden_activations_first_RBM.append([original_activation_hidden_first_RBM_Probabilities, copy.deepcopy(Activation_hidden_first_RBM)])

                original_input_second_RBM = Get_input(Activation_hidden_first_RBM)

                original_activation_hidden_second_RBM_Probabilities, _ = Activation_Hidden_Layer(
                                                                                    original_input_second_RBM,
                                                                                    Network_weights_Second_RBM[0],
                                                                                    Network_weights_Second_RBM[1],
                                                                                    stochasticity=False
                                                                                    )

                original_activation_hidden_second_RBM, penalty_2 = Activation_Hidden_Layer(
                                                                                   original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity = True
                                                                                   )

                Activation_hidden_second_RBM = copy.deepcopy(original_activation_hidden_second_RBM)

                Hidden_activations_second_RBM.append([original_activation_hidden_second_RBM_Probabilities, copy.deepcopy(Activation_hidden_second_RBM)])



            else:

                Activation_hidden_first_RBM, _ = Activation_Hidden_Layer(
                                                                        input_first_RBM,
                                                                        Network_weights_First_RBM[0],
                                                                        Network_weights_First_RBM[1],
                                                                        stochasticity=False
                                                                        )

                Hidden_activations_first_RBM.append(copy.deepcopy(Activation_hidden_first_RBM))

                original_input_second_RBM = Get_input(Activation_hidden_first_RBM)

                original_activation_hidden_second_RBM, _ = Activation_Hidden_Layer(
                                                                                   original_input_second_RBM,
                                                                                   Network_weights_Second_RBM[0],
                                                                                   Network_weights_Second_RBM[1],
                                                                                   stochasticity=False
                                                                                    )



                Hidden_activations_second_RBM.append(Activation_hidden_second_RBM)

            rec_input_second_RBM = Input_reconstruction(Activation_hidden_second_RBM, Network_weights_Second_RBM[0],
                                                        Network_weights_Second_RBM[2])

            Rec_error_for_inputs_second_RBM, \
            Rec_error_batch_second_RBM, \
            St_dev_error_batch_second_RBM = Reconstruction_errors(
                                                                  original_input_second_RBM,
                                                                  rec_input_second_RBM)

            Rec_errors_second_RBM.append([Rec_error_for_inputs_second_RBM, Rec_error_batch_second_RBM, St_dev_error_batch_second_RBM])

            Inputs_reconstructions_second_RBM.append(rec_input_second_RBM)

            Activation_hidden_first_RBM = rec_input_second_RBM

            rec_input_first_RBM = Input_reconstruction(
                                                       Activation_hidden_first_RBM,
                                                       Network_weights_First_RBM[0],
                                                       Network_weights_First_RBM[2]
                                                       )

            Rec_error_for_inputs_first_RBM, \
            Rec_error_batch_first_RBM, \
            St_dev_error_batch_first_RBM = Reconstruction_errors(
                                                                original_input_first_RBM,
                                                                rec_input_first_RBM
                                                                )

            Rec_errors_first_RBM.append([Rec_error_for_inputs_first_RBM, Rec_error_batch_first_RBM, St_dev_error_batch_first_RBM])

            Inputs_reconstructions_first_RBM.append(rec_input_first_RBM)

            input_first_RBM = rec_input_first_RBM

        #print('...Stop Gibbs sampling/resonances (WHOLE DBN)')

        # STOP RESONANCES/GIBBS SAMPLING OF WHOLE DBN %%%%%%%%%%%%%%%%%%



    # RETURNS

    return [original_input_first_RBM, Hidden_activations_first_RBM, Inputs_reconstructions_first_RBM, \
           Hidden_activations_second_RBM, Inputs_reconstructions_second_RBM, \
           Rec_errors_first_RBM, Rec_errors_second_RBM, penalty_2]

# TRAINING

def SENSORY_COMPONENT_TRAINING(
                               Training_hyperparameters,
                               Network_weights,
                               Sensory_Component_output,
                               Surprise
                                ):

    '''

    This is a macro-function that allows a single training step of a DBN/DRBM.


    args:

        Training_hyperparameters: hyperparameters of DBN/DRBM training (CD_weight, learning_rate_CD, learning_rate_RL)
        Network_weights: Weights of DBN
        Sensory_Component_output: output of DBN/DRBM activation (format =   [original_input_first_RBM, Hidden_activations_first_RBM,
                                                                            Inputs_reconstructions_first_RBM, Hidden_activations_second_RBM,
                                                                            Inputs_reconstructions_second_RBM, Rec_errors_first_RBM, Rec_errors_second_RBM, penalty])
        Surprise: Surprise computed by Goal-monitoring component (L1 of Goal_joints and Kuka_joints, interpolated in a [0, 1] continuous range



    return:

        Network_weights: Updated weights of DBN



    '''

    RBM_2_Input_Bin = Sensory_Component_output[1][0][1]
    RBM_2_Input_Sigm = Sensory_Component_output[1][0][0]

    RBM_2_Output_Bin = Sensory_Component_output[3][0][1]
    RBM_2_Output_Sigm_first = Sensory_Component_output[3][0][0]
    RBM_2_Output_Sigm_last = Sensory_Component_output[3][-1]

    RBM_2_Rec_first = Sensory_Component_output[4][0]
    RBM_2_Rec_last = Sensory_Component_output[4][-1]
    RBM_2_Penalty = Sensory_Component_output[7]


    # POTENTIAL UPDATE CD

    Weight_inputs_update_CD, \
    Weights_bias_inputs_to_hiddens_update_CD, \
    Weights_bias_hiddens_to_inputs_update_CD = Potential_update_CD(
                                                                    RBM_2_Input_Bin,
                                                                    RBM_2_Output_Bin,
                                                                    RBM_2_Rec_last,
                                                                    RBM_2_Output_Sigm_last,
                                                                    Training_hyperparameters[1],
                                                                    penalty = RBM_2_Penalty
                                                                 )

    # POTENTIAL UPDATE RL

    Weight_inputs_update_RL, \
    Weights_bias_inputs_to_hiddens_update_RL = Potential_update_Reinforcement(
                                                                               RBM_2_Input_Bin,
                                                                               RBM_2_Output_Sigm_first,
                                                                               RBM_2_Output_Bin,
                                                                               Training_hyperparameters[2],
                                                                               Surprise
                                                                              )

    # HIBRIDATION OF CD AND RL

    Weight_inputs_update, \
    Weights_bias_inputs_to_hiddens_update, \
    Weights_bias_hiddens_to_inputs_update = Potential_update_hibridation_CD_Reinforcement(
                                                                                          Weight_inputs_update_CD,
                                                                                          Weights_bias_inputs_to_hiddens_update_CD,
                                                                                          Weights_bias_hiddens_to_inputs_update_CD,
                                                                                          Weight_inputs_update_RL,
                                                                                          Weights_bias_inputs_to_hiddens_update_RL,
                                                                                          Training_hyperparameters[0]
                                                                                         )

    # NETWORK UPDATE

    Network_weights[1][0], \
    Network_weights[1][1], \
    Network_weights[1][2], _, _, _ = Effective_update(
                                            Network_weights[1][0],
                                            Network_weights[1][1],
                                            Network_weights[1][2],
                                            Weight_inputs_update,
                                            Weights_bias_inputs_to_hiddens_update.T,
                                            Weights_bias_hiddens_to_inputs_update.T
                          )


    return Network_weights

def SENSORY_COMPONENT_TRAINING_VARIABLES_COMPUTATION_AND_COLLECTION(

                                                                    Training_Check_Performance_variables,
                                                                    Network_weights,
                                                                    Sensory_Component_output


                                                                    ):

    '''

    This function allows the collection of training indice of Sensory component.

    args:

            Training_Check_Performance_variables: training_variables_DBN(format = [Rec_inputs_on_hidden, Rec_inputs_on_visible,
                                                         Hidden_Activations,
                                                         Rec_Errors_on_hidden, Rec_Errors_on_visible,
                                                         Weights_AVG, Weights_AVG_bias_to_hidden, Weights_AVG_bias_to_visible])
            Network_weights: networks of RBM_2
            Sensory_Component_output: output of Sensory component (format =   [original_input_first_RBM, Hidden_activations_first_RBM,
                                                                            Inputs_reconstructions_first_RBM, Hidden_activations_second_RBM,
                                                                            Inputs_reconstructions_second_RBM, Rec_errors_first_RBM, Rec_errors_second_RBM, penalty])

    return:

            Training_Check_Performance_variables: update training_variables_DBN

    '''

    Rec_error_on_hidden, Rec_error_percent_on_hidden = Reconstruction_errors_K(

                                                                                    Sensory_Component_output[1][0][1],
                                                                                    Sensory_Component_output[4][0]
                                                                              )

    Rec_error_on_visible, Rec_error_percent_on_visible = Reconstruction_errors_K(

                                                                                        Sensory_Component_output[0],
                                                                                        Sensory_Component_output[2][1]
                                                                                )


    AVG_Weights = [np.mean(Network_weights[1][0]), np.mean(Network_weights[1][1]), np.mean(Network_weights[1][2])]

    Training_Check_Performance_variables[0].append(Sensory_Component_output[4][0]) # Inputs_reconstructions_second_RBM
    Training_Check_Performance_variables[1].append(Sensory_Component_output[2][-1]) # Inputs_reconstructions_first_RBM
    Training_Check_Performance_variables[2].append(Sensory_Component_output[3][0][1]) # Hidden_activations_second_RBM
    Training_Check_Performance_variables[3].append(Rec_error_on_hidden) # Rec_errors_second_RBM
    Training_Check_Performance_variables[4].append(Rec_error_on_visible) # Rec_errors_first_RBM
    Training_Check_Performance_variables[5].append(AVG_Weights[0]) # AVG weights
    Training_Check_Performance_variables[6].append(AVG_Weights[1]) # AVG weights (b1)
    Training_Check_Performance_variables[7].append(AVG_Weights[2]) # # AVG weights (b2)

    return Training_Check_Performance_variables



# CONTROLLER COMPONENT -----------------

# INITIALIZATION

def CONTROLLER_COMPONENT_INITIALIZATION(
                                        Topology_Controller
                                        ):

    '''

    This function initializes the controller network (simple perceptron).

    Args:

            Topology_Controller: number of Input/Output units of Perceptron

    return:

            Controller_weights: weights of Controller

    '''

    Controller_weights = Executor_init(Topology_Controller[0], Topology_Controller[1])

    return Controller_weights

# ACTIVATION

def CONTROLLER_COMPONENT_ACTIVATON(
                                   input_,
                                   weights_controller
                                  ):


    '''
        This function execute a stochastic spread of a simple network.

        args:

            - input_: input of NET (hidden representation of RBM)
            - weights_critic: weights list of NET (input -> output, bias)

        return:

            - output_sigm: sigmoidal output of vector

            - output_bin: binary outoput vector of net (0/1)

    '''

    Output_pot = np.dot(input_, weights_controller[0]) + weights_controller[1]

    Output_sigm = 1 / (1 + np.exp(-(Output_pot)))

    Output_bin = copy.deepcopy(Output_sigm)

    random_treshold = np.random.random_sample((Output_bin.shape[0], Output_bin.shape[1]))  # noise, case


    Output_bin[Output_bin > random_treshold] = 1

    Output_bin[Output_bin < random_treshold] = 0

    return [Output_sigm, Output_bin]

# TRAINING

def CONTROLLER_COMPONENT_TRAINING(
                                    Sensory_Component_output,
                                    Training_hyperparameters,
                                    Network_weights,
                                    Controller_Component_output,
                                    Surprise
                                 ):

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

    Err = Controller_Component_output[1] - Controller_Component_output[0]

    Gradient = np.dot(Err.T, Sensory_Component_output[3][0][1])

    DeltaW = Training_hyperparameters * Gradient * Surprise

    DeltaW_bias = Training_hyperparameters * Err

    Network_weights[0] += DeltaW.T # TRY TO USE A .T

    Network_weights[1] += DeltaW_bias

    return Network_weights




# GOAL-MONITORING COMPONENT -----------------

# ACTIVATION

def GOAL_MONITORING_COMPONENT_ACTIVATION(
                                                Goal_joints,
                                                Present_State_joints
                                        ):

    '''

    This function computates the reward on the bases of present joints state and the target joint state.

    args:
            Goal_joints: target joint configuration for a specific object
            Present_State_joints: present joint configuration

    return:

            Reward = goal-dependent reward
            Accuracy = goal-dependent accuracy (1 only if r = 1, else 0)
    '''

    L1_norm = np.sum(abs(Goal_joints - Present_State_joints))

    L1_norm_interp = np.interp(L1_norm, (0, Goal_joints.shape[0]), (0, 1))

    Reward = 1 - L1_norm_interp

    Accuracy = 0

    if Reward == 1:

        Accuracy = 1

    else:

        Accuracy = 0

    return [Reward, Accuracy]

# EVALUATOR COMPONENT -----------------

# INITIALIZATION


def EVALUATOR_COMPONENT_INITIALIZATION(
                                            Topology_Evaluator
                                      ):

    '''

    This function initializes the evaluator component (ML perceptron)

    Args:
            Topology_Evaluator: number of Input/Output units of ML Perceptron

    return:
            Evaluator_weights: weights of Evaluator

    '''

    Evaluator_weights = Critic_init_MLP(

                                            Topology_Evaluator[0]
                                       )

    return Evaluator_weights

# ACTIVATION

def EVALUATOR_COMPONENT_ACTIVATION(
                                    input_,
                                    weights_critic

                                  ):

        '''

        This function computes the spread of a ML Perceptron

        args:

            input_: original input of net
            weights_critic: weights matrices of critic

        return:

            Pred: predicted value of critic (expected reward)

        '''

        Hidden_activation = np.dot(input_, weights_critic[0])

        Hidden_activation = sigmoid(Hidden_activation)

        Pred = np.dot(Hidden_activation, weights_critic[1])



        return [Pred[0][0], Hidden_activation]

# TRAINING

def EVALUATOR_COMPONENT_TRAINING(
                                    Input_,
                                    Hidden_,
                                    weights_critic,
                                    Pred_R,
                                    Obtained_R,
                                    Training_hyperparameters
                                ):
    '''

    This function computes the evaluator component (MLP) update

    args:

        input_: original input of net
        Hidden_: activation of hidden layer of critic
        weights_critic: weights matrices of critic
        Pred_R: predicted value of critic (expected reward)
        Obtained_R: reward produced by network
        learning_rate_critic: learning rate of critic

    return:

        weights_critic: weights matrix of critic
        Err_output: Surprise

    '''


    Err_output = Obtained_R - Pred_R

    Err_hidden = np.dot(Err_output, weights_critic[1].T)

    DeltaW_input_to_hidden = Training_hyperparameters * np.dot(Err_hidden.T, Input_)

    DeltaW_hidden_to_output = Training_hyperparameters * np.dot(Err_output.T, Hidden_)


    weights_critic[0] += DeltaW_input_to_hidden.T

    weights_critic[1] += DeltaW_hidden_to_output.T


    return Err_output, weights_critic


# WHOLE SYSTEM -----------------


# INITIALIZATION

def WHOLE_SYSTEM_INITIALIZATION(

                                        Sensory_Topology,
                                        Controller_Topology,
                                        Evaluator_Topology

                                ):

    '''

    This function initializes the whole system (each component topology and contenitor variables, etc).

    args:

        Sensory_Topology: Topology of sensory component (DBN)
        Controller_Topology: Topology of controller component (perceptron)
        Evaluator_Topology: Topology of evaluator component (ML perceptron)

    return:

           Network_weights_RBM_1, Network_weights_RBM_2, Simulation_variables: Weights and simulation variables of DBN (contentiors, counts...)
           Network_weights_Controller: Weights of Controller
           Network_weights_Evaluator: Weights of Evaluator

    '''

    Rewards = []
    Surprises = []
    Accuracies = []

    Network_weights_RBM_1, Network_weights_RBM_2, Simulation_variables = SENSORY_COMPONENT_INITIALIZATION(

                                                                                                            Sensory_Topology[0],
                                                                                                            Sensory_Topology[1],
                                                                                                         )

    Network_weights_Controller = CONTROLLER_COMPONENT_INITIALIZATION(
                                                                        Controller_Topology

                                                                    )

    Network_weights_Evaluator = EVALUATOR_COMPONENT_INITIALIZATION(
                                                                        Evaluator_Topology
                                                                  )

    return [Rewards, Surprises, Accuracies],\
           [Network_weights_RBM_1, Network_weights_RBM_2, Simulation_variables], \
           Network_weights_Controller, \
           Network_weights_Evaluator

# ACTIVATION (single stochastic/online activation step)

def WHOLE_SYSTEM_ACTIVATION(
                                        Camera_Input,
                                        Sensory_variables,
                                        Controller_variables,
                                        Evaluator_variables
                           ):

    '''

    This Function allows an activation of all components of an embodied robotic architecture. In particular it activates
    the sensory component (a Deep Belifef Network, DBN) on the base of a Kuka camera input, the controller (a simple perceptron)
     and the evaluator component (a ML perceptron), both on the base of the more abstract representation returned by the DBN.

    Args:

         Camera_Input: visual input from the kuka camera
         Sensory_variables: Weights and hiper-parameters of DBN/DRBM
         Controller_variables: Weights and hiper-parameters of R-based perceptron
         Evaluator_variables: Weights and hiper-parameters of ML perceptron

    return:

        Sensory_Component_output:results of DBN/DRBM activation (format =   [original_input_first_RBM, Hidden_activations_first_RBM,
                                                                            Inputs_reconstructions_first_RBM, Hidden_activations_second_RBM,
                                                                            Inputs_reconstructions_second_RBM, Rec_errors_first_RBM, Rec_errors_second_RBM])
        Controller_Component_output: results of R-based perceptron activation (format =   [Output_sigm, Output_bin])
        Evaluator_Component_output: results of ML perceptron activation (format =   [Pred, Hidden_activation])

    '''

    # SENSORY COMPONENT activation

    Sensory_Component_output = SENSORY_COMPONENT_ACTIVATION(
                                                                              Camera_Input,
                                                                              Sensory_variables[0][0],
                                                                              Sensory_variables[0][1],
                                                                              Sensory_variables[1],
                                                                              Sensory_variables[2],
                                                                              Sensory_variables[3]
                                                           )

    # CONTROLLER COMPONENT activation

    Controller_Component_output = CONTROLLER_COMPONENT_ACTIVATON(
                                                                                     Sensory_Component_output[3][0][1],
                                                                                     Controller_variables
                                                                  )
    # EVALUATOR COMPONENT activation

    Evaluator_Component_output = EVALUATOR_COMPONENT_ACTIVATION(
                                                                                    Sensory_Component_output[3][0][1],
                                                                                    Evaluator_variables
                                                                )

    return Sensory_Component_output, Controller_Component_output, Evaluator_Component_output


# TRAINING (single stochastic/online training step)

def WHOLE_SYSTEM_TRAINING(  Sensory_Component_weights, Sensory_Component_output, Sensory_Component_training_hyperparameters,
                            Controller_Component_weights, Controller_Component_output, Controller_Component_training_hyperparameters,
                            Evaluator_Component_weights, Evaluator_Component_output, Evaluator_Component_training_hyperparameters,
                            Goal_joints
                         ):

    '''

    This Function allows a stochastic/online training step of all components of an embodied robotic architecture (see WHOLE_SYSTEM_ACTIVATION function),
    on the basis of a Known goal (correct joints position fo a specific object).

    Args:
         Sensory_Component_weights: weights of DBN/DRBM
         Sensory_Component_output: output of DBN/DRBM activation (format =   [original_input_first_RBM, Hidden_activations_first_RBM,
                                                                            Inputs_reconstructions_first_RBM, Hidden_activations_second_RBM,
                                                                            Inputs_reconstructions_second_RBM, Rec_errors_first_RBM, Rec_errors_second_RBM])
         Sensory_Component_training_hyperparameters: hyper-parameters of DBN/DRBM

         Controller_Component_weights: weight of r-based perceptron
         Controller_Component_output: results of R-based perceptron activation (format =   [Output_sigm, Output_bin])
         Controller_Component_training_hyperparameters: hyper-parameters of R-based perceptron

         Evaluator_Component_weights: weight of ML perceptron
         Evaluator_Component_output: results of ML perceptron activation (format =   [Pred, Hidden_activation])
         Evaluator_Component_training_hyperparameters: hyper-parameters of ML perceptron

         Goal_joints: joints vector corresponding to a target position of a specific object

    return:

             Sensory_Component_weights: updated weights of DBN/DRBM
             Controller_Component_weights: updated weight of r-based perceptron
             Evaluator_Component_weights: updated weight of ML perceptron




    '''

    Reward, Accuracy =  GOAL_MONITORING_COMPONENT_ACTIVATION(
                                                            Goal_joints,
                                                            Controller_Component_output[1]
                                                            )

    Surprise, Evaluator_Component_weights = EVALUATOR_COMPONENT_TRAINING(
                                                                            Sensory_Component_output[3][0][1],
                                                                            Evaluator_Component_output[1],
                                                                            Evaluator_Component_weights,
                                                                            Evaluator_Component_output[0],
                                                                            Reward,
                                                                            Evaluator_Component_training_hyperparameters
                                                                        )

    Sensory_Component_weights = SENSORY_COMPONENT_TRAINING(
                                                            Sensory_Component_training_hyperparameters,
                                                            Sensory_Component_weights,
                                                            Sensory_Component_output,
                                                            Surprise
                                                          )

    Controller_Component_weights = CONTROLLER_COMPONENT_TRAINING(
                                                                    Sensory_Component_output,
                                                                    Controller_Component_training_hyperparameters,
                                                                    Controller_Component_weights,
                                                                    Controller_Component_output,
                                                                    Surprise

                                                                )


    return [Reward, Surprise, Accuracy], Evaluator_Component_weights, Sensory_Component_weights, Controller_Component_weights

def WHOLE_SYSTEM_TRAINING_VARIABLES_COMPUTATION_AND_COLLECTION(

                                                                    Training_Check_Performance_variables_Sensory,
                                                                    Training_Check_Performance_variables_Whole_System,

                                                                    Sensory_Component_weights,

                                                                    Sensory_Component_output,
                                                                    Whole_system_output

                                                              ):

    '''

    This function collects the data of components during the training (DBN activations, surprises, rewards, accuracies).

    Args:

        :param Training_Check_Performance_variables_Sensory:
        :param Training_Check_Performance_variables_Whole_System:
        :param Sensory_Component_weights:
        :param Sensory_Component_output:
        :param Whole_system_output:

    return:

    '''

    Training_Check_Performance_variables_Sensory = SENSORY_COMPONENT_TRAINING_VARIABLES_COMPUTATION_AND_COLLECTION(

                                                                                                                    Training_Check_Performance_variables_Sensory,
                                                                                                                    Sensory_Component_weights,
                                                                                                                    Sensory_Component_output

                                                                                                                  )

    Training_Check_Performance_variables_Whole_System.append(Whole_system_output)

    return Training_Check_Performance_variables_Sensory, Training_Check_Performance_variables_Whole_System