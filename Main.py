
# import sys
#
# Path = sys.path

from System_Components_Functions import *
from Enviroments import *
import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')


# MAIN FUNCTION

def MAIN(TRAIN_BASIC_ENV,
         TRAIN_BASIC_ENV_ADVERSARY,

         FIRST_RBM_TRAINING,
         SECOND_RBM_TRAINING,

         SHOW_TRAINING_RESULTS_BASIC_ENV,
         SHOW_TRAINING_RESULTS_BASIC_ENV_ADVERSARY,

         SHOW_TRAINING_SINGLE_RBM_FIRST,
         SHOW_TRAINING_SINGLE_RBM_SECOND,

         TRAIN_ROBOTIC_ENV,
         TRAIN_ROBOTIC_ENV_ADVERSARY,

         SHOW_TRAINING_RESULTS_ROBOTIC_ENV,
         SHOW_TRAINING_RESULTS_ROBOTIC_ENV_ADVERSARY,
         ADVERSARY_COMPARISON,


         TEST_GENERATIVE_MODEL,
         TEST_GENERATIVE_MODEL_ADVERSARY,

         TEST_DBN,
         TEST_RBM,

         VISUAL_CHECK,
         VISUAL_CHECK_COMPARISON,
         VISUAL_CHECK_EACH_INPUT,

         TEST_REPRESENTATIONS,
         TEST_REPRESENTATIONS_ADVERSARY,

         TEST_REPRESENTATIONS_VISIBLE,
         TEST_REPRESENTATIONS_FIRST_HIDDEN,
         TEST_REPRESENTATIONS_SECOND_HIDDEN,

         SHOW_TEST_REPRESENTATIONS_RESULTS,
         ADVERSARY_TESTER_COMPARISON,

         SHOW_TEST_REPRESENTATIONS_RESULTS_RBM,
         SHOW_TEST_REPRESENTATIONS_RESULTS_DBN,

         SHOW_TEST_REPRESENTATIONS_RESULTS_VISIBLE,
         SHOW_TEST_REPRESENTATIONS_RESULTS_FIRST_HIDDEN,
         SHOW_TEST_REPRESENTATIONS_RESULTS_SECOND_HIDDEN,



         TEST_RESONANCES_GIBBS_SAMPLING,
         TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY,
         TEST_RESONANCES_GIBBS_SAMPLING_RBM,
         TEST_RESONANCES_GIBBS_SAMPLING_DBN,

         salient_feature_DBN, Input_units_FIRST_DBN, Hidden_units_FIRST_DBN, Input_units_SECOND_DBN,
         Hidden_units_SECOND_DBN,
         number_of_resonances_FIRST_DBN,
         number_of_resonances_SECOND_DBN, number_of_resonances_WHOLE_DBN,


         Graphic_on_FIRST,
         Prints_on_FIRST,
         Batch_on_FIRST, salient_feature_FIRST, Input_units_FIRST, Hidden_units_FIRST,
         Tot_epocs_FIRST,
         number_of_resonances_FIRST, learning_rate_FIRST, alfa_FIRST, target_sparsety_FIRST,
         target_min_error_FIRST, save_choice_FIRST,

         Graphic_on_SECOND, Prints_on_SECOND, Batch_on_SECOND,

         salient_feature_SECOND, Input_units_SECOND, Hidden_units_SECOND, Tot_epocs_SECOND,
         number_of_resonances_SECOND, learning_rate_SECOND, alfa_SECOND, target_sparsety_SECOND,
         target_min_error_SECOND,
         save_choice_SECOND,

         Env_to_test,
         Executor_deep,
         Executor_Input,
         Executor_size_Output,
         Learning_rate_SECOND_RL_Contribution,
         CD_weight_robotic_env,
         L_rate_executor,
         L_rate_executor_critic,
         tot_epocs_robotic_env,
         Tester_Learning_Rule,
         size_hidden_tester,
         L_rate_tester,
         L_rate_tester_critic,
         Binary_Reward_format_tester, tot_epocs_test

         ):

    '''


    The main function is composed by many training enviroments (they include also sub-sections to plot the training data) and
    a macro-section  that allows to analyse and test the internal computations of Networks (layers activations, weights, sampling..).


    - TRAIN_BASIC_ENV. This section allows to train the network (only RBM/DBN without other components). Use TRAIN_BASIC_ENV_ADVERSARY
            to train an hypothetical adversarial network


            * FIRST_RBM_TRAINING: learning of first RBM
            * SECOND_RBM_TRAINING: learning of second RBM


    - SHOW_TRAINING_RESULTS. This section loads the training data (only RBM/DBN) and allows to visualize them off-line.
                             Use SHOW_TRAINING_RESULTS_ADVERSARY to visualize the results of an hypothetical adversarial network.

            * SHOW_TRAINING_SINGLE_RBM_FIRST: show training data of first RBM
            * SHOW_TRAINING_SINGLE_RBM_SECOND: show training data of second RBM


    - TRAIN_ROBOTIC_ENV. This section allows to train a neurorobotic achitecture in a robotic-like enviroment.
                        Use TRAIN_ROBOTIC_ENV_ADVERSARY to train an hypothetical adversarial network


    - SHOW_TRAINING_RESULTS_ROBOTIC_ENV. This section loads the training data (a composited system, i.e. second R-based RBM + R-based perceptron)
                                        and allows to visualize them off-line. It includes a sub-section that
                                        allows to comparate the rewards/accuracies oftwo net training (modified version and default version).
                                        Use SHOW_TRAINING_RESULTS_ROBOTIC_ENV_ADVERSARY to visualize the results of an hypothetical adversarial network.




    - TEST_GENERATIVE_MODEL. This section allows to test many features of the generative model. Use TEST_GENERATIVE_MODEL_ADVERSARY
                        to test an hypothetical adversarial network



            * TEST_RBM: spread/reconstruction of the whole dataset (64 input) of first RBM
            * TEST_DBN: spread/reconstruction of the whole dataset (64 input) of the whole DBN

            Each section allows to choose load the weights NET dependinding on training enviroment (basic env., robotic...)

            Each sub-section of these is coupled by two supplementary graphical functions:

                    % VISUAL_CHECK: it shows the visual reconstructions of whole dataset (64 inputs) and the 3D layer activations
                                    of visible layer of RBM (input layer of DBN).

                                    % VISUAL_CHECK_COMPARISON: it shows a visual comparison between the data of upper-section (VISUAL_CHECK)
                                                               and the data returned by an adversarial network (a net trained with default algorythm).


                    %% VISUAL_CHECK_EACH_INPUT: it allows to execute a spread of net for each specific input, opening a windows
                                                that shows (I) the single reconstruction, (II) the activation of single units into
                                                hidden layers and (III) their receptive fields.

    - TEST_REPRESENTATIONS. this function test the representations (activation of generative model layers) with an extrinsic task
                            (perceptron classification) depending a specific salient feature (color, form or size),
                            topology (RBM or DBN) and training rule of tester ('backprop' or 'REINFORCE').
                            use TEST_REPRESENTATIONS_ADVERSARY to test also the represetations of an hypothetical adversarial network.


                            * TEST_REPRESENTATIONS_RBM: test of the representations for a single RBM (First)
                            ** TEST_REPRESENTATIONS_DBN: test of the representations for a whole DBN

    - SHOW_TEST_REPRESENTATIONS_RESULTS:    this section loads the training data of tester and allows to visualize them off-line.
                                            It includes three sub-section as the previous one (TEST_REPRESENTATIONS, i.e. layer dependent)
                                            and each sub-section has a sub-section that allows to show a single tester or a visual comparison
                                            of two testers.


    - TEST_RESONANCES_GIBBS_SAMPLING: this function executes a spread of generative model (RBM or DBN) of a single input and allows
                                      to study the Gibbs sampling process (resonances). In particular it returns different
                                      graphical windows of attractor fields into the 3D activation space.
                                      Use TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY to test an hypothetical adversarial network.

                                      * TEST_RESONANCES_GIBBS_SAMPLING_RBM: test of sampling process from input to first hidden
                                      ** TEST_RESONANCES_GIBBS_SAMPLING_DBN: test of sampling process from input to first hidden
                                                                            and from first hidden to second hidden.
                                                                            This function allows to also study the sampling
                                                                            process of whole DBN (input -> first hidden
                                                                            -> second hidden -> first hidden -> input)

    The Main function has 3 Ester Eggs....enjoy!

    '''



    # %%%%%%%%%%%%% TRAINING AND VISUAL CHECK POST-TRAINING OF RBM/DBN WITHOUT ANY OTHER COMPONENT %%%%%%%%%%%%%

    if TRAIN_BASIC_ENV:

        weights_save_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

        training_data_save_path = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

        Saves_path_training_visual_outputs = '.\\Training_Visual_Outputs\\'

        Saves_Folders_Paths = [weights_save_path, training_data_save_path, Saves_path_training_visual_outputs]

        if TRAIN_BASIC_ENV_ADVERSARY:

            weights_save_path = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


            training_data_save_path = '.\\Training_data\\Basic_enviroment_data\\Adversary\\'

            Saves_path_training_visual_outputs = '.\\Training_Visual_Outputs\\'

            Saves_Folders_Paths = [weights_save_path, training_data_save_path, Saves_path_training_visual_outputs]


        if FIRST_RBM_TRAINING:

            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Random_initialization_weights(Input_units_FIRST, Hidden_units_FIRST)

            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            dataset = Load_images(Input_units_FIRST)

            Ideal_actions, Ideal_actions_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_FIRST)

            Network_weights_FIRST = RBM_Training(Network_weights_FIRST,
                                                 dataset,

                                                 [Ideal_actions, Ideal_actions_batch, Legend_lab],

                                                 Batch_on_FIRST,
                                                 number_of_resonances_FIRST,
                                                 save_choice_FIRST, Graphic_on_FIRST, Prints_on_FIRST,
                                                 Saves_Folders_Paths,

                                                 Tot_epocs_FIRST, learning_rate_FIRST, alfa_FIRST, target_sparsety_FIRST, target_min_error_FIRST)

        elif SECOND_RBM_TRAINING:


            load_weights_tested_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'


            dataset = Load_images(Input_units_FIRST)

            Ideal_actions, Ideal_actions_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_SECOND)

            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, load_weights_tested_path)

            Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
            Weights_bias_hiddens_to_inputs_SECOND = Random_initialization_weights(Input_units_SECOND,
                                                                                  Hidden_units_SECOND)


            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            Network_weights_SECOND = [Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND,
                                      Weights_bias_hiddens_to_inputs_SECOND]


            Network_weights_SECOND = RBM_Training(Network_weights_SECOND,
                                                  dataset,
                                                  [Ideal_actions, Ideal_actions_batch, Legend_lab],

                                                  Batch_on_SECOND,
                                                  number_of_resonances_SECOND,
                                                  save_choice_SECOND, Graphic_on_SECOND, Prints_on_SECOND,

                                                  Tot_epocs_SECOND, learning_rate_SECOND, alfa_SECOND,
                                                  target_sparsety_SECOND, target_min_error_SECOND,
                                                  Network_weights_FIRST)

    if SHOW_TRAINING_RESULTS_BASIC_ENV:

        load_path_tested = '.\\Training_data\\Basic_enviroment_data\\Tested\\'

        load_weights_tested_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

        if SHOW_TRAINING_RESULTS_BASIC_ENV_ADVERSARY:

            load_path_tested = '.\\Training_data\\Basic_enviroment_data\\Adversary\\'

            load_weights_tested_path = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


        if SHOW_TRAINING_SINGLE_RBM_FIRST:


            Errors_FIRST, STDs_Errors_FIRST = load_training_data_JOIN([Input_units_FIRST, Hidden_units_FIRST],
                                                                 learning_modality = 'UL', load_path = load_path_tested)

            UL_Learning_RBM_FIRST_img = Window_builder_reconstruction_errors(Errors_FIRST, STDs_Errors_FIRST,target_min_error_FIRST)

            # WEIGHTS DISTRIBUTIONS PLOTS

            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST,
                                                                load_weights_tested_path)

            Weights_img_FIRST = Windows_builder_network_weights(Weight_inputs_FIRST,
                                                          Weights_bias_inputs_to_hiddens_FIRST,
                                                          Weights_bias_hiddens_to_inputs_FIRST)

        if SHOW_TRAINING_SINGLE_RBM_SECOND:



            Errors_SECOND, STDs_Errors_SECOND = load_training_data_JOIN([Input_units_SECOND, Hidden_units_SECOND],
                                                                   learning_modality = 'UL', load_path = load_path_tested)

            UL_Learning_RBM_SECOND_img = Window_builder_reconstruction_errors(Errors_SECOND, STDs_Errors_SECOND, target_min_error_SECOND)

            # WEIGHTS DISTRIBUTIONS PLOTS

            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST,
                                                                load_weights_tested_path)

            Weights_img_FIRST = Windows_builder_network_weights(Weight_inputs_FIRST,
                                                                Weights_bias_inputs_to_hiddens_FIRST,
                                                                Weights_bias_hiddens_to_inputs_FIRST)

            Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
            Weights_bias_hiddens_to_inputs_SECOND = Load_weights(Input_units_SECOND, Hidden_units_SECOND,
                                                                load_weights_tested_path)

            Weights_img_SECOND = Windows_builder_network_weights(Weight_inputs_SECOND,
                                                          Weights_bias_inputs_to_hiddens_SECOND,
                                                          Weights_bias_hiddens_to_inputs_SECOND)


        plt.show()

    # %%%%%%%%%%%%% TRAINING AND VISUAL CHECK POST-TRAINING OF RBM/DBN INTEGRATED INTO ROBOTIC ARCHITECTURE/ENVIROMENTS %%%%%%%%%%%%%


    if TRAIN_ROBOTIC_ENV:

        Saves_path_training_visual_outputs = '.\\Training_Visual_Outputs\\'

        if TRAIN_ROBOTIC_ENV_ADVERSARY:

            save_path_weights = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

            Saves_path = '.\\Training_data\\Robotic_enviroment\\Adversary (CD)\\'

            Save_Folders_Paths = [save_path_weights, Saves_path, Saves_path_training_visual_outputs]

        else:

            save_path_weights = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'

            Saves_path = '.\\Training_data\\Robotic_enviroment\\Tested (R-CD)\\'

            Save_Folders_Paths = [save_path_weights, Saves_path, Saves_path_training_visual_outputs]

        clean_training_data_folder(Saves_path_training_visual_outputs)
        clean_training_data_folder(Saves_path)

        dataset = Load_images(Input_units_FIRST)

        Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
        Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, save_path_weights)

        Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
        Weights_bias_hiddens_to_inputs_SECOND = Random_initialization_weights(Input_units_SECOND,
                                                                              Hidden_units_SECOND)

        Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                 Weights_bias_hiddens_to_inputs_FIRST]
        Network_weights_SECOND = [Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND,
                                  Weights_bias_hiddens_to_inputs_SECOND]

        if Executor_deep:

            if Executor_Input == 'visible':

                Executor_weights = Executor_init_MLP(Network_weights_FIRST[2].shape[1], Executor_size_Output)


            elif Executor_Input == 'first hidden':

                Executor_weights = Executor_init_MLP(Network_weights_FIRST[1].shape[1], Executor_size_Output)


            elif Executor_Input == 'second hidden':

                Executor_weights = Executor_init_MLP(Network_weights_SECOND[1].shape[1], Executor_size_Output)




        else:

            if Executor_Input == 'visible':

                Executor_weights = Executor_init(Network_weights_FIRST[2].shape[1], Executor_size_Output)


            elif Executor_Input == 'first hidden':

                Executor_weights = Executor_init(Network_weights_FIRST[1].shape[1], Executor_size_Output)


            elif Executor_Input == 'second hidden':

                Executor_weights = Executor_init(Network_weights_SECOND[1].shape[1], Executor_size_Output)



        Network_weights_SECOND, \
        Executor_weights = Robotic_Agent_Training(
                                                     [[Input_units_FIRST, Hidden_units_FIRST], [Input_units_SECOND, Hidden_units_SECOND]],
                                                     number_of_resonances_FIRST_DBN,
                                                     number_of_resonances_SECOND_DBN,
                                                     number_of_resonances_WHOLE_DBN,
                                                     learning_rate_SECOND,
                                                     Learning_rate_SECOND_RL_Contribution,
                                                     CD_weight_robotic_env,
                                                     [Network_weights_SECOND[1].shape[1], Executor_size_Output],
                                                     L_rate_executor,
                                                     [Network_weights_SECOND[1].shape[1], 1],
                                                     L_rate_executor_critic,
                                                     dataset,
                                                     save_choice_SECOND,
                                                     Graphic_on_SECOND,
                                                     Prints_on_SECOND,
                                                     Save_Folders_Paths,
                                                     salient_feature_SECOND,
                                                     tot_epocs_robotic_env,
                                                     )

    if SHOW_TRAINING_RESULTS_ROBOTIC_ENV:


            if SHOW_TRAINING_RESULTS_ROBOTIC_ENV_ADVERSARY and not ADVERSARY_COMPARISON:

                load_path_adversary = '.\\Training_data\\Robotic_enviroment\\Adversary (CD)\\'

                load_weights_adversary_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

                training_load_path = copy.deepcopy(load_path_adversary)

                weights_load_path = copy.deepcopy(load_weights_adversary_path)

            else:

                load_path_tested = '.\\Training_data\\Robotic_enviroment\\Tested (R-CD)\\'

                load_weights_tested_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'

                training_load_path = copy.deepcopy(load_path_tested)

                weights_load_path = copy.deepcopy(load_weights_tested_path)



            Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
            Weights_bias_hiddens_to_inputs_SECOND = Load_weights(Input_units_SECOND, Hidden_units_SECOND, weights_load_path)

            Learning_rate, Learning_rate_critic, R_range, R_range_interp, ideal_actions, Reinforces_for_each_epoc, \
            Reinforces_for_each_epoc_STD, Surprises_for_each_epoc, Surprises_for_each_epoc_STD, Accuracies_for_each_epoc, Accuracies_for_each_epoc_STD, \
            Reconstruction_errors_for_each_epoc, Weights_network, Sum_weights_network_for_each_epoc, \
            Sum_bias_inputs_to_hiddens_weights_for_each_epoc, Sum_bias_hiddens_to_inputs_weights_for_each_epoc, \
            Reinforces_for_each_input, Surprises_for_each_input, \
            Accuracies_for_each_input = load_training_data_reinforced_CD_JOIN([Input_units_SECOND, Hidden_units_SECOND],
                                                                              training_load_path)



            RL_Learning_RBM_SECOND_img = Window_builder_check_panel_RBM(Reinforces_for_each_epoc,
                                                                        Reinforces_for_each_epoc_STD,
                                                                        Surprises_for_each_epoc,
                                                                        Surprises_for_each_epoc_STD,
                                                                        Reinforces_for_each_input,
                                                                        Surprises_for_each_input,
                                                                        Weight_inputs_SECOND,
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
                                                                        ideal_actions)

            if not ADVERSARY_COMPARISON:

                Rewards_Accuracies_Network = Window_builder_tester_performances_REINFORCE('Robotic_like env',
                                                                                                         salient_feature_SECOND,
                                                                                                         Reinforces_for_each_epoc,
                                                                                                         Accuracies_for_each_epoc)

            else:

                load_path_adversary = '.\\Training_data\\Robotic_enviroment\\Adversary (CD)\\'

                load_weights_adversary_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

                Weight_inputs_SECOND_ADVERSARY, Weights_bias_inputs_to_hiddens_SECOND_ADVERSARY, \
                Weights_bias_hiddens_to_inputs_SECOND_ADVERSARY = Load_weights(Input_units_SECOND, Hidden_units_SECOND, # CHANGE HERE
                                                                     load_weights_adversary_path)

                Learning_rate, Learning_rate_critic, R_range, R_range_interp, ideal_actions, Reinforces_for_each_epoc_ADVERSARY, \
                Reinforces_for_each_epoc_STD_ADVERSARY, Surprises_for_each_epoc_ADVERSARY, Surprises_for_each_epoc_STD_ADVERSARY, Accuracies_for_each_epoc_ADVERSARY, Accuracies_for_each_epoc_STD_ADVERSARY, \
                Reconstruction_errors_for_each_epoc_ADVERSARY, Weights_network_ADVERSARY, Sum_weights_network_for_each_epoc_ADVERSARY, \
                Sum_bias_inputs_to_hiddens_weights_for_each_epoc_ADVERSARY, Sum_bias_hiddens_to_inputs_weights_for_each_epoc_ADVERSARY, \
                Reinforces_for_each_input_ADVERSARY, Surprises_for_each_input_ADVERSARY, \
                Accuracies_for_each_input_ADVERSARY = load_training_data_reinforced_CD_JOIN(
                    [Input_units_SECOND, Hidden_units_SECOND], load_path_adversary) # CHANGE HERE

                RL_Learning_RBM_SECOND_ADVERSARY_img = Window_builder_check_panel_RBM(Reinforces_for_each_epoc_ADVERSARY,
                                                                            Reinforces_for_each_epoc_STD_ADVERSARY,
                                                                            Surprises_for_each_epoc_ADVERSARY,
                                                                            Surprises_for_each_epoc_STD_ADVERSARY,
                                                                            Reinforces_for_each_input_ADVERSARY,
                                                                            Surprises_for_each_input_ADVERSARY,
                                                                            Weight_inputs_SECOND_ADVERSARY,
                                                                            Sum_weights_network_for_each_epoc_ADVERSARY,
                                                                            Sum_bias_inputs_to_hiddens_weights_for_each_epoc_ADVERSARY,
                                                                            Sum_bias_hiddens_to_inputs_weights_for_each_epoc_ADVERSARY,
                                                                            Accuracies_for_each_epoc_ADVERSARY,
                                                                            Accuracies_for_each_epoc_STD_ADVERSARY,
                                                                            Accuracies_for_each_input_ADVERSARY,
                                                                            Reconstruction_errors_for_each_epoc_ADVERSARY,
                                                                            Learning_rate,
                                                                            Learning_rate_critic, R_range,
                                                                            R_range_interp,
                                                                            ideal_actions)

                Accuracies_both_sys = np.vstack((Accuracies_for_each_epoc_ADVERSARY, Accuracies_for_each_epoc))

                Rewards_both_sys = np.vstack((Reinforces_for_each_epoc_ADVERSARY, Reinforces_for_each_epoc))

                Comparison_Two_Networks = Window_builder_tester_performances_REINFORCE_DOUBLE_COMPARISON('Robotic_like env',
                                                                                        salient_feature_SECOND,
                                                                                        Rewards_both_sys.T,
                                                                                        Accuracies_both_sys.T)



            plt.show()


    # %%%%%%%%%%%%% TEST MAIN %%%%%%%%%%%%%

    if TEST_GENERATIVE_MODEL:

        dataset = Load_images(Input_units_FIRST)

        if Env_to_test == 'basic':

            load_weights_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

            load_weights_path_adversary = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'

            if TEST_GENERATIVE_MODEL_ADVERSARY and not VISUAL_CHECK_COMPARISON:

                load_weights_path = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


        elif Env_to_test == 'robotic':


            load_weights_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'

            load_weights_path_adversary = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

            load_weights_path_FIRST = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'


            if TEST_GENERATIVE_MODEL_ADVERSARY and not VISUAL_CHECK_COMPARISON:

                load_weights_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

                load_weights_path_FIRST = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'


        if TEST_RBM:


            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, load_weights_path)

            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            Weights_img = Windows_builder_network_weights(Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, Weights_bias_hiddens_to_inputs_FIRST)


            print('')
            print('------------------------')
            print('')
            print(' *** RBM whole dataset spread/reconstruction (input <-> first hidden) *** ')
            print('')
            print('------------------------')
            print('')

            Input_FIRST, Hidden_activations_FIRST, Inputs_reconstructions_FIRST, \
            Rec_errors_FIRST = RBM_DBN_Activation(dataset, 1, Network_weights_FIRST, salient_feature_FIRST)

            print('')
            print('------------------------')
            print(" Batch Reconstruction Error (L1 norm) = ", Rec_errors_FIRST[-1][1])
            print('')
            print(" Reconstruction error for each input (L1 norm) = ", np.array(Rec_errors_FIRST[-1][0]))
            print('------------------------')

            if VISUAL_CHECK_COMPARISON:

                Weight_inputs_FIRST_ADVERSARY, Weights_bias_inputs_to_hiddens_FIRST_ADVERSARY, \
                Weights_bias_hiddens_to_inputs_FIRST_ADVERSARY = Load_weights(Input_units_FIRST, Hidden_units_FIRST,
                                                                              load_weights_path_adversary)

                Network_weights_FIRST_ADVERSARY = [Weight_inputs_FIRST_ADVERSARY,
                                                   Weights_bias_inputs_to_hiddens_FIRST_ADVERSARY,
                                                   Weights_bias_hiddens_to_inputs_FIRST_ADVERSARY]

                Input_FIRST_ADVERSARY, Hidden_activations_FIRST_ADVERSARY, Inputs_reconstructions_FIRST_ADVERSARY, \
                Rec_errors_FIRST_ADVERSARY = RBM_DBN_Activation(dataset, number_of_resonances_FIRST,
                                                                Network_weights_FIRST_ADVERSARY,
                                                                salient_feature_FIRST)

                print('')
                print(" (ADVERSARY) RBM Batch Reconstruction Error (L1 norm) = ", Rec_errors_FIRST_ADVERSARY[-1][1])
                print('')
                print(" (ADVERSARY) RBM Reconstruction error for each input (L1 norm) = ",
                      np.array(Rec_errors_FIRST_ADVERSARY[-1][0]))
                print('------------------------')

            if VISUAL_CHECK:


                Rec_inputs_RBM_img, Inputs_reconstructions_RBM = Window_builder_all_inputs_recostructions(
                    Inputs_reconstructions_FIRST[-1])

                # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                Visible_CD_Biased, Explained_Var_Visible_Biased, Explained_Var_ratio_Visible_Biased = dim_reduction(
                    Inputs_reconstructions_FIRST)

                # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                First_Hidden_CD_Biased, Explained_Var_First_Hidden_Biased, Explained_Var_ratio_First_Hidden_Biased = dim_reduction(
                    Hidden_activations_FIRST[-1], 3)

                if VISUAL_CHECK_COMPARISON:

                    Rec_inputs_RBM_ADVERSARY_img, Inputs_reconstructions_RBM_ADVERSARY = Window_builder_all_inputs_recostructions(
                        Inputs_reconstructions_FIRST_ADVERSARY[-1])

                    # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                    Visible_CD_Pure, Explained_Var_Visible, Explained_Var_ratio_Visible = dim_reduction(
                        Inputs_reconstructions_FIRST_ADVERSARY)

                    # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                    First_Hidden_CD_Pure, Explained_Var_First_Hidden, Explained_Var_ratio_First_Hidden = dim_reduction(
                        Hidden_activations_FIRST_ADVERSARY[-1])

                print('')
                print('-------------RBM PCA--------------')
                print('')
                print(' Explained_Var (Visible): ')

                if VISUAL_CHECK_COMPARISON:
                    print('          - Pure = ' + str(Explained_Var_Visible) + str(' ') + str('(') + str(
                        Explained_Var_ratio_Visible) + str(')'))
                    print('          - Total Variance explained = ', np.sum(Explained_Var_Visible), '%')
                    print('')

                print('          - Biased = ' + str(Explained_Var_Visible_Biased) + str(' ') + str('(') + str(
                    Explained_Var_ratio_Visible_Biased) + str(')'))
                print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_Biased), '%')
                print('')
                print('----------------------------')
                print('')
                print(' Explained_Var (Hidden): ')
                print('')

                if VISUAL_CHECK_COMPARISON:

                    print('          - Pure = ' + str(Explained_Var_First_Hidden) + str(' ') + str('(') + str(
                        Explained_Var_ratio_First_Hidden) + str(')'))
                    print('          - Total Variance explained = ', np.sum(Explained_Var_ratio_First_Hidden), '%')
                    print('')

                print('          - Biased = ' + str(Explained_Var_First_Hidden_Biased) + str(' ') + str('(') + str(
                    Explained_Var_ratio_First_Hidden_Biased) + str(')'))
                print('          - Total Variance explained = ', np.sum(Explained_Var_ratio_First_Hidden_Biased), '%')
                print('')
                print('----------------------------')

                # 3D PLOTS OF LAYER ACTIVATIONS%%%%%

                Labels_dataset, Labels_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_FIRST)


                if VISUAL_CHECK_COMPARISON:

                    Visible_activation_disentanglement_img = Window_builder_layers_activations(
                        Visible_CD_Biased,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'input',
                        2,
                        Visible_CD_Pure
                        )


                    First_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                        First_Hidden_CD_Biased,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'FHL',
                        2,
                        First_Hidden_CD_Pure
                    )

                else:

                    # INPUT LAYER %


                    Visible_activation_disentanglement_img = Window_builder_layers_activations(
                        Visible_CD_Biased,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'input',
                        2,
                        )

                    # FIRST HIDDEN LAYER %

                    First_Hidden_activation_disentanglement_img = Window_builder_layers_activations(
                        First_Hidden_CD_Biased,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'FHL',
                        2,
                                            )




                plt.show()


                if VISUAL_CHECK_EACH_INPUT:

                    for input in range(0, dataset.shape[0]):

                        hiddens_RF_to_plots = 10

                        Specific_original_input = dataset[input, :]
                        Specific_reconstructed_input = Inputs_reconstructions_FIRST[-1][input, :]
                        Specific_hidden_activation = Hidden_activations_FIRST[-1][input, :]
                        Specific_rec_error = (Rec_errors_FIRST[-1][0][input])

                        Input_spread_reconstruction_RBM_W = Window_builder_single_input_recostruction(
                            Specific_original_input, Specific_reconstructed_input,
                            Specific_hidden_activation, Weight_inputs_FIRST,
                            Weights_bias_hiddens_to_inputs_FIRST, Specific_rec_error,
                            hiddens_RF_to_plots)

                        print('')
                        print('------------------------')
                        print('')
                        print(' *** RBM single input spread/reconstruction (input <-> first hidden) *** ')
                        print('')
                        print('------------------------')

                        print(str(" Image n ") + str(input) + str(' ----------------------------'))
                        print('')
                        print(" Rec_Err (L1 norm) = ", Specific_rec_error)
                        print('')
                        print('----------------------------')

                        plt.show()

        elif TEST_DBN:


            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST_DBN, Hidden_units_FIRST_DBN, load_weights_path)

            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
            Weights_bias_hiddens_to_inputs_SECOND = Load_weights(Input_units_SECOND_DBN, Hidden_units_SECOND_DBN, load_weights_path)

            Network_weights_SECOND = [Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND,
                                      Weights_bias_hiddens_to_inputs_SECOND]

            Weights_FIRST_img = Windows_builder_network_weights(Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, Weights_bias_hiddens_to_inputs_FIRST)

            Weights_SECOND_img = Windows_builder_network_weights(Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, Weights_bias_hiddens_to_inputs_SECOND)



            print('')
            print('----------------------')
            print(' *** DBN whole dataset spread/reconstruction (input <-> first hidden <-> second hidden) *** ')
            print('')
            print('------------------------')
            print('')


            Input_FIRST, Hidden_activations_FIRST, Hidden_activations_SECOND, Inputs_reconstructions_FIRST, \
            Inputs_reconstructions_SECOND, Rec_errors_FIRST, Rec_errors_SECOND = RBM_DBN_Activation(dataset,
                                                                                                    1,
                                                                                                    Network_weights_FIRST,
                                                                                                    salient_feature_DBN,
                                                                                                    1,
                                                                                                    Network_weights_SECOND,
                                                                                                    0)

            print('')
            print(" DBN Batch Reconstruction Error (L1 norm) = ", Rec_errors_FIRST[-1][1])
            print('')
            print(" DBN Reconstruction error for each input (L1 norm) = ", np.array(Rec_errors_FIRST[-1][0]))
            print('------------------------')

            if VISUAL_CHECK_COMPARISON:

                Weight_inputs_FIRST_ADVERSARY, Weights_bias_inputs_to_hiddens_FIRST_ADVERSARY, \
                Weights_bias_hiddens_to_inputs_FIRST_ADVERSARY = Load_weights(Input_units_FIRST_DBN,
                                                                              Hidden_units_FIRST_DBN,
                                                                              load_weights_path_FIRST)

                Network_weights_FIRST_ADVERSARY = [Weight_inputs_FIRST_ADVERSARY,
                                                   Weights_bias_inputs_to_hiddens_FIRST_ADVERSARY,
                                                   Weights_bias_hiddens_to_inputs_FIRST_ADVERSARY]

                Weight_inputs_SECOND_ADVERSARY, Weights_bias_inputs_to_hiddens_SECOND_ADVERSARY, \
                Weights_bias_hiddens_to_inputs_SECOND_ADVERSARY = Load_weights(Input_units_SECOND_DBN,
                                                                               Hidden_units_SECOND_DBN,
                                                                               load_weights_path_adversary) # CHANGE HERE

                Network_weights_SECOND_ADVERSARY = [Weight_inputs_SECOND_ADVERSARY,
                                                    Weights_bias_inputs_to_hiddens_SECOND_ADVERSARY,
                                                    Weights_bias_hiddens_to_inputs_SECOND_ADVERSARY]

                Input_FIRST_ADVERSARY, Hidden_activations_FIRST_ADVERSARY, Hidden_activations_SECOND_ADVERSARY, Inputs_reconstructions_FIRST_ADVERSARY, \
                Inputs_reconstructions_SECOND_ADVERSARY, Rec_errors_FIRST_ADVERSARY, Rec_errors_SECOND_ADVERSARY = RBM_DBN_Activation(
                    dataset,
                    number_of_resonances_FIRST_DBN,
                    Network_weights_FIRST_ADVERSARY,
                    salient_feature_DBN,
                    number_of_resonances_SECOND_DBN,
                    Network_weights_SECOND_ADVERSARY,
                    number_of_resonances_WHOLE_DBN)

                print('')
                print(" (ADVERSARY) DBN Batch Reconstruction Error (L1 norm) = ", Rec_errors_FIRST_ADVERSARY[-1][1])
                print('')
                print(" (ADVERSARY) DBN Reconstruction error for each input (L1 norm) = ", np.array(Rec_errors_FIRST_ADVERSARY[-1][0]))
                print('------------------------')

            if VISUAL_CHECK:

                Rec_inputs_RBM_img, Inputs_reconstructions_RBM = Window_builder_all_inputs_recostructions(
                    Inputs_reconstructions_FIRST[-2])

                Rec_inputs_DBN_img, Inputs_reconstructions_DBN = Window_builder_all_inputs_recostructions(
                    Inputs_reconstructions_SECOND[-1],
                    Network_weights_FIRST[0],
                    Network_weights_FIRST[2])

                # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                Visible_CD_Biased, Explained_Var_Visible_Biased, Explained_Var_ratio_Visible_Biased = dim_reduction(
                    Inputs_reconstructions_DBN, 3)

                # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                Visible_CD_Biased_SECOND, Explained_Var_Visible_Biased_SECOND, Explained_Var_ratio_Visible_Biased_SECOND = dim_reduction(
                    Inputs_reconstructions_SECOND[-1], 3)

                # DIMENSIONAL REDUCTIONS (SECOND HIDDEN LAYER)%%%%%

                Visible_CD_Biased_SECOND_HIDDEN, \
                Explained_Var_Visible_Biased_SECOND_HIDDEN, \
                Explained_Var_ratio_Visible_Biased_SECOND_HIDDEN = dim_reduction(Hidden_activations_SECOND[-1], 3)

                if VISUAL_CHECK_COMPARISON:

                    Rec_inputs_DBN_ADVERSARY_img, Inputs_reconstructions_ADVERSARY_DBN = Window_builder_all_inputs_recostructions(
                        Inputs_reconstructions_SECOND_ADVERSARY[-1],
                        Network_weights_FIRST_ADVERSARY[0],
                        Network_weights_FIRST_ADVERSARY[2])



                    # DIMENSIONAL REDUCTIONS (INPUT LAYER)%%%%%

                    Visible_CD_Pure, Explained_Var_Visible, Explained_Var_ratio_Visible = dim_reduction(Inputs_reconstructions_ADVERSARY_DBN, 3)


                    # DIMENSIONAL REDUCTIONS (FIRST HIDDEN LAYER)%%%%%

                    Visible_CD_Pure_SECOND, Explained_Var_Visible_SECOND, \
                    Explained_Var_ratio_Visible_SECOND = dim_reduction(Inputs_reconstructions_SECOND_ADVERSARY[-1], 3)


                    # DIMENSIONAL REDUCTIONS (SECOND HIDDEN LAYER)%%%%%

                    Visible_CD_Pure_SECOND_HIDDEN, Explained_Var_Visible_SECOND_HIDDEN, \
                    Explained_Var_ratio_Visible_SECOND_HIDDEN = dim_reduction(Hidden_activations_SECOND_ADVERSARY[-1], 3)


                print('')
                print('------------- DBN PCA--------------')
                print('')
                print(' Explained_Var (Visible): ')

                if VISUAL_CHECK_COMPARISON:

                    print('          - Pure = ' + str(Explained_Var_Visible) + str(' ') + str('(') + str(
                        Explained_Var_ratio_Visible) + str(')'))
                    print('          - Total Variance explained = ', np.sum(Explained_Var_Visible), '%')
                    print('')

                print('          - Biased = ' + str(Explained_Var_Visible_Biased) + str(' ') + str('(') + str(
                    Explained_Var_ratio_Visible_Biased) + str(')'))
                print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_Biased), '%')
                print('')
                print('----------------------------')
                print('')
                print(' Explained_Var (Hidden): ')
                print('')

                if VISUAL_CHECK_COMPARISON:

                    print('          - Pure = ' + str(Explained_Var_Visible_SECOND) + str(' ') + str('(') + str(
                        Explained_Var_ratio_Visible_SECOND) + str(')'))
                    print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_SECOND), '%')
                    print('')

                print('          - Biased = ' + str(Explained_Var_Visible_Biased_SECOND) + str(' ') + str('(') + str(
                    Explained_Var_ratio_Visible_Biased_SECOND) + str(')'))
                print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_Biased_SECOND), '%')
                print('')
                print('----------------------------')
                print('')
                print(' Explained_Var (Second Hidden): ')
                print('')

                if VISUAL_CHECK_COMPARISON:

                    print('          - Pure = ' + str(Explained_Var_Visible_SECOND_HIDDEN) + str(' ') + str('(') + str(
                        Explained_Var_ratio_Visible_SECOND_HIDDEN) + str(')'))
                    print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_SECOND_HIDDEN), '%')
                    print('')

                print('          - Biased = ' + str(Explained_Var_Visible_Biased_SECOND_HIDDEN) + str(' ') + str('(') + str(
                    Explained_Var_ratio_Visible_Biased_SECOND_HIDDEN) + str(')'))
                print('          - Total Variance explained = ', np.sum(Explained_Var_Visible_Biased_SECOND_HIDDEN), '%')
                print('')
                print('----------------------------')

                # 3D PLOTS OF LAYER ACTIVATIONS%%%%%

                Labels_dataset, Labels_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_DBN)

                if VISUAL_CHECK_COMPARISON:

                    # INPUT LAYER %

                    Visible_activation_disentanglement_FIRST_img = Window_builder_layers_activations(Visible_CD_Biased,
                                                                                                     Labels_dataset,
                                                                                                     Labels_batch,
                                                                                                     Legend_lab,
                                                                                                     'input',
                                                                                                     2,
                                                                                                     Visible_CD_Pure
                                                                                                            )


                    Visible_activation_disentanglement_SECOND_img = Window_builder_layers_activations(
                                                                                                    Visible_CD_Biased_SECOND,
                                                                                                    Labels_dataset,
                                                                                                     Labels_batch,
                                                                                                     Legend_lab,
                                                                                                     'FHL',
                                                                                                     2,
                                                                                                     Visible_CD_Pure_SECOND
                                                                                                     )





                    Visible_activation_disentanglement_SECOND_HIDDEN_img = Window_builder_layers_activations(
                        Visible_CD_Biased_SECOND_HIDDEN,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'SHL',
                        2, Visible_CD_Pure_SECOND_HIDDEN)

                    # IDEALS PLOTS%%%%%

                    # if Learning_modality_DBN == 'SL' or  Learning_modality_DBN == 'RL':

                        # if Hidden_units_SECOND_DBN == 12:
                        #
                        #     ideals = np.eye(12, dtype=int)
                        #
                        # else:
                        #
                        #     ideals, ideals_batch = load_ideals([Weight_inputs_SECOND])
                        #
                        #
                        # Ideals_img = Window_builder_ideals_recostruction(ideals_batch, Network_weights_SECOND[0],
                        #                                                      Network_weights_SECOND[2],
                        #                                                      Network_weights_FIRST[0],
                        #                                                      Network_weights_FIRST[2])

                else:

                    # INPUT LAYER %


                    Visible_activation_disentanglement_FIRST_img = Window_builder_layers_activations(
                        Visible_CD_Biased,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'input',
                        2,
                        )

                    # FIRST HIDDEN LAYER %


                    Visible_activation_disentanglement_SECOND_img = Window_builder_layers_activations(
                        Visible_CD_Biased_SECOND,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'FHL',
                        2,
                                            )

                    # SECOND HIDDEN LAYER %


                    Visible_activation_disentanglement_SECOND_HIDDEN_img = Window_builder_layers_activations(
                        Visible_CD_Biased_SECOND_HIDDEN,
                        Labels_dataset,
                        Labels_batch,
                        Legend_lab,
                        'SHL',
                        2,
                        )

                plt.show()

                if VISUAL_CHECK_EACH_INPUT:

                    for input in range(0, dataset.shape[0]):

                        hiddens_RF_to_plots = 10

                        Specific_original_input = dataset[input, :]

                        Specific_reconstructed_input = Inputs_reconstructions_FIRST[-1][input, :]
                        Specific_hidden_activation = Hidden_activations_FIRST[-1][input, :]

                        Specific_rec_error = (Rec_errors_FIRST[-1][0][input])
                        Specific_hidden_activation_second = Hidden_activations_SECOND[-1] [input, :]

                        Input_spread_reconstruction_RBM_W = Window_builder_single_input_recostruction(
                            Specific_original_input, Specific_reconstructed_input,
                            Specific_hidden_activation, Weight_inputs_FIRST,
                            Weights_bias_hiddens_to_inputs_FIRST, Specific_rec_error,
                            hiddens_RF_to_plots, Specific_hidden_activation_second,
                            Weight_inputs_SECOND, Weights_bias_hiddens_to_inputs_SECOND)

                        print('')
                        print('------------------------')
                        print('')
                        print(' *** DBN single input spread/reconstruction (input <-> first hidden <-> second hidden) *** ')
                        print('')
                        print('------------------------')

                        print(str(" Image n ") + str(input) + str(' ----------------------------'))
                        print('')
                        print(" Rec_Err (L1 norm) = ", Specific_rec_error)
                        print('')
                        print('----------------------------')

                        plt.show()

    if TEST_REPRESENTATIONS:

        Save_Folders_Paths = '.\\Tester_data\\Tested (R-CD)\\'

        Save_Folders_Paths_ADVERSARY = '.\\Tester_data\\Adversary (CD)\\'


        if Env_to_test == 'basic':

            load_weights_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

            load_weights_path_adversary = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


        elif Env_to_test == 'robotic':

            load_weights_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'

            load_weights_path_adversary = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'


        dataset = Load_images(Input_units_FIRST)

        if TEST_REPRESENTATIONS_RBM:

            Save_Folders_Paths += '.\\RBM\\'

            Save_Folders_Paths_ADVERSARY += '.\\RBM\\'


            clean_training_data_folder(Save_Folders_Paths)

            clean_training_data_folder(Save_Folders_Paths_ADVERSARY)



            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST,
                                                                load_weights_path)

            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            Input_FIRST, Hidden_activations_FIRST, Inputs_reconstructions_FIRST, \
            Rec_errors_FIRST = RBM_DBN_Activation(dataset, 1, Network_weights_FIRST,
                                                  salient_feature_FIRST)

            if TEST_REPRESENTATIONS_ADVERSARY:

                Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
                Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST,
                                                                    load_weights_path_adversary)

                Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                         Weights_bias_hiddens_to_inputs_FIRST]

                Input_FIRST_ADVERSARY, Hidden_activations_FIRST_ADVERSARY, Inputs_reconstructions_FIRST_ADVERSARY, \
                Rec_errors_FIRST_ADVERSARY = RBM_DBN_Activation(dataset, 1, Network_weights_FIRST,
                                                      salient_feature_FIRST)

            # VISIBLE REPRESENTATIONS TEST %%%

            if TEST_REPRESENTATIONS_VISIBLE:

                Utility_test(Inputs_reconstructions_FIRST[-1], Tester_Learning_Rule, 'Visible',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths,  False)

                if TEST_REPRESENTATIONS_ADVERSARY:

                    Utility_test(Inputs_reconstructions_FIRST_ADVERSARY[-1], Tester_Learning_Rule, 'Visible',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths_ADVERSARY,  True)

            # FIRST HIDDEN REPRESENTATIONS TEST %%%

            if TEST_REPRESENTATIONS_FIRST_HIDDEN:


                    Utility_test(Hidden_activations_FIRST[-1],
                                      Tester_Learning_Rule, 'First Hidden',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths,  False)

                    if TEST_REPRESENTATIONS_ADVERSARY:

                        Utility_test(Hidden_activations_FIRST_ADVERSARY[-1],
                                          Tester_Learning_Rule, 'First Hidden',
                                          salient_feature_DBN, size_hidden_tester,
                                          L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths_ADVERSARY,  True)

        if TEST_REPRESENTATIONS_DBN:

            Save_Folders_Paths += '.\\DBN\\'

            Save_Folders_Paths_ADVERSARY += '.\\DBN\\'

            clean_training_data_folder(Save_Folders_Paths)

            clean_training_data_folder(Save_Folders_Paths_ADVERSARY)


            Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
            Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, load_weights_path)

            Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                     Weights_bias_hiddens_to_inputs_FIRST]

            Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
            Weights_bias_hiddens_to_inputs_SECOND = Load_weights(Input_units_SECOND, Hidden_units_SECOND, load_weights_path)

            Network_weights_SECOND = [Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND,
                                      Weights_bias_hiddens_to_inputs_SECOND]




            Input_FIRST, Hidden_activations_FIRST, Hidden_activations_SECOND, Inputs_reconstructions_FIRST, \
            Inputs_reconstructions_SECOND, Rec_errors_FIRST, Rec_errors_SECOND = RBM_DBN_Activation(dataset,
                                                                                                    1,
                                                                                                    Network_weights_FIRST,
                                                                                                    salient_feature_DBN,
                                                                                                    1,
                                                                                                    Network_weights_SECOND)


            if TEST_REPRESENTATIONS_ADVERSARY:


                Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
                Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, load_weights_path_adversary)

                Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                         Weights_bias_hiddens_to_inputs_FIRST]

                Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND, \
                Weights_bias_hiddens_to_inputs_SECOND = Load_weights(Input_units_SECOND,
                                                                     Hidden_units_SECOND, load_weights_path_adversary) # CHANGE HERE

                Network_weights_SECOND = [Weight_inputs_SECOND, Weights_bias_inputs_to_hiddens_SECOND,
                                          Weights_bias_hiddens_to_inputs_SECOND]

                Input_FIRST_ADVERSARY, Hidden_activations_FIRST_ADVERSARY, Hidden_activations_SECOND_ADVERSARY, Inputs_reconstructions_FIRST_ADVERSARY, \
                Inputs_reconstructions_SECOND_ADVERSARY, Rec_errors_FIRST_ADVERSARY, Rec_errors_SECOND_ADVERSARY = RBM_DBN_Activation(dataset,
                                                                                                        1,
                                                                                                        Network_weights_FIRST,
                                                                                                        salient_feature_DBN,
                                                                                                        1,
                                                                                                        Network_weights_SECOND)



            # VISIBLE REPRESENTATIONS TEST %%%

            if TEST_REPRESENTATIONS_VISIBLE:

                    Utility_test(Inputs_reconstructions_FIRST[-1], Tester_Learning_Rule, 'Visible',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths, False)

                    if TEST_REPRESENTATIONS_ADVERSARY:

                        Utility_test(Inputs_reconstructions_FIRST_ADVERSARY[-1], Tester_Learning_Rule, 'Visible',
                                          salient_feature_DBN, size_hidden_tester,
                                          L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths_ADVERSARY, True)


            # FIRST HIDDEN REPRESENTATIONS TEST %%%

            if TEST_REPRESENTATIONS_FIRST_HIDDEN:


                    Utility_test(Inputs_reconstructions_SECOND[-1],
                                      Tester_Learning_Rule, 'First Hidden',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths,  False)

                    if TEST_REPRESENTATIONS_ADVERSARY:

                        Utility_test(Inputs_reconstructions_SECOND_ADVERSARY[-1],
                                          Tester_Learning_Rule, 'First Hidden',
                                          salient_feature_DBN, size_hidden_tester,
                                          L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths_ADVERSARY, True)


            if TEST_REPRESENTATIONS_SECOND_HIDDEN:


                    Utility_test(Hidden_activations_SECOND[-1],
                                      Tester_Learning_Rule, 'Second Hidden',
                                      salient_feature_DBN, size_hidden_tester,
                                      L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths, False)

                    if TEST_REPRESENTATIONS_ADVERSARY:

                        Utility_test(Hidden_activations_SECOND_ADVERSARY[-1],
                                          Tester_Learning_Rule, 'Second Hidden',
                                          salient_feature_DBN, size_hidden_tester,
                                          L_rate_tester, L_rate_tester_critic, tot_epocs_test, Binary_Reward_format_tester, Save_Folders_Paths_ADVERSARY, True)

        plt.show()

    if SHOW_TEST_REPRESENTATIONS_RESULTS:

        salient_feature_tester = copy.deepcopy(salient_feature_DBN)

        if SHOW_TEST_REPRESENTATIONS_RESULTS_RBM:

            Save_Folders_Paths = '.\\Tester_data\\Tested (R-CD)\\RBM\\'

            Save_Folders_Paths_ADVERSARY = '.\\Tester_data\\Adversary (CD)\\RBM\\'

        if SHOW_TEST_REPRESENTATIONS_RESULTS_DBN:

            Save_Folders_Paths = '.\\Tester_data\\Tested (R-CD)\\DBN\\'

            Save_Folders_Paths_ADVERSARY = '.\\Tester_data\\Adversary (CD)\\DBN\\'

        if SHOW_TEST_REPRESENTATIONS_RESULTS_VISIBLE:

            size_input_tester = 28 * 28 * 3


            if not ADVERSARY_TESTER_COMPARISON:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester,
                              L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Rewards_Accuracies_Network = Window_builder_tester_performances_REINFORCE('visible',
                                                                                              salient_feature_tester,
                                                                                              Rewards,
                                                                                              Accuracies)

                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Losses_Network = Window_builder_tester_performances(Errors,'visible', salient_feature_tester)





            else:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester, L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Accuracies_ADVERSARY, Rewards_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)

                    Accuracies_both_sys = np.vstack((Accuracies_ADVERSARY, Accuracies))

                    Rewards_both_sys = np.vstack((Rewards_ADVERSARY, Rewards))

                    Tester_img_VISIBLE = Window_builder_tester_performances_REINFORCE_DOUBLE_COMPARISON('visible',
                                                                                      salient_feature_tester,
                                                                                      Rewards_both_sys.T,
                                                                                      Accuracies_both_sys.T)



                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Errors_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)

                    Errors_both_sys = np.hstack((Errors_ADVERSARY, Errors))

                    Tester_img_VISIBLE = Window_builder_tester_performances_DOUBLE_COMPARISON(Errors_both_sys, 'visible', salient_feature_tester)

        if SHOW_TEST_REPRESENTATIONS_RESULTS_FIRST_HIDDEN:

            size_input_tester = copy.deepcopy(Input_units_SECOND_DBN)

            if not ADVERSARY_TESTER_COMPARISON:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester,
                              L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Rewards_Accuracies_Network = Window_builder_tester_performances_REINFORCE('hidden',
                                                                                              salient_feature_tester,
                                                                                              Rewards,
                                                                                              Accuracies)

                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Losses_Network = Window_builder_tester_performances(Errors,'hidden', salient_feature_tester)


            else:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester,
                              L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Accuracies_ADVERSARY, Rewards_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)

                    Accuracies_both_sys = np.vstack((Accuracies_ADVERSARY, Accuracies))

                    Rewards_both_sys = np.vstack((Rewards_ADVERSARY, Rewards))

                    Tester_img_HIDDEN = Window_builder_tester_performances_REINFORCE_DOUBLE_COMPARISON('hidden',
                                                                                      salient_feature_tester,
                                                                                      Rewards_both_sys.T,
                                                                                      Accuracies_both_sys.T)

                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Errors_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)

                    Errors_both_sys = np.hstack((Errors_ADVERSARY, Errors))

                    Tester_img_HIDDEN = Window_builder_tester_performances_DOUBLE_COMPARISON(Errors_both_sys, 'hidden',
                                                                            salient_feature_tester)

        if SHOW_TEST_REPRESENTATIONS_RESULTS_SECOND_HIDDEN:

            size_input_tester = copy.deepcopy(Hidden_units_SECOND_DBN)

            if not ADVERSARY_TESTER_COMPARISON:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester,
                              L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Rewards_Accuracies_Network = Window_builder_tester_performances_REINFORCE('hidden second',
                                                                                              salient_feature_tester,
                                                                                              Rewards,
                                                                                              Accuracies)

                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Losses_Network = Window_builder_tester_performances(Errors,'hidden second', salient_feature_tester)




            else:

                if Tester_Learning_Rule == 'REINFORCE':

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester,
                              L_rate_tester_critic]

                    Accuracies, Rewards, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester, # CHANGE HERE
                              L_rate_tester_critic]

                    Accuracies_ADVERSARY, Rewards_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)


                    Accuracies_both_sys = np.vstack((Accuracies_ADVERSARY, Accuracies))

                    Rewards_both_sys = np.vstack((Rewards_ADVERSARY, Rewards))

                    Tester_img_HIDDEN_SECOND = Window_builder_tester_performances_REINFORCE_DOUBLE_COMPARISON('hidden second',
                                                                                     salient_feature_tester,
                                                                                     Rewards_both_sys.T,
                                                                                     Accuracies_both_sys.T)



                else:

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors, Parameters = tester_load_data(Params, load_path = Save_Folders_Paths)

                    Params = [salient_feature_tester, size_input_tester, size_hidden_tester, L_rate_tester]

                    Errors_ADVERSARY, Parameters_ADVERSARY = tester_load_data(Params, True, load_path = Save_Folders_Paths_ADVERSARY)

                    Errors_both_sys = np.hstack((Errors_ADVERSARY, Errors))

                    Tester_img_HIDDEN_SECOND = Window_builder_tester_performances_DOUBLE_COMPARISON(Errors_both_sys, 'hidden second',
                                                                           salient_feature_tester)

        plt.show()

    if TEST_RESONANCES_GIBBS_SAMPLING:

            dataset = Load_images(Input_units_FIRST)

            if Env_to_test == 'basic':

                load_weights_path = '.\\Weights_layers_activations\\Basic_enviroment\\Tested\\'

                load_weights_path_adversary = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


                if TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY:

                    load_weights_path = '.\\Weights_layers_activations\\Basic_enviroment\\Adversary\\'


            elif Env_to_test == 'robotic':

                load_weights_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Tested (R-CD)\\'

                load_weights_path_adversary = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'

                if TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY:

                    load_weights_path = '.\\Weights_layers_activations\\Robotic_enviroment\\Adversary (CD)\\'


            if TEST_RESONANCES_GIBBS_SAMPLING_RBM:

                Labels_dataset, Labels_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_FIRST)


                Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST, \
                Weights_bias_hiddens_to_inputs_FIRST = Load_weights(Input_units_FIRST, Hidden_units_FIRST, load_weights_path)

                Network_weights_FIRST = [Weight_inputs_FIRST, Weights_bias_inputs_to_hiddens_FIRST,
                                         Weights_bias_hiddens_to_inputs_FIRST]

                print('')
                print('------------------------')
                print('')
                print(' *** RBM dataset spread/reconstruction (input <-> first hidden) *** ')
                print('')
                print('------------------------')
                print('')

                Input_FIRST, Hidden_activations_FIRST, Inputs_reconstructions_FIRST, \
                Rec_errors_FIRST = RBM_DBN_Activation(dataset, 1, Network_weights_FIRST, salient_feature_FIRST)

                for input in range(0, dataset.shape[0]):


                        Input_net = dataset[input, :]
                        Input_net = np.reshape(Input_net, (1, Input_net.shape[0]))

                        print('')
                        print('------------------------')
                        print('')
                        print(' *** RBM single input multiple spreads/reconstructions (input <-> first hidden) *** ')
                        print('')
                        print('------------------------')
                        print('')


                        Single_Input_FIRST, Single_Input_Hidden_activations_FIRST, Single_Input_reconstructions_FIRST, \
                        Single_Rec_errors_FIRST = RBM_DBN_Activation(Input_net, number_of_resonances_FIRST,
                                                                     Network_weights_FIRST, salient_feature_FIRST)



                        # INPUT LAYER REDUCTION VISUALIZATION %

                        Single_Input_reconstructions_FIRST = np.vstack(Single_Input_reconstructions_FIRST)

                        Single_Input_Visible, Single_Input_Explained_Var_Visible, \
                        Single_Input_Explained_Var_ratio_Visible = dim_reduction(Single_Input_reconstructions_FIRST,
                                                                                       3)

                        Datatset_Visible, Datatset_Explained_Var_Visible, \
                        Datatset_Explained_Var_ratio_Visible = dim_reduction(Inputs_reconstructions_FIRST, 3)

                        Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            Datatset_Visible,
                            Labels_dataset,
                            Labels_batch,
                            Legend_lab,
                            Single_Input_Visible,
                            'IL',
                        )


                        # FIRST HIDDEN LAYER REDUCTION VISUALIZATION %

                        Single_Input_Hidden_activations_FIRST = np.vstack(Single_Input_Hidden_activations_FIRST)

                        Single_Input_Hidden, Single_Input_Explained_Var_Hidden, \
                        Single_Input_Explained_Var_ratio_Hidden = dim_reduction(Single_Input_Hidden_activations_FIRST, 3)

                        Datatset_Hidden_, Datatset_Explained_Var_FIRST_Hidden, \
                        Datatset_Explained_Var_ratio_Hidden = dim_reduction(Hidden_activations_FIRST[0], 3)

                        First_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            Datatset_Hidden_,
                            Labels_dataset,
                            Labels_batch,
                            Legend_lab,
                            Single_Input_Hidden,
                            'FHL'
                        )




                        hiddens_RF_to_plots = 10

                        Input_spread_reconstruction_RBM_W = Window_builder_single_input_recostruction(
                            Input_net.T, Single_Input_reconstructions_FIRST[-1].T,
                            Single_Input_Hidden_activations_FIRST[-1], Weight_inputs_FIRST,
                            Weights_bias_hiddens_to_inputs_FIRST, Single_Rec_errors_FIRST[-1][0],
                            hiddens_RF_to_plots)

                        Single_Rec_errors_FIRST_temp = np.array(Single_Rec_errors_FIRST)
                        Single_Rec_errors_FIRST_temp_1 = np.stack(Single_Rec_errors_FIRST_temp[:, 0])
                        Single_Rec_errors_FIRST_temp_2 = np.stack(Single_Rec_errors_FIRST_temp[:, 2])

                        Info_loss_window = Window_builder_Information_Loss_single_input(Single_Rec_errors_FIRST_temp_1)

                        print(str(" Image n ") + str(input) + str(' ----------------------------'))
                        print('')
                        print(" Rec_Err (L1 norm), first sampling step = ", Single_Rec_errors_FIRST[0][0])
                        print('')
                        print(" Rec_Err (L1 norm), last sampling step = ", Single_Rec_errors_FIRST[-1][0])
                        print('')
                        print('----------------------------')

                        plt.show()



            elif TEST_RESONANCES_GIBBS_SAMPLING_DBN:

                Labels_dataset, Labels_batch, Legend_lab = Ideal_actions_initialization(4, salient_feature_DBN)

                Weight_inputs_FIRST_DBN, Weights_bias_inputs_to_hiddens_FIRST_DBN, \
                Weights_bias_hiddens_to_inputs_FIRST_DBN = Load_weights(Input_units_FIRST_DBN, Hidden_units_FIRST_DBN, load_weights_path)

                Network_weights_FIRST_DBN = [Weight_inputs_FIRST_DBN, Weights_bias_inputs_to_hiddens_FIRST_DBN,
                                         Weights_bias_hiddens_to_inputs_FIRST_DBN]

                Weight_inputs_SECOND_DBN, Weights_bias_inputs_to_hiddens_SECOND_DBN, \
                Weights_bias_hiddens_to_inputs_SECOND_DBN = Load_weights(Input_units_SECOND_DBN, Hidden_units_SECOND_DBN, load_weights_path)

                Network_weights_SECOND_DBN = [Weight_inputs_SECOND_DBN, Weights_bias_inputs_to_hiddens_SECOND_DBN,
                                          Weights_bias_hiddens_to_inputs_SECOND_DBN]

                print('')
                print('------------------------')
                print('')
                print(' *** DBN dataset spread/reconstruction (input <-> first hidden) *** ')
                print('')
                print('------------------------')
                print('')

                Input_FIRST, Hidden_activations_FIRST, Hidden_activations_SECOND, Inputs_reconstructions_FIRST, \
                Inputs_reconstructions_SECOND, Rec_errors_FIRST, Rec_errors_SECOND = RBM_DBN_Activation(dataset,
                                                                                                        1,
                                                                                                        Network_weights_FIRST_DBN,
                                                                                                        salient_feature_DBN,
                                                                                                        1,
                                                                                                        Network_weights_SECOND_DBN)

                for input in range(0, dataset.shape[0]):

                        Input_net = dataset[input, :]
                        Input_net = np.reshape(Input_net, (1, Input_net.shape[0]))

                        print('')
                        print('------------------------')
                        print('')
                        print(' *** DBN single input multiple spreads/reconstructions (input <-> first hidden) *** ')
                        print('')
                        print('------------------------')
                        print('')

                        Single_Input_FIRST, Single_Input_Hidden_activations_FIRST, Single_Input_Hidden_activations_SECOND, \
                        Single_Input_reconstructions_FIRST, Single_Input_reconstructions_SECOND, \
                        Single_Rec_errors_FIRST, Single_Rec_errors_SECOND = RBM_DBN_Activation(Input_net,
                                                                                               number_of_resonances_FIRST_DBN,
                                                                                               Network_weights_FIRST_DBN,
                                                                                               salient_feature_DBN,
                                                                                               number_of_resonances_SECOND_DBN,
                                                                                               Network_weights_SECOND_DBN,
                                                                                               number_of_resonances_WHOLE_DBN)

                        hiddens_RF_to_plots = 10

                        if number_of_resonances_WHOLE_DBN != 0:

                            # INPUT LAYER REDUCTION VISUALIZATION %%%%%%

                            Single_Input_reconstructions_FIRST = np.vstack(Single_Input_reconstructions_FIRST)
                            Single_Input_Visible_FIRST, Single_Input_Explained_Var_FIRST_Visible, \
                            Single_Input_Explained_Var_ratio_Visible_FIRST = dim_reduction(Single_Input_reconstructions_FIRST,
                                                                                           3)

                            Datatset_Visible_FIRST_red, Datatset_Explained_Var_FIRST_Visible, \
                            Datatset_Explained_Var_ratio_Visible_FIRST = dim_reduction(Inputs_reconstructions_FIRST[0], 3)

                            First_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Visible_FIRST_red,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Visible_FIRST,
                                'IL',
                                )

                            # First_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Visible_FIRST,
                            #     Datatset_Visible_FIRST_red, 'IL',
                            #     salient_feature_DBN)

                            # FIRST HIDDEN LAYER REDUCTION VISUALIZATION %%%%%%

                            Single_Input_Hidden_activations_FIRST = np.vstack(Single_Input_Hidden_activations_FIRST)
                            Single_Input_Hidden_FIRST, Single_Input_Explained_Var_FIRST_Hidden, \
                            Single_Input_Explained_Var_ratio_Hidden_FIRST = dim_reduction(Single_Input_Hidden_activations_FIRST,
                                                                                          3)


                            Datatset_Hidden_FIRST, Datatset_Explained_Var_FIRST_Hidden, \
                            Datatset_Explained_Var_ratio_Hidden_FIRST = dim_reduction(Hidden_activations_FIRST[0], 3)

                            First_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Hidden_FIRST,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Hidden_FIRST,
                                'FHL'
                                )

                            # First_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Hidden_FIRST,
                            #     Datatset_Hidden_FIRST, 'FHL',
                            #     salient_feature_DBN)

                            # FIRST HIDDEN LAYER (RECONSTRUCTED FROM SECOND HIDDEN LAYER) REDUCTION VISUALIZATION %%%%%%

                            Single_Input_reconstructions_SECOND = np.vstack(Single_Input_reconstructions_SECOND)

                            Single_Input_Visible_SECOND, Single_Input_Explained_Var_SECOND_Visible, \
                            Single_Input_Explained_Var_ratio_Visible_SECOND = dim_reduction(
                                Single_Input_reconstructions_SECOND,
                                3)

                            Datatset_Visible_SECOND, Datatset_Explained_Var_SECOND_Visible, \
                            Datatset_Explained_Var_ratio_Visible_SECOND = dim_reduction(Inputs_reconstructions_SECOND[0], 3)

                            Second_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Visible_SECOND,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Visible_SECOND,
                                'FHLR'
                                )

                            # Second_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Visible_SECOND,
                            #     Datatset_Visible_SECOND, 'FHLR',
                            #     salient_feature_DBN)

                            # SECOND HIDDEN LAYER REDUCTION VISUALIZATION %%%%%%

                            Single_Input_Hidden_activations_SECOND = np.vstack(Single_Input_Hidden_activations_SECOND)

                            Single_Input_Hidden_SECOND, Single_Input_Explained_Var_SECOND_Hidden, \
                            Single_Input_Explained_Var_ratio_Hidden_SECOND = dim_reduction(
                                Single_Input_Hidden_activations_SECOND,
                                3)

                            Datatset_Hidden_SECOND, Datatset_Explained_Var_SECOND_Hidden, \
                            Datatset_Explained_Var_ratio_Hidden_SECOND = dim_reduction(Hidden_activations_SECOND[0], 3)

                            Second_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Hidden_SECOND,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Hidden_SECOND,
                                'SHL'
                                )
                            # Second_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Hidden_SECOND,
                            #     Datatset_Hidden_SECOND, 'SHL',
                            #     salient_feature_DBN)

                            Input_spread_reconstruction_DBN_W = Window_builder_single_input_recostruction(
                                Input_net.T, Single_Input_reconstructions_FIRST[-1].T,
                                Single_Input_reconstructions_SECOND[-1].T, Weight_inputs_FIRST_DBN,
                                Weights_bias_hiddens_to_inputs_FIRST_DBN, Single_Rec_errors_FIRST[-1][0],
                                hiddens_RF_to_plots, Single_Input_Hidden_activations_SECOND[-1], Weight_inputs_SECOND_DBN,
                                Weights_bias_hiddens_to_inputs_SECOND_DBN)

                        else:

                            # FIRST HIDDEN LAYER (RECONSTRUCTED FROM SECOND HIDDEN LAYER) REDUCTION VISUALIZATION %%%%%%

                            Single_Input_reconstructions_SECOND = np.vstack(Single_Input_reconstructions_SECOND)

                            Single_Input_Visible_SECOND, Single_Input_Explained_Var_SECOND_Visible, \
                            Single_Input_Explained_Var_ratio_Visible_SECOND = dim_reduction(
                                Single_Input_reconstructions_SECOND,
                                3)

                            Datatset_Visible_SECOND, Datatset_Explained_Var_SECOND_Visible, \
                            Datatset_Explained_Var_ratio_Visible_SECOND = dim_reduction(
                                Inputs_reconstructions_SECOND[0], 3)

                            Second_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Visible_SECOND,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Visible_SECOND,
                                'FHLR',
                                )

                            # Second_Visible_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Visible_SECOND,
                            #     Datatset_Visible_SECOND, 'FHLR',
                            #     salient_feature_DBN)

                            # SECOND HIDDEN LAYER REDUCTION VISUALIZATION %%%%%%

                            Single_Input_Hidden_activations_SECOND = np.vstack(Single_Input_Hidden_activations_SECOND)

                            Single_Input_Hidden_SECOND, Single_Input_Explained_Var_SECOND_Hidden, \
                            Single_Input_Explained_Var_ratio_Hidden_SECOND = dim_reduction(
                                Single_Input_Hidden_activations_SECOND,
                                3)

                            Datatset_Hidden_SECOND, Datatset_Explained_Var_SECOND_Hidden, \
                            Datatset_Explained_Var_ratio_Hidden_SECOND = dim_reduction(Hidden_activations_SECOND[0], 3)

                            Second_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                                Datatset_Hidden_SECOND,
                                Labels_dataset,
                                Labels_batch,
                                Legend_lab,
                                Single_Input_Hidden_SECOND,
                                'SHL',
                                )

                            # Second_Hidden_Gibbs_Sampling_window = Window_builder_layers_activations_resonances_Gibbs_sampling(
                            #     Single_Input_Hidden_SECOND,
                            #     Datatset_Hidden_SECOND, 'SHL',
                            #     salient_feature_DBN)

                            Input_spread_reconstruction_DBN_W = Window_builder_single_input_recostruction(
                                Input_net.T, Single_Input_reconstructions_FIRST[-1].T,
                                Single_Input_reconstructions_SECOND[-1].T, Weight_inputs_FIRST_DBN,
                                Weights_bias_hiddens_to_inputs_FIRST_DBN, Single_Rec_errors_FIRST[-1][0],
                                hiddens_RF_to_plots, Single_Input_Hidden_activations_SECOND[-1], Weight_inputs_SECOND_DBN,
                                Weights_bias_hiddens_to_inputs_SECOND_DBN)


                        Single_Rec_errors_FIRST_temp = np.array(Single_Rec_errors_FIRST)
                        Single_Rec_errors_FIRST_temp_1 = np.stack(Single_Rec_errors_FIRST_temp[:, 0])
                        Single_Rec_errors_FIRST_temp_2 = np.stack(Single_Rec_errors_FIRST_temp[:, 2])

                        Single_Rec_errors_SECOND_temp = np.array(Single_Rec_errors_SECOND)
                        Single_Rec_errors_SECOND_temp_1 = np.stack(Single_Rec_errors_SECOND_temp[:, 0])
                        Single_Rec_errors_SECOND_temp_2 = np.stack(Single_Rec_errors_SECOND_temp[:, 2])

                        Info_loss_window = Window_builder_Information_Loss_single_input(Single_Rec_errors_FIRST_temp_1,
                                                                                        Single_Rec_errors_SECOND_temp_1)

                        print(str(" Image n ") + str(input) + str(' ----------------------------'))
                        print('')
                        print(" Rec_Err (L1 norm), first sampling step = ", Single_Rec_errors_FIRST[0][0])
                        print('')
                        print(" Rec_Err (L1 norm), last sampling step = ", Single_Rec_errors_FIRST[-1][0])
                        print('')
                        print('----------------------------')

                        plt.show()

            plt.show()

    return '\n____________MAIN COMPUTATIONS FINISHED____________'



'''

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CHECK: 
FIX:
UPDATE:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''

# ------------------------------ BASIC ENVIROMENT: RBM (NO OTHER COMPONENTS) ------------------------------

# %%%%%%%%%%%%% FLAGS TO TRAIN THE RBM NETWORKS WITHOUT ANY OTHER COMPONENT %%%%%%%%%%%%%

TRAIN_BASIC_ENV = False
TRAIN_BASIC_ENV_ADVERSARY = False

FIRST_RBM_TRAINING = False
SECOND_RBM_TRAINING = False

# %%%%%%%%%%%%% FLAGS TO SHOW POST-TRAINING RESULTS OF TRAIN WITHOUT ANY OTHER COMPONENT %%%%%%%%%%%%%

SHOW_TRAINING_RESULTS_BASIC_ENV = False
SHOW_TRAINING_RESULTS_BASIC_ENV_ADVERSARY = False

SHOW_TRAINING_SINGLE_RBM_FIRST = False
SHOW_TRAINING_SINGLE_RBM_SECOND = False

# ------------------------------ ROBOTIC ENVIROMENT: NEURO-ROBOTIC ARCHITECTURE ------------------------------

# %%%%%%%%%%%%% FLAGS TO TRAIN A NEURO-ROBOTIC ARCHITECTURE %%%%%%%%%%%%%

TRAIN_ROBOTIC_ENV = False
TRAIN_ROBOTIC_ENV_ADVERSARY = False

# %%%%%%%%%%%%% FLAGS TO SHOW POST-TRAINING RESULTS OF ROBOTIC ENVIROMENT  %%%%%%%%%%%%%

SHOW_TRAINING_RESULTS_ROBOTIC_ENV = False
SHOW_TRAINING_RESULTS_ROBOTIC_ENV_ADVERSARY = False

ADVERSARY_COMPARISON = False

# ------------------------------ ANALYSIS ENVIROMENT: INTERNAL PROCESSES OF RBM/DBN (REPRESENTATIONS, GIBBS SAMPLING, UTILITY TEST..) ------------------------------

TEST_GENERATIVE_MODEL = False
TEST_GENERATIVE_MODEL_ADVERSARY = False

TEST_RBM = False
TEST_DBN = False

#  %%% VISUAL CHECK

VISUAL_CHECK = False
VISUAL_CHECK_COMPARISON = False

VISUAL_CHECK_EACH_INPUT = False

#  %%% REPRESENTATIONS UTILITY TEST ------------------------------------------------------------------------------

TEST_REPRESENTATIONS = False
TEST_REPRESENTATIONS_ADVERSARY = False

TEST_REPRESENTATIONS_RBM  = False
TEST_REPRESENTATIONS_DBN =  False


TEST_REPRESENTATIONS_VISIBLE = False
TEST_REPRESENTATIONS_FIRST_HIDDEN = False
TEST_REPRESENTATIONS_SECOND_HIDDEN = False

# %%%%%%%%%%%%% FLAGS TO SHOW POST-TRAINING RESULTS OF UTILITY TEST (PRETRAINED RBM + PERCEPTRON) %%%%%%%%%%%%%

SHOW_TEST_REPRESENTATIONS_RESULTS = False
ADVERSARY_TESTER_COMPARISON = False

SHOW_TEST_REPRESENTATIONS_RESULTS_RBM = False
SHOW_TEST_REPRESENTATIONS_RESULTS_DBN = False

SHOW_TEST_REPRESENTATIONS_RESULTS_VISIBLE = False
SHOW_TEST_REPRESENTATIONS_RESULTS_FIRST_HIDDEN = False
SHOW_TEST_REPRESENTATIONS_RESULTS_SECOND_HIDDEN = False

#  %%% RESONANCES/GIBBS SAMPLING  TEST ---------------------------------------------------------------------------

TEST_RESONANCES_GIBBS_SAMPLING = False
TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY = False

TEST_RESONANCES_GIBBS_SAMPLING_RBM = False
TEST_RESONANCES_GIBBS_SAMPLING_DBN  = False


# %%%%%%%%%%%%% FLAGS/PARAMETERS/VARIABLES OF MAIN SECTIONS %%%%%%%%%%%%%

# ********************************************** TRAINING PARAMETERS (BASIC ENVIROMENT AND ROBOTIC ENVIROMENT) **********************************************

# FIRST RBM: TRAINING ------------

#  %%% VISUALIZATION FUNCTIONS

Graphic_on_FIRST = True          #   SHOW (SAVE) GRAPHICAL OUTPUTS

Prints_on_FIRST = True           #   SHOW CONSOLE OUTPUTS (PRINTS)

#  %%% LEARNING FEATURES

save_choice_FIRST = True         #   SAVE WEIGHTS CHOICE

Batch_on_FIRST = True            #   STOCHASTIC/BATCH LEARNING MODE

salient_feature_FIRST = 'none'   #   IN CASE OF SL OR RL: FEATURE TO FOCUS

#  %%% LEARNING PARAMETERS

Tot_epocs_FIRST = 10000        #   NUMBER OF TOTAL EPOCHS

number_of_resonances_FIRST = 1   #   NUMBER OF RESONANCES (SAMPLING STEPS) OF NET (1 FOR NO GIBBS SAMPLING AND MINIMUM 4 FOR SAMPLING)

learning_rate_FIRST = 0.01       #   LR OF CONTRASTIVE DIVERGENCE (CD)

alfa_FIRST = 0.9                 #   MOMENTUM

target_sparsety_FIRST = 1        #   SPARSETY PARAMETER (EG. 0.1 -> MAX 10% OF ACTIVATED UNITS DURING A SINGLE SPREAD)

target_min_error_FIRST = 0.001   #   TARGET RECONSTRUCTION ERROR

# FIRST RBM: STRUCTURE ---

Input_units_FIRST = 28*28*3      #   INPUT UNITS

Hidden_units_FIRST = 200       #   HIDDEN UNITS

# SECOND RBM: TRAINING ------------

#  %%% VISUALIZATION FUNCTIONS

Graphic_on_SECOND = True          #   SHOW (SAVE) GRAPHICAL OUTPUTS

Prints_on_SECOND = True           #   SHOW CONSOLE OUTPUTS (PRINTS)

#  %%% LEARNING FEATURES

Batch_on_SECOND = True           #   STOCHASTIC/BATCH LEARNING MODE

save_choice_SECOND = True         #   SAVE WEIGHTS CHOICE

salient_feature_SECOND = 'none'   #   IN CASE OF SL OR RL: FEATURE TO FOCUS

#  %%% LEARNING PARAMETERS

Tot_epocs_SECOND = 50000        #   NUMBER OF TOTAL EPOCHS

number_of_resonances_SECOND = 1   #   NUMBER OF RESONANCES (SAMPLING STEPS) OF NET (1 FOR NO GIBBS SAMPLING AND MINIMUM 4 FOR SAMPLING)

learning_rate_SECOND = 0.001     #   LR OF CONTRASTIVE DIVERGENCE (CD)

Learning_rate_SECOND_RL_Contribution = 0.01

alfa_SECOND = 0.9              #   MOMENTUM

target_sparsety_SECOND = 1        #   SPARSETY PARAMETER (EG. 0.1 -> MAX 10% OF ACTIVATED UNITS DURING A SINGLE SPREAD)

target_min_error_SECOND = 0.01     #   TARGET RECONSTRUCTION ERROR

## SECOND RBM: STRUCTURE ---

Input_units_SECOND = copy.deepcopy(Hidden_units_FIRST) #   INPUT UNITS

Hidden_units_SECOND = 10         #   HIDDEN UNITS


## WHOLE DBN: STRUCTURE AND OTHER PARAMETERS ---

salient_feature_DBN = 'none'      #   IN CASE OF SL OR RL: FEATURE TO FOCUS

#   NUMBER OF RESONANCES (SAMPLING STEPS) OF NET (1 FOR NO GIBBS SAMPLING AND MINIMUM 4 FOR SAMPLING)
number_of_resonances_FIRST_DBN = 50     #    FIRST RBM
number_of_resonances_SECOND_DBN = 50   #    SECOND RBM
number_of_resonances_WHOLE_DBN = 0    #    WHOLE DBN


#  %%% WHOLE DBN: STRUCTURE ------------

Input_units_FIRST_DBN = 28*28*3   #   INPUT UNITS

Hidden_units_FIRST_DBN = copy.deepcopy(Hidden_units_FIRST)     #   HIDDEN UNITS (FIRST LAYER)

Input_units_SECOND_DBN = copy.deepcopy(Hidden_units_FIRST_DBN) #   HIDDEN UNITS (FIRST LAYER)

Hidden_units_SECOND_DBN = copy.deepcopy(Hidden_units_SECOND)   #   HIDDEN UNITS (SECOND LAYER)

# EXECUTOR (ONLY ROBOTIC ENV.): TRAINING ------------

Binary_Reward_format_robotic_env = False

CD_weight_robotic_env = 0.001 # False or float

L_rate_executor = 0.01

L_rate_executor_critic = 0.001

tot_epocs_robotic_env = 30000


# EXECUTOR (ONLY ROBOTIC ENV.): STRUCTURE ------------

Executor_deep = False

Executor_Input = 'second hidden'

Executor_size_Output = 10





# ********************************************** NETWORK TEST PARAMETERS **********************************************

Env_to_test = 'robotic' # robotic or basic

# REPRESENTATIONS TESTER PARAMETERS **********************************************
# %%% TESTER: TRAIN, STRUCTURE ---
Binary_Reward_format_tester = False

Tester_Learning_Rule = 'BACKPROP'  # REINFORCE or BACKPROP

L_rate_tester = 0.01
L_rate_tester_critic = 0.01
tot_epocs_test = 5000

# %%% TESTER: STRUCTURE ---

size_hidden_tester = 10


_ = MAIN(TRAIN_BASIC_ENV,
         TRAIN_BASIC_ENV_ADVERSARY,

         FIRST_RBM_TRAINING,
         SECOND_RBM_TRAINING,

         SHOW_TRAINING_RESULTS_BASIC_ENV,
         SHOW_TRAINING_RESULTS_BASIC_ENV_ADVERSARY,

         SHOW_TRAINING_SINGLE_RBM_FIRST,
         SHOW_TRAINING_SINGLE_RBM_SECOND,

         TRAIN_ROBOTIC_ENV,
         TRAIN_ROBOTIC_ENV_ADVERSARY,

         SHOW_TRAINING_RESULTS_ROBOTIC_ENV,
         SHOW_TRAINING_RESULTS_ROBOTIC_ENV_ADVERSARY,
         ADVERSARY_COMPARISON,

         TEST_GENERATIVE_MODEL,
         TEST_GENERATIVE_MODEL_ADVERSARY,

         TEST_DBN,
         TEST_RBM,

         VISUAL_CHECK,
         VISUAL_CHECK_COMPARISON,
         VISUAL_CHECK_EACH_INPUT,

         TEST_REPRESENTATIONS,
         TEST_REPRESENTATIONS_ADVERSARY,

         TEST_REPRESENTATIONS_VISIBLE,
         TEST_REPRESENTATIONS_FIRST_HIDDEN,
         TEST_REPRESENTATIONS_SECOND_HIDDEN,

         SHOW_TEST_REPRESENTATIONS_RESULTS,
         ADVERSARY_TESTER_COMPARISON,

         SHOW_TEST_REPRESENTATIONS_RESULTS_RBM,
         SHOW_TEST_REPRESENTATIONS_RESULTS_DBN,

         SHOW_TEST_REPRESENTATIONS_RESULTS_VISIBLE,
         SHOW_TEST_REPRESENTATIONS_RESULTS_FIRST_HIDDEN,
         SHOW_TEST_REPRESENTATIONS_RESULTS_SECOND_HIDDEN,

         TEST_RESONANCES_GIBBS_SAMPLING,
         TEST_RESONANCES_GIBBS_SAMPLING_ADVERSARY,

         TEST_RESONANCES_GIBBS_SAMPLING_RBM,
         TEST_RESONANCES_GIBBS_SAMPLING_DBN,


         salient_feature_DBN,
         Input_units_FIRST_DBN,
         Hidden_units_FIRST_DBN,
         Input_units_SECOND_DBN,
         Hidden_units_SECOND_DBN,

         number_of_resonances_FIRST_DBN,
         number_of_resonances_SECOND_DBN,
         number_of_resonances_WHOLE_DBN,


         Graphic_on_FIRST,
         Prints_on_FIRST,
         Batch_on_FIRST, salient_feature_FIRST, Input_units_FIRST, Hidden_units_FIRST,
         Tot_epocs_FIRST,
         number_of_resonances_FIRST, learning_rate_FIRST, alfa_FIRST,
         target_sparsety_FIRST,
         target_min_error_FIRST, save_choice_FIRST,

         Graphic_on_SECOND,
         Prints_on_SECOND,
         Batch_on_SECOND,

         salient_feature_SECOND,
         Input_units_SECOND,
         Hidden_units_SECOND,
         Tot_epocs_SECOND,

         number_of_resonances_SECOND, learning_rate_SECOND, alfa_SECOND, target_sparsety_SECOND,
         target_min_error_SECOND,
         save_choice_SECOND,

         Env_to_test,
         Executor_deep,
         Executor_Input,
         Executor_size_Output,

         Learning_rate_SECOND_RL_Contribution,
         CD_weight_robotic_env,
         L_rate_executor,
         L_rate_executor_critic,
         tot_epocs_robotic_env,
         Tester_Learning_Rule,
         size_hidden_tester,
         L_rate_tester,
         L_rate_tester_critic,
         Binary_Reward_format_tester,
         tot_epocs_test
         )

print(_)



