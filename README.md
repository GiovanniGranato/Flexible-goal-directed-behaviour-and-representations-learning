# Flexible-goal-directed-behaviour-and-representations-learning

This project investigates the relation between representation learning, i.e. how the brain of an embodied agent can acquire efficient perceptual representationsof the world, and flexible goal-directed behaviour.

In particular this repository contains the code of GEMMA (Generativity-based Embodied Manipulative Architecture), a neurorobotic architecture used here as a computational model of the brain to approach the research topic and study this phenomenon. The model is able to develop an action-depending perception (i.e. to develop a perceptual representation of salient features of the world) adapt to execute a "goal-directed action", trough the exploitation a deep generative model trained with a novel learning rule that hibridates an unsupervised algorythm (Contrastive Divergence; Hinton 2006) and a reinforcement learning algorythm (REINFORCE; Williams, 1995). The architecture is composed by many neuro-inspired components (see figure), such as:

- A sensory component, formed by a generative model (a Deep Belief Network formed by two stacked Restricted Boltzmann Machines; Hinton, 2006; Hinton, 2012), that learns to extract the visual regularities in the world and to execute a dimensional reduction, also influenced by the efficacy of the agent's actions in the world (reward).

- A controller, formed by a RL-based perceptron, that learns to execute an action depending on the manipulated world state that receives from the sensory component and the feedback obtained from the world (reward)

- An evaluator component, formed by a multi-layer perceptron, that learns to predict the reward that the agent will obtain depending on a specific state of the world.

- A goal-monitoring component, formed by a simple function that executes an euclidean distance, that computes the reward depending on a distance between the goal action and the executed action.

Despite for now the architecture acts in a virtual world, it shows an embodied nature because has all components that allow its to interact with a virtual enviroment and to change its perception depending on these interactions, and then improving its interactions with the world. Moreover GEMMA is "ready to be linked" to a a Kuka robot with few tecnical adaptations.

The repository is organized in this way:

-- Main:                                   # Main file to run the training enviroments and to execute many tests on the generative model (see the documentation of Main function)

-- Environments                         # Training enviroments (training of a single RBM, training of GEMMA, utility test of internal representations)

-- System_Components                    # Activation and training functions of the generative model and GEMMA

-- Basic_functions                      # Basic functions such as save/load functions, weights_initialization, layers activations etc

-- Weights_layers_activations           # Folder that contains the saved weights of the generative model

-- Training_Visual_Outputs              # Folder that contains the visual output of the training that is taking place (learning curves, rewards, internal represetnations etc)

-- Training_data                        # Folder that contains the data of training of generative model (without other components) or the GEMMA

-- Tester_data                          # Folder that contains the utility test data (see documentation of the function "utility test" in "Enviroments.py")

![alt text] (https://github.com/GiovanniGranato/Flexible-goal-directed-behaviour-and-representations-learning/blob/master/GEMMA.jpg?raw=true)



