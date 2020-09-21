# Flexible-goal-directed-behaviour-and-representations-learning

This project investigates the relation between representation learning, i.e. how the brain of an embodied agent can acquire efficient perceptual representationsof the world, and flexible goal-directed behaviour.

In particular this repository contains the code of GEMMA (Generativity-based Embodied Manipulative Architecture), a neurorobotic architecture used here as a computational model of the brain to approach the research topic and study this phenomenon. The model is able to develop an action-depending perception (i.e. to develop a perceptual representation of salient features of the world) adapt to execute a "goal-directed action", trough the exploitation of a novel learning rule that hibiridates an unsupervised algorythm (Contrastive Divergence; Hinton 2006) and a reinforcement learning algorythm (REINFORCE; Williams, 1995). The architecture is composed by many neuro-inspired components (see figure), such as:

- A sensory component, formed by a Deep Belief Network (a stack of two Restricted Boltzmann Machines; Hinton, 2006; Hinton, 2012), that learns to extract the visual regularities in the world and to execute a dimensional reduction, also influenced by the efficacy of the agent's actions in the world (reward).

- A controller, formed by a RL-based perceptron, that learns to execute an action depending on the manipulated world state that receives from the sensory component and the feedback obtained from the world (reward)

- An evaluator component, formed by a multi-layer perceptron, that learns to predict the reward that the agent will obtain depending on a specific state of the world.

- A goal-monitoring component, formed by a simple function that executes an euclidean distance, that computes the reward depending on a distance between the goal action and the executed action.

Despite for now the architecture acts in a virtual world, it shows an embodied nature because has all components that allow its to interact with a virtual enviroment and to change its perception depending on these interactions. Moreover GEMMA is "ready to be linked" to a a Kuka robot with few tecnical adaptations.

The repository is organized in this way:

