.. _reinforcement_learning:

======================
Reinforcement Learning
======================

In machine learning, supervised is sometimes contrasted with unsupervised learning. This is a useful distinction, but there are some problem domains that have share characteristics with each without fitting exactly in either category. In cases where the algorithm does not have explicit labels but does receive a form of feedback, we are dealing with a third and distinct paradigm of machine learning - reinforcement learning.

There are different problem types and algorithms, but all reinforcement learning problems have the following aspects in common:

  * an **agent** - the algorithm or "AI" responsible for making decisions

<<<<<<< HEAD
  * an **environment**, consisting of different **states** in which the agent may find itself
=======
Programmatic and a theoretical introduction to reinforcement learning:https://spinningup.openai.com/

>>>>>>> upstream/master

  * a **reward** signal which is returned by the environment as a function of the current state

  * **actions**, each of which takes the agent from one state to another

  * a **policy**, i.e. a mapping from states to actions that defines the agent's behavior

The goal of reinforcement learning is to learn the optimal policy, that is the policy that maximizes expected (discounted) cumulative reward. 

Many RL algorithms will include a value function or a Q-function. A value function gives the expected cumulative reward for each state under the current policy In other words, it answers the question, "If I begin in state :math:`i` and follow my policy, what will be my expected reward?"

In most algorithms, expected cumulative reward is discounted by some factor :math:`\gamma \in (0, 1)`; a typical value for :math:`\gamma` is 0.9. In addition to more accurately modeling the behavior of humans and other animals, :math:`\gamma < 1` helps to ensure that algorithms converge even when there is no terminal state or when the terminal state is never found (because otherwise expected cumulative reward may also become infinite).

Note on Terminology
-------------------

For mostly historical reasons, engineering and operations research use different words to talk about the same concepts. For example, the general field of reinforcement learning itself is sometimes referred to as optimal control, approximate dynamic programming, or neuro-dynamic programming.\ :sup:`1`

Eploration vs. Exploitation
---------------------------

One dilemma inherent to the RL problem setting is the tension between the desire to choose the best known option and the need to try something new in order to discover other options that may be even better. Choosing the best known action is known as exploitation, while choosing a different action is known as exploration. 

Typically, this is solved by adding to the policy a small probability of exploration. For example, the policy could be to choose the optimal action (optimal with regard to what is known) with probability 0.95, and exploring by randomly choosing some other action with probability 0.5 (if uniform across all remaining actions: probability 0.5/(n-1) where n is the number of states).

MDPs and Tabular methods
------------------------

Many problems can be effectively modeled as Markov Decision Processes (MDPs), and usually as `Partially Observable Markov Decision Processes (POMDPs) <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`. That is, we have 

  * a set of states :math:`S`
  * a set of actions :math:`A`
  * a set of conditional state transition probabilities :math:`T`
  * a reward function :math:`R: S \times A \rightarrow \mathbb{R}`
  * a set of observations :math:`\Omega`
  * a set of condition observation probabilities :math:`O`
  * a discount factor :math:`\gamma \in [0]`

Given these things, the goal is to choose the action at each time step which will maximize :math:`E \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]`, the expected discounted reward.
   
Monte Carlo methods
-------------------

One possible approach is to run a large number of simulations to learn :math:`p^*`. This is good for cases where we know the environment and can run many simulations reasonably quickly. For example, it is fairly trivial to compute an optimal policy for the card game `21 (blackjack) <https://en.wikipedia.org/wiki/Twenty-One_(card_game)>` by running many simulations, and the same is true for most simple games.

Temporal-Difference Learning
----------------------------

TODO

Planning
--------

TODO

On-Policy vs. Off-Policy Learning
---------------------------------

TODO

Model-Free vs. Model-Based Approaches
-------------------------------------

TODO

Imitation Learning
------------------

TODO

Q-Learning
----------

TODO

Deep Q-Learning
---------------

Deep Q-learning pursues the same general methods as Q-learning. Its innovation is to add a neural network, which makes it possible to learn a very complex Q-function. This makes it very powerful, especially because it makes a large body of well-developed theory and tools for deep learning useful to reinforcement learning problems.

Examples of Applications
------------------------

TODO

Links
-----

  * `Practical Applications of Reinforcement Learning (tTowards Data Science) <https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12>`_

  * `Reinforcement learning (GeeksforGeeks) <https://www.geeksforgeeks.org/what-is-reinforcement-learning/>`_ 

  * `Reinforcement Learning Algorithms: An Intuitive Overview (SmartLabAI) <https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc>`_ 



.. rubric:: References

.. [1] https://en.wikipedia.org/wiki/Reinforcement_learning#Introduction
.. [2] Reinforcement Learning: An Introduction (Sutton and Barto, 2018)




