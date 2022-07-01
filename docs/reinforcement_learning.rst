.. _reinforcement_learning:

======================
Reinforcement Learning
======================

In machine learning, supervised is sometimes contrasted with unsupervised learning. This is a useful distinction, but there are some problem domains that have share characteristics with each without fitting exactly in either category. In cases where the algorithm does not have explicit labels but does receive a form of feedback, we are dealing with a third and distinct paradigm of machine learning - reinforcement learning.

Programmatic and a theoretical introduction to reinforcement learning:https://spinningup.openai.com/

There are different problem types and algorithms, but all reinforcement learning problems have the following aspects in common:

  * an **agent** - the algorithm or "AI" responsible for making decisions

  * an **environment**, consisting of different **states** in which the agent may find itself

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

Q Learning, a model-free RL algorithm, is to update Q values to the optimal by iteration. It is an off-policy method that select the optimal action based on the current estimated Q\* and does not follow the current policy.

The algorithm of Q Learning is:

	#. Initialize t = 0.
	#. Start at initial state s\ :sub:`t` = 0.
	#. The agent chooses a\ :sub:`t` = ɛ-greedy
	   action.
	#. For given a\ :sub:`t`, the agent retrieves
	   the reward r\ :sub:`t+1` as well as the next
	   state s\ :sub:`t+1`.
	#. Get (but do not perform) the next action
	   a\ :sub:`t+1` =
	   argmax\ :sub:`a∈A`\ Q(s\ :sub:`t+1`, a).
	#. Compute the TD target y\ :sub:`t` =
	   r\ :sub:`t+1` + γ · Q(s\ :sub:`t+1`,
	   a\ :sub:`t+1`), where γ is the discounted
	   factor.
	#. Calculate the TD error δ = y\ :sub:`t` −
	   Q(s\ :sub:`t`, a\ :sub:`t`).
	#. Update Q(s\ :sub:`t`, a\ :sub:`t`) ←
	   Q(s\ :sub:`t`, a\ :sub:`t`) + α\ :sub:`t` ·
	   δ, where α\ :sub:`t` is the step size
	   (learning rate) at t.
	#. Update t ← t + 1 and repeat step 3-9 until
	   Q(s, a) converge.
	   
Epsilon-Greedy Algorithm

.. math::

	\begin{equation}
	a_{t} = \begin{cases}
	argmax_{a∈A} & \text{if } p = 1 - e \\
	random\, action\ &\text{otherwise}
	\end{cases}
	\end{equation}

The agent performs optimal action for exploitation or random action for exploration during training. It acts randomly in the beginning with the ɛ = 1 and chooses the best action based on the Q function with a decreasing ɛ capped at some small constant not equal to zero.

Q-Table / Q-Matrix

	+-------------+---------------+---------------+-----+---------------+
	|             | a\ :sub:`1`   | a\ :sub:`2`   | ... | a\ :sub:`n`   |
	+-------------+---------------+---------------+-----+---------------+
	| s\ :sub:`1` | Q             | Q             | ... | Q             |
	|             | (s\ :sub:`1`, | (s\ :sub:`1`, |     | (s\ :sub:`1`, |
	|             | a\ :sub:`1`)  | a\ :sub:`2`)  |     | a\ :sub:`3`)  |
	+-------------+---------------+---------------+-----+---------------+
	| s\ :sub:`2` | Q             | Q             | ... | Q             |
	|             | (s\ :sub:`2`, | (s\ :sub:`2`, |     | (s\ :sub:`2`, |
	|             | a\ :sub:`1`)  | a\ :sub:`2`)  |     | a\ :sub:`3`)  |
	+-------------+---------------+---------------+-----+---------------+
	| ...         | ...           | ...           | ... | ...           |
	+-------------+---------------+---------------+-----+---------------+
	| s\ :sub:`m` | Q             | Q             | ... | Q             |
	|             | (s\ :sub:`m`, | (s\ :sub:`m`, |     | (s\ :sub:`m`, |
	|             | a\ :sub:`1`)  | a\ :sub:`2`)  |     | a\ :sub:`3`)  |
	+-------------+---------------+---------------+-----+---------------+
	
It's a lookup table storing the action-value function Q(s, a) for state-action pairs where there are M states and n actions. We can initialize the Q(s, a) arbitrarily except s = terminal state. For s = final state, we set it equal to the reward on that state.

Reasons of using Q Learning are:

	-  It’s applicable for the discrete action space of our environment.
	-  When we don’t have the true MDP model: transitional probability matrix and rewards (Model-Free Setting).
	-  It's able to learn from incomplete episodes because of TD learning.

Drawbacks of Q Learning are:

	-  When the state space and action space are continuous and extremely large, due to the curse of dimensionality, it’s nearly impossible to maintain a Q-matrix when the data is large.
	-  Using a Q-table is unable to infer optimal action for unseen states.
	   
Deep Q-Learning
---------------

Deep Q-learning pursues the same general methods as Q-learning. Its innovation is to add a neural network, which makes it possible to learn a very complex Q-function. This makes it very powerful, especially because it makes a large body of well-developed theory and tools for deep learning useful to reinforcement learning problems.

Examples of Applications
------------------------

  * `Getting Started With OpenAI Gym: Creating Custom Gym Environments <https://blog.paperspace.com/creating-custom-environments-openai-gym/>`_

  * `What Is Q-Learning: The Best Guide To Understand Q-Learning (Simplilearn) <https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning>`_

  * `REINFORCEMENT LEARNING (DQN) TUTORIAL (PyTorch) <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>`_

  * `QWOP Game AI (DQN/DDQN) <https://github.com/yatshunlee/qwop_RL>`_

Links
-----

  * `Practical Applications of Reinforcement Learning (tTowards Data Science) <https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12>`_

  * `Reinforcement learning (GeeksforGeeks) <https://www.geeksforgeeks.org/what-is-reinforcement-learning/>`_ 

  * `Reinforcement Learning Algorithms: An Intuitive Overview (SmartLabAI) <https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc>`_ 
  
  * `Q-learning(Wikipedia) <https://en.wikipedia.org/wiki/Q-learning>`_

  * `Epsilon-Greedy Algorithm in Reinforcement Learning (GeeksforGeeks) <https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/>`_

  * `OpenAI Gym Documentation <https://www.gymlibrary.ml/>`_

  * `Stable-Baselines3 Documentation <https://stable-baselines3.readthedocs.io/en/master/#>`_
  
  * `David Silver Teaching Material <https://www.davidsilver.uk/teaching/>`_



.. rubric:: References

.. [1] https://en.wikipedia.org/wiki/Reinforcement_learning#Introduction
.. [2] Reinforcement Learning: An Introduction (Sutton and Barto, 2018)
.. [3] Silver, David. "Lecture 5: Model-Free Control." UCL, Computer Sci. Dep. Reinf Learn. Lect. (2015): 101-140.
.. [4] En.wikipedia.org. 2022. Q-learning - Wikipedia. [online] Available at: <https://en.wikipedia.org/wiki/Q-learning> [Accessed 15 June 2022].




