# RL-Algorithms

# About

Rlax Implementation of state-of-the-art model-free reinforcement learning algorithms on Openai gym environments.


### High level Overview of the Algorithms

| Generation | Method | Description | Key Idea |
|------------|--------|-------------|----------|
| 1 | Tabular methods (Q-learning, SARSA) | Use a table to store the expected reward for each action in each state. | Simple and effective but do not scale well to large state spaces. |
| 2 | Function approximation | Represent the Q-function as a parameterized function. (such as a linear function or a neural network.) | Allows RL to scale to much larger state spaces. |
| 3 | Policy gradients (REINFORCE) | Directly learn a policy that specifies which action to take in each state. The REINFORCE algorithm is a foundational policy gradient method that uses the gradient of the expected reward to improve the policy. | Use the gradient of the expected reward to improve the policy. |
| 4 | Actor-Critic methods (A2C, A3C) | Combine value-based methods and policy-based methods. | Use a critic to estimate the value of actions, and an actor to improve the policy based on the critic's estimates. |
| 5 | Trust Region methods (TRPO, PPO) | Add an additional constraint to the learning process to ensure the policy doesn't change too much in one step. | Makes the learning process more stable. |
| 6 | Distributional RL (C51, QR-DQN) | Aim to learn the full distribution of possible rewards. | Provides more information and can lead to better policies. |
| 7 | Model-based RL | First learn a model of the environment, then use this model to plan and make decisions. | More sample-efficient than model-free methods, but learning an accurate model can be challenging. |


# Content

# Table of Contents

1. [Introduction](#intro) 
2. [Basic building blocks of an RL algorithm](#basic-rl) 
3. [Implementing Q-learning](#q-learning) 
    - [Deep Q Networks (DQN)](#dqn)
    - [Experience Replay](#experience-replay)
    - [Target Networks](#target-networks)
4. [Experimenting with different environments using OpenAI Gym](#openai-gym) 
5. [Implementing policy gradients (REINFORCE)](#reinforce) 
6. [Exploration strategies in RL](#exploration-strategies) 
7. [Fine-tuning and optimizing RL algorithms](#fine-tuning) 
8. [Implementing advanced RL algorithms](#advanced-rl) 
    - [Actor-Critic (AC/A2C)](#ac-a2c) 
    - [Deep Deterministic Policy Gradient (DDPG)](#ddpg) 
    - [Twin Delayed DDPG (TD3)](#td3)
    - [Soft Actor-Critic (SAC)](#sac) 
    - [Proximal Policy Optimization (PPO)](#ppo) 

9. [Best practices for training stable and robust agents](#best-practices) 
10. [Troubleshooting and debugging RL algorithms](#troubleshooting) 
11. [Key Paper in RL](#Key_Papers_in_Deep_RL)


# Intro
Let's start with an example: Suppose you want to implement the ϵ-greedy algorithm for a simple reinforcement learning problem. ϵ-greedy is a simple method to balance exploration and exploitation by choosing between a random action and the action currently believed to be the best with a certain probability.

In Python, you might write this using NumPy like so:

```python
def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(len(q_values))
    else:
        action = np.argmax(q_values)
    return action
```



# Basic-RL

# Q-learning

# Openai-gym

Gymnasium Envs (RL Algorithms for Different Environments)

## Classic Control
Classic Control is simple,  simpler RL methods such as Q-learning and Policy Gradient methods may works well, the state and action spaces are relatively small and the physics are deterministic.

- **Acrobot**: Q-learning or SARSA (small discrete action space)
- **Cart Pole**: DQN (small discrete action space and continuous state space)
- **Mountain Car Continuous**: DDPG (continuous action space)
- **Mountain Car**: DQN (discrete action space)
- **Pendulum**: DDPG (continuous action space)

## Box2D
These environments are more complex, requiring RL methods such as DDPG or PPO.

- **Bipedal Walker**: PPO or TRPO (complex task with continuous action space)
- **Car Racing**: PPO or A3C (complex task with high dimensional state space)
- **Lunar Lander**: DQN or DDPG (discrete and continuous versions available)

## Toy Text
These environments have simple, discrete state and action spaces, so tabular methods like Q-learning may be effective, but deep RL methods can also be applied here.

- **Blackjack**: Monte Carlo Control or Q-Learning (eposidic, perfect information game)
- **Taxi**: Q-learning or SARSA (small discrete state and action space)
- **Cliff Walking**: Q-learning or SARSA (gridworld environment with discrete states and actions)
- **Frozen Lake**: Q-learning or Value Iteration (gridworld environment with discrete states and actions)

## MuJoCo
These are complex, continuous control tasks that typically require advanced methods like DDPG, TRPO, or PPO.

- **Ant, Half Cheetah, Hopper, Humanoid Standup, Humanoid, Inverted Double Pendulum, Inverted Pendulum, Reacher, Swimmer, Pusher, Walker2D**: DDPG, TRPO or PPO (complex tasks with continuous action spaces)

## Atari
These are complex tasks with high-dimensional state spaces (if you use the raw pixels as input) that typically require deep RL methods. DQN was specifically developed for these kinds of tasks.

- **All Atari games**: DQN or its variations (Double DQN, Dueling DQN), or policy gradient methods like A3C or PPO (reason: high dimensional state space with discrete actions)



# reinforce

# Advanced-RL

## Actor-Critic (AC) / A2C

**File:** `sdt_ppo_gae_discrete.py` - Replace the network layers of policy in PPO to be a Soft Decision Tree, for achieving explainable RL.




# Key_Papers_in_Deep_RL

## 1. Model-Free RL
### Deep Q-Learning
- [1] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), Mnih et al, 2013. Algorithm: DQN.
- [2] [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527), Hausknecht and Stone, 2015. Algorithm: Deep Recurrent Q-Learning.
- [3] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang et al, 2015. Algorithm: Dueling DQN.
- [4] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), Hasselt et al 2015. Algorithm: Double DQN.
- [5] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul et al, 2015. Algorithm: Prioritized Experience Replay (PER).
- [6] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel et al, 2017. Algorithm: Rainbow DQN.

### Policy Gradients
- [7] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih et al, 2016. Algorithm: A3C.
- [8] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al, 2015. Algorithm: TRPO.
- [9] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al, 2015. Algorithm: GAE.
- [10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al, 2017. Algorithm: PPO-Clip, PPO-Penalty.
- [11] [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), Heess et al, 2017. Algorithm: PPO-Penalty.
- [12] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144), Wu et al, 2017. Algorithm: ACKTR.
- [13] [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224), Wang et al, 2016. Algorithm: ACER.
- [14] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018. Algorithm: SAC.

### Distributional RL
- [18] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare et al, 2017. Algorithm: C51.
- [19] [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044), Dabney et al, 2017. Algorithm: QR-DQN.
- [20] [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923), Dabney et al, 2018. Algorithm: IQN.
- [21] [Dopamine: A Research Framework for Deep Reinforcement Learning](https://arxiv.org/abs/1812.06110), Anonymous, 2018. Contribution: Introduces Dopamine, a code repository containing implementations of DQN, C51, IQN, and Rainbow.

## 2. Exploration
### Intrinsic Motivation
- [32] [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674), Houthooft et al, 2016. Algorithm: VIME.
- [33] [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868), Bellemare et al, 2016. Algorithm: CTS-based Pseudocounts.
- [34] [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310), Ostrovski et al, 2017. Algorithm: PixelCNN-based Pseudocounts.
- [35] [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717), Tang et al, 2016. Algorithm: Hash-based Counts.
- [36] [EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01250), Fu et al, 2017. Algorithm: EX2.
- [37] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363), Pathak et al, 2017. Algorithm: Intrinsic Curiosity Module (ICM).
- [38] [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355), Burda et al, 2018. Contribution: Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments.
- [39] [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), Burda et al, 2018. Algorithm: RND.

## 3. Transfer and Multitask RL
- [43] [Progressive Neural Networks](https://arxiv.org/abs/1606.04671), Rusu et al, 2016. Algorithm: Progressive Networks.
- [44] [Universal Value Function Approximators](https://arxiv.org/abs/1502.02167), Schaul et al, 2015. Algorithm: UVFA.
- [45] [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397), Jaderberg et al, 2016. Algorithm: UNREAL.
- [46] [The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300), Cabi et al, 2017. Algorithm: IU Agent.
- [47] [PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/abs/1701.08734), Fernando et al, 2017. Algorithm: PathNet.
- [48][Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907), Wulfmeier et al, 2017. Algorithm: MATL.
- [49] [Learning an Embedding Space for Transferable Robot Skills](https://arxiv.org/abs/1802.03776), Hausman et al, 2018.
- [50] [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), Andrychowicz et al, 2017. Algorithm: Hindsight Experience Replay (HER).

## 4. Memory
- [54] [Model-Free Episodic Control](https://arxiv.org/abs/1606.04460), Blundell et al, 2016. Algorithm: MFEC.
- [55] [Neural Episodic Control](https://arxiv.org/abs/1703.01988), Pritzel et al, 2017. Algorithm: NEC.
- [56] [Neural Map: Structured Memory for Deep Reinforcement Learning](https://arxiv.org/abs/1702.08360), Parisotto and Salakhutdinov, 2017. Algorithm: Neural Map.
- [57] [Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/abs/1803.10760), Wayne et al, 2018. Algorithm: MERLIN.
- [58] [Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822), Santoro et al, 2018. Algorithm: RMC.

## 5. Model-Based RL
### Model is Learned
- [59] [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203), Weber et al, 2017. Algorithm: I2A.
- [60] [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596), Nagabandi et al, 2017. Algorithm: MBMF.
- [61] [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/abs/1803.00101), Feinberg et al, 2018. Algorithm: MVE.
- [62] [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675), Buckman et al, 2018. Algorithm: STEVE.
- [63] [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/abs/1802.10592), Kurutach et al, 2018. Algorithm: ME-TRPO.
- [64] [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214), Clavera et al, 2018. Algorithm: MB-MPO.
- [65] [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.11096), Ha and Schmidhuber, 2018.

## 6. Scaling RL
- [72] [Accelerated Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1803.02811), Stooke and Abbeel, 2018. Contribution: Systematic analysis of parallelization in deep RL across algorithms.
- [73] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561), Espeholt et al, 2018. Algorithm: IMPALA.
- [74] [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933), Horgan et al, 2018. Algorithm: Ape-X.
- [75] [Recurrent Experience Replay in Distributed Reinforcement Learning](https://arxiv.org/abs/1805.11593), Anonymous, 2018. Algorithm: R2D2.
- [76] [RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381), Liang et al, 2017. Contribution: A scalable library of RL algorithm implementations.

## 7. Safety
- [81] [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), Amodei et al, 2016. Contribution: establishes a taxonomy of safety problems, serving as an important jumping-off point for future research.
- [82] [Deep Reinforcement Learning From Human Preferences](https://arxiv.org/abs/1706.03741), Christiano et al, 2017. Algorithm: LFP.
- [83] [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), Achiam et al, 2017. Algorithm: CPO.
- [84] [Safe Exploration in Continuous Action Spaces](https://arxiv.org/abs/1801.08757), Dalal et al, 2018. Algorithm: DDPG+Safety Layer.
- [85] [Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173), Saunders et al, 2017. Algorithm: HIRL.
- [86] [Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning](https://arxiv.org/abs/1711.06782), Eysenbach et al, 2017. Algorithm: Leave No Trace.

## 8. Imitation Learning and Inverse Reinforcement Learning
- [87] [Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](https://www.ri.cmu.edu/pub_files/2010/6/Ziebart_PhD_Thesis_2010.pdf), Ziebart 2010. Contributions: Crisp formulation of maximum entropy IRL.
- [88] [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn et al, 2016. Algorithm: GCL.
- [89] [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho and Ermon, 2016. Algorithm: GAIL.
- [90] [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717), Peng et al, 2018. Algorithm: DeepMimic.
- [91] [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821), Peng et al, 2018. Algorithm: VAIL.
- [92] [One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/abs/1810.05017), Le Paine et al, 2018. Algorithm: MetaMimic.

## 9. Reproducibility, Analysis, and Critique
- [93] [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778), Duan et al, 2016. Contribution: rllab.
- [94] [Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133), Islam et al, 2017.
- [95] [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), Henderson et al, 2017.
- [96] [Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods](https://arxiv.org/abs/1810.02525), Henderson et al, 2018.
- [97] [Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?](https://arxiv.org/abs/1811.02553), Ilyas et al, 2018.
- [98] [Simple Random Search Provides a Competitive Approach to Reinforcement Learning](https://arxiv.org/abs/1803.07055), Mania et al, 2018.
- [99] [Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/abs/1907.02057), Wang et al, 2019.

## 10. Bonus: Classic Papers in RL Theory or Review
- [100] [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton et al, 2000. Contributions: Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes.
- [101] [An Analysis of Temporal-Difference Learning with Function Approximation](https://ieeexplore.ieee.org/document/639377), Tsitsiklis and Van Roy, 1997. Contributions: Variety of convergence results and counter-examples for value-learning methods in RL.
- [102] [Reinforcement Learning of Motor Skills with Policy Gradients](https://www.researchgate.net/publication/220632147_Reinforcement_Learning_of_Motor_Skills_with_Policy_Gradients), Peters and Schaal, 2008. Contributions: Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods.
- [103] [Approximately Optimal Approximate Reinforcement Learning](https://www.jmlr.org/papers/volume13/jiang12a/jiang12a.pdf), Kakade and Langford, 2002. Contributions: Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms.
- [104] [A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf), Kakade, 2002. Contributions: Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL.
- [105] [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf), Szepesvari, 2009. Contributions: Unbeatable reference on RL before deep RL, containing foundations and theoretical background.


