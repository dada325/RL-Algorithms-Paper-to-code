# -RL-Algorithms-JAX

# About

Rlax Implementation of state-of-the-art model-free reinforcement learning algorithms on Openai gym environments.

# Some classic Papers and my reviews

| Algorithm                                       | Authors           | Year |
|-------------------------------------------------|-------------------|------|
| [A2C / A3C (Asynchronous Advantage Actor-Critic)](https://arxiv.org/abs/1602.01783) | Mnih et al        | 2016 |
| [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347)              | Schulman et al    | 2017 |
| [TRPO (Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477)         | Schulman et al    | 2015 |
| [DDPG (Deep Deterministic Policy Gradient)](https://arxiv.org/abs/1509.02971)       | Lillicrap et al   | 2015 |
| [TD3 (Twin Delayed DDPG)](https://arxiv.org/abs/1802.09477)                         | Fujimoto et al    | 2018 |
| [SAC (Soft Actor-Critic)](https://arxiv.org/abs/1801.01290)                         | Haarnoja et al    | 2018 |
| [DQN (Deep Q-Networks)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)                           | Mnih et al        | 2013 |
| [C51 (Categorical 51-Atom DQN)](https://arxiv.org/abs/1707.06887)                   | Bellemare et al   | 2017 |
| [QR-DQN (Quantile Regression DQN)](https://arxiv.org/abs/1710.10044)               | Dabney et al      | 2017 |
| [HER (Hindsight Experience Replay)](https://arxiv.org/abs/1707.01495)               | Andrychowicz et al| 2017 |
| [World Models](https://worldmodels.github.io/)                                  | Ha and Schmidhuber| 2018 |
| [I2A (Imagination-Augmented Agents)](https://arxiv.org/abs/1707.06203)              | Weber et al       | 2017 |
| [MBMF (Model-Based RL with Model-Free Fine-Tuning)](https://sites.google.com/view/mbmf)| Nagabandi et al  | 2017 |
| [MBVE (Model-Based Value Expansion)](https://arxiv.org/abs/1803.00101)              | Feinberg et al    | 2018 |
| [AlphaZero](https://arxiv.org/abs/1712.01815)                                      | Silver et al      | 2017 |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)|LeCun et al| 2022 | 
| [DreamerV3](https://arxiv.org/pdf/2301.04104.pdf)                                  | Hafner et al      | 2023|	

Please note that this repo is a collection of algorithms I implemented and tested out of my own interest. But I think it could be helpful to share it with others and I'm expecting useful discussions on my implementations. I found that there are no Implementation for JAX, so I think maybe it is challging to make my own one. The repo is presented in a way more like a tutorial and maybe the code is not that clean as I might refactor the code a couple of times. 

# Content

# Table of Contents

1. [Introduction to Rlax and its integration with JAX](#intro-rlax-jax) 🤝
2. [Basic building blocks of an RL algorithm](#basic-rl) 🚀
3. [Implementing Q-learning using Rlax and JAX](#q-learning) 📝
    - [Deep Q Networks (DQN)](#dqn)
    - [Experience Replay](#experience-replay)
    - [Target Networks](#target-networks)
4. [Experimenting with different environments using OpenAI Gym](#openai-gym) 🧪
5. [Implementing policy gradients (REINFORCE)](#reinforce) 🤖
6. [Exploration strategies in RL](#exploration-strategies) 🎲
7. [Fine-tuning and optimizing RL algorithms](#fine-tuning) 📈
8. [Implementing advanced RL algorithms](#advanced-rl) 🚀
    - [Actor-Critic (AC/A2C)](#ac-a2c) 🧮
    - [Soft Actor-Critic (SAC)](#sac) 📈
    - [Deep Deterministic Policy Gradient (DDPG)](#ddpg) 🤖
    - [Twin Delayed DDPG (TD3)](#td3) 🧮
    - [Proximal Policy Optimization (PPO)](#ppo) 📈
    - [QT-Opt (including Cross-entropy (CE) Method)](#qt-opt) 🎲
    - [PointNet](#pointnet) 🧮
    - [Transporter](#transporter) 🤖
    - [Recurrent Policy Gradient](#recurrent-policy-gradient) 📈
    - [Soft Decision Tree](#soft-decision-tree) 🧮
    - [Probabilistic Mixture-of-Experts](#mixture-of-experts) 🤖
    - [QMIX](#qmix) 🧮
9. [Best practices for training stable and robust agents](#best-practices) 🤖
10. [Troubleshooting and debugging RL algorithms](#troubleshooting) 🧪


# Intro-rlax-jax
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

Now we can do the same with the following:

```python
import jax
import rlax

def epsilon_greedy(q_values, epsilon):
    return rlax.epsilon_greedy(epsilon).sample(seed, q_values)
```

# Basic-RL

# Q-learning

# advanced-RL

## Actor-Critic (AC) / A2C

**File:** `ac.py` - Extensible AC/A2C, easy to change to be DDPG, etc. A very extensible version of vanilla AC/A2C, supporting for all continuous/discrete deterministic/non-deterministic cases.


## Soft Actor-Critic (SAC)

Multiple versions of SAC are implemented.

### SAC Version 1

**File:** `sac.py` - Using state-value function.

**Reference Paper:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)

### SAC Version 2

**File:** `sac_v2.py` - Using target Q-value function instead of state-value function.

**Reference Paper:** [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

### SAC Discrete

**File:** `sac_discrete.py` - For discrete action space.

**Reference Paper:** [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)

### SAC Discrete PER

**File:** `sac_discrete_per.py` - For discrete action space, and with prioritized experience replay (PER).

## Deep Deterministic Policy Gradient (DDPG)

**File:** `ddpg.py` - Implementation of DDPG.

## Twin Delayed DDPG (TD3)

**File:** `td3.py` - Implementation of TD3.

**Reference Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

## Proximal Policy Optimization (PPO)

For continuous environments, two versions are implemented:

**Version 1:** `ppo_continuous.py` and `ppo_continuous_multiprocess.py`

**Version 2:** `ppo_continuous2.py` and `ppo_continuous_multiprocess2.py`

For discrete environment:

**File:** `ppo_gae_discrete.py` - With Generalized Advantage Estimation (GAE)

## DQN

**File:** `dqn.py` - A simple DQN.

## QT-Opt

Two versions are implemented here. PointNet for landmarks generation from images with unsupervised learning is implemented here. This method is also used for image-based reinforcement learning as a SOTA algorithm, called Transporter.

**Original Paper:** Unsupervised Learning of Object Landmarksthrough Conditional Image Generation

**Paper for RL:** Unsupervised Learning of Object Keypointsfor Perception and Control

## Recurrent Policy Gradient

**Files:**

- `rdpg.py`: DDPG with LSTM policy.
- `td3_lstm.py`: TD3 with LSTM policy.
- `sac_v2_lstm.py`: SAC with LSTM policy.
- `sac_v2_gru.py`: SAC with GRU policy.

**References:**

- Memory-based control with recurrent neural networks
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization

## Soft Decision Tree as function approximator for PPO

**File:** `sdt_ppo_gae_discrete.py` - Replace the network layers of policy in PPO to be a Soft Decision Tree, for achieving explainable RL.

**Reference Paper:** [CDT: Cascading Decision Trees for Explainable Reinforcement Learning](https://arxiv.org/pdf/1910.07207)

## Probabilistic Mixture-of-Experts (PMOE)

PMOE uses a differentiable multi-modal Gaussian distribution to replace the standard unimodal Gaussian```markdown
distribution for policy representation.

**Files:**
- `pmoe_sac.py`: Based on off-policy SAC.
- `pmoe_ppo.py`: Based on on-policy PPO.

**Reference Paper:** [Probabilistic Mixture-of-Experts for Efficient Deep Reinforcement Learning](https://arxiv.org/abs/1910.07207)

## QMIX

**File:** `qmix.py` - A fully cooperative multi-agent RL algorithm, demo environment using pettingzoo.

**Reference Paper:** [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a.html)

## Phasic Policy Gradient (PPG)

**Status:** To Do

**Reference Paper:** [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)

## Maximum a Posteriori Policy Optimisation (MPO)

**Status:** To Do

**Reference Paper:** [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920)

## Advantage-Weighted Regression (AWR)

**Status:** To Do

**Reference Paper:** [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177)


