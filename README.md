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

# Basic-RL
