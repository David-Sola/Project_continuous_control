[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Project 2: Continuous Control

### Introduction

In this project the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment shall be solved
as shown in the gif below with a trained agent.
This repository contains the solution for a single agent and not for moltuiple agents.
Nevertheless the pre-trained agent can also be taken for the environment with 20 reachers

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Getting Started

1. Follow the instructions in the [Udacity DRLND repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
2. Clone this repository into the p2_continuous-control folder of the udacity repository
3. Open the continuous control iPythonNotebook to either train a new agent or use the pre-trained agent which is already provided

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to  either:

- Train a new agent or
- See how the pretrained agent behaves in the environment

The code cells which needs to be executed are documented in the Navigation.ipynb