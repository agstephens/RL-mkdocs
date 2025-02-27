
By: Abdulrahman Altahhan, Feb 2025

#  Welcome to Reinforcement Learning and Robotics!

Welcome to the Reinforcement Learning and Robotics module, where you learn the fundamentals of reinforcement learning(RL) with various simulaitons including robotics. The module focuses on RL as a general robust framework that allows us to deal with autonomous agent learning. 

In each unit, you will learn essential RL ideas and algorithms and see how these can be implemented in simplified environments. Each lesson involves some reading material, along with some videos explaining these lessons. This will then be followed by running and experimenting with the Jupyter notebooks we provide, which implement the main ideas you have read and show you the RL algorithms you studied in action. It is essential that you read the material and engage with these notebooks in the way you see fit. This includes studying and running the provided code to gain insight into the covered algorithms, observing the algorithm convergence behaviour, experimenting with the hyperparameter and their effect on the agent's behaviour, and taking a turn to implement some concepts that were left out for you. 

There are also some lessons that familiarise you with robotics as an application domain, which covers some simple concepts in robotics without delving deep into classical robotics, which is outside the scope of this module. Our units conclude with a simple, practical tutorial on utilising a simulated robot. You will use the provided code to control a simulated mobile robot called TurtleBot. The lessons are meant to gradually build your ability to deal with atonomous agents in a simplified environment. The final project will focus on comparing different RL solutions to solve a robot navigation problem. We will provide you access to an Azure Ubuntu virtual machine already set up with ROS2 to run the sheets. You do not need to set up your own VM.

Reinforcement Learning (RL) is dedicated to acquiring an optimal policy to enhance agent performance. Traditionally, RL involves the agent seeking an optimal policy to maximise cumulative rewards garnered by navigating various environmental states. While the agent is commonly perceived as a physical entity interacting with its surroundings, it can also encompass abstract systems, with states representing system configurations or settings. Notably, recent RL advancements have ventured into novel territories, such as optimising language models like LLMs for perplexity or other metrics.

In this module, our focus primarily revolves around simulated agents, aligning with the nature of our programme. However, the underlying principles remain universal and applicable across diverse scenarios. The concept of guiding learning through rewards is deeply ingrained in biological organisms, from complex human brains to single-cell organisms like amoebas, all driven by the innate urge to maximise survival and proliferation.

While RL offers tremendous efficacy when configured appropriately, but it is susceptible to spectacular failure when its conditions are unmet. Its inherent stochasticity adds layers of complexity, contributing to its volatile nature. Nonetheless, this volatility serves as a catalyst for researchers to delve deeper into understanding the governing rules of RL processes.

Our module adopts a pragmatic approach, aiming to provide participants with a solid theoretical grounding in RL without overly delving into intricate mathematical details. Simultaneously, we equip learners with practical skills and techniques to harness RL's benefits effectively. This balanced approach ensures that you grasp RL's essence while gaining valuable real-world application tools.


**Unit 1 Learning outcomes**

1. understand the armed bandit problem and its simplification of isolating the action space
1. understand the value function and the action-value function and their essential roles in RL
1. understand the difference between associative and non-associative problems
1. understand the underlying theory of RL including MDP and Bellman equation
1. understand the difference between prediction and control in RL settings

**Unit 2 Learning outcomes**

1. understand how to predict the value function for a policy in tabular settings
1. understand how to control an agent by inferring its policy from an action-value function
1. understand the idea of generalised policy iteration (GPI) that is utilised by many RL methods
1. understand the difference between full-backup action-value based control methods and direct policy estimation control methods
1. understand how MC        obtains an unbiased but high-variance estimation via interaction with the environment
1. understand how REINFORCE obtains an unbiased but high variance estimation via interaction with using the environment

**Unit 3 Learning outcomes**

1. appreciate the importance of bootstrapping and its essential role in RL
1. understand n-steps methods and the trade-offs that n represents
1. understand the difference between n-steps-backup action-value based control methods and direct policy estimation control methods
1. understand how TD obtains a biased but low variance estimation via interaction with the environment
1. understand how actor-critic obtains a biased but low variance estimation via interaction with using the environment
1. understand online and offline algorithms tradeoffs
1. understand planning methods and how to build a model along the way by learning

**Unit 4 Learning outcomes**

1. apply RL to control an agent in a more complex environment representation
1. understand on-policy and off-policy algorithms trade-offs 
1. be familiar with the convergence of RL algorithms in tabular and approximation settings and their practical limitations

**Unit 5 Learning outcomes**

1. understand how to predict the value function for a policy using function approximation
1. understand eligibility traces methods and the trade-offs that their depth represents
1. understand how to control an agent by inferring its policy from an action-value function with function approximation
1. apply RL to control a robot

**Unit 6 Learning outcomes**

1. apply RL to control an agent in Games


## RL in context: advantages and challenges
In our RL coverage, you will notice that we do not do classical robotics and we believe it is not the way to go except possibly for industrial robots. So, we do not need motion planning or accurate trajectory calculations/kinematics to allow the robot to execute a task, we simply let it learn by itself how to solve the task and this is what reinforcement learning is about. In our coverage we also do not supervise or directly teach the robot, this type of interference is called imitation. 

This is all great but what about challenges? Obviously, we still have several challenges to this approach. One is the number of experiments required to learn the task which can be numerous, which exposes the physical agents (real robots) to the risk of wear and tear due to repetition and can be very lengthy and tedious for the human that is supervising the task. This can be partially overcome by starting in simulation and then moving to a real robot with a technique called [sim-to-real](https://ai.googleblog.com/2021/06/toward-generalized-sim-to-real-transfer.html) where we can employ GANs(generative adversarial neural networks).  The other challenge is the time required for training regardless whether it is in simulation or in real scenarios. 


The origin of these problems is actually the exploitation/exploration needed by RL algorithms which is in the heart of what we will be doing in all of our RL coverage. Reducing the amount of training required is important and remains an active area for research. One approach is via experience replay and the other is via eligibility traces as we shall see later.

## Textbook
The accompanying textbook of this and consecutive units is the Sutton Barto book Introduction to RL available online [here](http://incompleteideas.net/book/RLbook2020.pdf). Please note that we explain the ideas of RL from a practical perspective and not from a theoretical perspective which is already covered in the textbook.

**list of symbols**
- $v0$: denotes an initial value
- $θ$: denotes a threshold
- $nS$: denotes state space dimension 
- $nA$: denotes actions space dimension 
- $nR$: denotes rewards space dimension
- $nU$: denotes the number of updates
- goal: a terminal state

- $r$: current step reward
- $s$: current step state
- $a$: current step action
- $rn$: next step reward
- $sn$: next step state
- $an$: next step action

- $α$: learning rate
- $ε$: exploration rate
- $dα$: decay factor for α
- $dε$: decay factor for ε
- $G(t+1,t+n)$: the return between time step t+1 and t+n

- Rs: is the sum of rewards of an episode
- Ts: is the steps of a set of an episode
- Es: is the errors (RMSE) of an episode


## Module Plan
We cover the tabular solution methods in the first three units, while approximate solution methods will be covered in subsequent units.

Tabular and approximate solution methods fall under two types of RL methods that we will attempt to deal with 

1. Prediction methods: AKA Policy Evaluation Methods that attempt to find the best estimate for the value-function or action-value function given a policy.
2. Control methods: AKA Policy Improvement Methods that attempt to find the best policy, often by starting from an initial policy and then moving into a better and improved policy.

Policy improvement methods:
1. Methods that improve the policy via improving the action-value function
2. Methods that improve the policy directly

We start by assuming that the policy is fixed. This will help us develop algorithms predicting the state‘s value function (expected return). Then, we will move to the policy improvement methods, i.e., methods that help us compare our policy with other policies and move to a better policy when necessary. Then, we move to the control case (policy and value iteration methods).

Note that the guarantee from the policy-improvement theorem no longer applies when we move from the table representation of the value function for small state space to the parametric function approximation representation for large state space. This will encourage us to move to direct policy-improvement methods instead of improving the policy via improving the value function.

## VM and Prerequisite Packages 

Video Introduction to [Azure VM](https://leeds365-my.sharepoint.com/:v:/g/personal/scsaalt_leeds_ac_uk/EWEwLfWr5w9Mnl0E29LjvJUBlsG-nWbee6RzKHKYSp_D-Q?e=V79IXI)

We provide you with Azure VM which has [ROS 2 Foxy Fitzroy](https://docs.ros.org/en/foxy/Installation.html) already installed. ROS2 commands need to be run from the terminal not from a conda-activated terminal (due to compatibility), and they use the default system Python 3.8. The VM has the libraries required for ROS2 along with TurtleBot3 installed with the worlds required for assessment.


We have tested the notebooks on Python 3.8, so they should work smoothly for higher versions.
Note that ROS2 code must be run with the default VM Python3.8 kernel. For the best experience, use VScode

The machine has decent cores and memory (according to Azure 4 cores | 8GB RAM | 128GB Standard SSD). The VM has Ubuntu 20 and Xfce (Xubuntu) interface due to its lightweight (to give you the best experience remotely- to come as close as a local machine feeling) and it is tailored to give the same feeling as the usual Ubuntu Genome. You can run hardinfo in the terminal to check the VM specs. I hope you will enjoy it. 

To access the VM, please use the usual remote desktop app available on your system. You will receive an email with access to your VM. The username is rl, and the password is rl@ros2. 


You will have sudo access. Please apply caution when dealing with the system and avoid installing packages so as not to break it, which can be time-consuming. You will have around a 40-hour time limit, so please be mindful not to leave the system running unless necessary so as not to run out of time. Usually, you would want time for running the exercises and save plenty of time (1/2) for your project training (this is where the VM will be most useful).

If the VM becomes corrupted for some reason, then you can reimage it by going to Azure Lab page and selecting the three dots, then reimage. That *will cause all the data you have on the machine to be lost*. You are advised to back up your data, you may want to use OneDrive or other backup methods.


## Code Structure and Notebooks Cascade Dependency

**Important note: Please place the downloaded libraries within the same folder of the all notebooks to allow the imports to work appropriately**

We have provided you with two [RL libraries](/RLlibrary.zip){:download="RLlibrary.zip"} designed for this module. One has bespoke environments and one that has base RL classes that makes working with algorithms very easy and as close as it can be to just provide an update rule.

Worksheets can be cloned and launched in GitHub Codespaces (repo is: AltahhanAi/RL-worksheets)

[![Open in GitHub Codespaces-](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repository=AltahhanAi/RL-worksheets)




## Installing other libraries that will be needed later
<!-- general -->
```
!pip install --upgrade pip
!pip3 install opencv-python
!pip3 install scikit-learn
!pip install matplotlib
!pip3 install tqdm
```

<!--  neural networks and deep learning -->
```
!sudo apt install python3-testresources
!pip3 install -U setuptools
!pip3 install -U tensorflow[and-cuda]
!pip3 install keras
```

<!-- Better Readability -->
```
!pip install jupyterlab
!pip install jupyterthemes

!jt -t solarizedl -T -N  # -T, -N keeps the toolbar and header
```

### available themes:
- oceans16 
- grade3 
- chesterish 
- solarizedl 
- solarizedd 
- gruvboxl
- !jt -r # resets back to the default theme

## Better Readability
For better readability and experience, please use Jupyter Lab or Vcode(if you are using Azure VM) to navigate between the different notebooks easily. If you want to use Jupyter Notebooks and not Jupyter Lab, we recommend that you also increase the cells' width for a better reading experience. We provided a function that increase your notebook width which is envoked automatically when you import an environment (grid particularly) as you shall see later. You may want to utilise also the table of contents button in Jupetr Lab.

 - Unit 1: 
   1. [Tabular Methods](unit1/lesson1/lesson1.md)
   2. [K-Arm Bandit](unit1/lesson2/lesson2.md)
   3. [MDP](unit1/lesson3/lesson3.md)
   4. [ROS](unit1/lesson4/lesson4.md)

  - Unit 2: 
    5. [Dynamic Programming](unit2/lesson5/lesson5.md)
    6. [Monte Carlo](unit2/lesson6/lesson6.md)
    7. [Mobile Robots](unit2/lesson7/lesson7.md)

  - Unit 3:
    8. [Temporal Difference](unit3/lesson8/lesson8.md)
    9. [n-Step Methods](unit3/lesson9/lesson9.md)
    10. [Planning in RL(optional)](unit3/lesson10/lesson10.md)
    11. [Localisation and SLAM](unit3/lesson11/lesson11.md)

  - Unit 4: 
    12. [Function Approximation Methods](unit4/lesson12/lesson12.md)
    13. [Linear Approximation for Prediction](unit4/lesson13/lesson13.md)
    14. [Linear Approximation for Control](unit4/lesson14/lesson14.md)
    

  - Unit 5: 
    15. [Linear Approximation with Eligibility Traces(prediction and control)](unit5/lesson15/lesson15.md)
    16. [Nonlinear Approximation for Control](unit5/lesson16/lesson16.md)
    17. [Application on Robot Navigation](unit5/lesson17/lesson17.md)
    
  - Unit 6: 
    18. [Application on Games(optional)](unit6/lesson18/lesson18.md)
