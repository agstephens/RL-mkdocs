







To help you, Abdulrahman recorded a set of videos that covers important concepts of RL.

## Markov Decision Processes (MDP)






We provided you with a simple code to in the worksheet below to demosntrate how the G changes with time steps.

```python
G = 0
R = 1
γ = 0.9
T = 100
for t in range(T,0,-1):
    if t> 70: print('G_',t,'=',round(G,3))
    G = R + γ*G 
```
    G_ 100 = 0
    G_ 99 = 1.0
    G_ 98 = 1.9
    G_ 97 = 2.71
    G_ 96 = 3.439
    ...




```python
G = 0
γ = 0.9 # change to 1 to see the effect
T = 100
for t in range(T,1,-1): # note that if we use a forward loop, our calculations will be all wrong although the code runs
    if t>70: print('G_',t,'=',round(G,2), round(γ**(T-t-1),2) if t<T else 0)
    R=1 if t==T else 0
    G = R + γ*G
```
    G_ 100 = 0 0
    G_ 99 = 1.0 1.0
    G_ 98 = 0.9 0.9
    G_ 97 = 0.81 0.81
    ...


## Bellman Equation for V and Q


## Bellman Optimality Equations for V and Q




## Grid World Environments

Ok, so now we are ready to tackle the practicals, please go ahead and download the worksheet and run and experiement with the provided code to build some grid world environments and visualise them and make a simple robot agent takes some steps/actions within these environments!.

You will need to download a python library (Grid.py) that we bespokley developed to help you run RL algorithms on toy problems and be abel to easily visualise them as needed, the code is optimised to run efficiently and you will be able to use these environmnets to test different RL algorithms extensively. Please place the library in the same directory of the worksheet. In general it would be a good idea to place all worksheets and libraries provided in one directory. This will make importing and runing code easier and more streamlined.


In a grid world, we have a set of cells that the agent can move between them inside a box. The agent can move left, right, up and down. We can also allow the agent to move diagonally, but this is uncommon. 

 We needed to be as efficient as possible, and hence we have chosen to represent each state by its count, where we count from the bottom left corner up to the right top corner, and we start with 0 up to nS-1 where nS is the number of states. This will allow us to streamline the process of accessing and storing a state and will be of at most efficiency. We also deal with actions similarly, i.e. each action of the nA actions is given an index 0..nA-1. For the usual grid, this means 0,1,2,3 for actions left, right, up and down. We represent a 2-d grid by a 1-d array, and so care must be taken on how the agent moves between cells. 

Moving left or right seems easy because we can add or subtract from the *current* state. But when the agent is on the edge of the box, we cannot allow for an action that takes it out of the box. So if the agent is on the far right, we cannot allow it to go further to the right. To account for this issue, we have written a valid() function to validate an action. Moving up and down is similar, but we need to add and subtract a full row, which is how many columns we have in our grid. the valid() function checks for the current state and what would be the next state, and it knows that an agent will overstep the boundary as follows: if s%cols!=0, this means that the agent was not on the left edge and executing a right action (s+a)%cols==0 will take it to the left edge. This means it was on the right edge and wanted to move off this right edge. Other checks are similar. We have also accounted for moving diagonally so the agent will not overstep the boundaries.

We have also accounted for different reward schemes that we might want to use later in other lessons. These are formulated as an array of 4 elements [intermediate, goal1, goal2, cliff] the first reward represents the reward the agent obtains if it is on any intermediate cell. Intermediate cells are non-terminal cells. Goals or terminal states are those that a task would be completed if the agent steps into them. By setting the goals array, we can decide which cells are terminal/goals. As we can see, there are two goals, this will allow us to deal with all the classical problems we will tackle in our RL treatments, but we could have set up more. So, our reward array's second and third elements are for the goals. The last entry is for a cliff. A cliff cell is a special type of non-terminal cell where the agent will emulate falling off a cliff and usually is given a high negative reward and then will be hijacked and put in its start position when it steps into these types of cells. These types of cells are non-terminal in the sense that the agent did not achieve the task when it went to them, but they provoke a reset of the agent position with a large negative reward.

The most important function in our class is the step(a) function. An agent will take a specific action in the environment via this function. Via this function, we return the reward from our environment and a flag (done) indicating whether the task is accomplished. This makes our environment compatible with the classic setup of an OpenAI Gym Atari games, which we deal with towards the end of our RL treatment.



## Conclusion
In this lesson, we covered the basics of RL functions and concepts that we will utilise in other lessons. We also provided you with a environment that you can directly utilise to build simple grid world environments. Please note that you are not required to study the Gris.py file or understand how the grid is programmatically built, but you need to understand how it operates.!

