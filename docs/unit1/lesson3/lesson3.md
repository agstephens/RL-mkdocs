Author: Abdulrahman Altahhan, 2025.

<script>
  window.MathJax = {
    tex: {
      tags: "ams",  // Enables equation numbering
    //   displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>


# Lesson 3- Markov Decision Processes and value functions

In this lesson we cover various grid world environments that are tackled as examples in the accompanying text book available online [here](http://incompleteideas.net/book/RLbook2020.pdf). Please note that we explain the ideas of this topic from a practical perspective and not from a theoretical perspective which is already covered in the textbook.

**Learning outcomes**

1. understand MDP and its elements
1. understand the return for a time step $t$
1. understand the expected return of a state $s$
1. understand the Bellman optimality equations
1. become familiar with the different types of grid world problems that we will tackle in later units
1. become familiar with the way we assign a reward to an environment
1. be able to execute actions in a grid world and observe the result
1. be able to to visualise a policy and its action-value function

**Reading**:
The accompanying reading of this lesson is **chapter 3** from our text book by Sutton and Barto available online [here](http://incompleteideas.net/book/RLbook2020.pdf). 



To help you, Abdulrahman recorded a set of videos that covers important concepts of RL.

## Markov Decision Processes (MDP)

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=f93d3c1e-261f-42bd-93cd-92f67e120d99&embed=%7B%22ust%22%3Atrue%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200" frameborder="0" scrolling="no" allowfullscreen title="Markov Decision Processes (MDP)" enablejsapi=1></iframe>


<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=79d3a9ea-5401-4f3c-bcf2-5907255ef8da&embed=%7B%22ust%22%3Atrue%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200"frameborder="0" scrolling="no" allowfullscreen title="2. Dynamics.mkv"></iframe>

## The Return $G_t$ of a time step $t$
We start by realising the 

$$
\begin{align*}
    G_t = R_{t+1} + &\gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + ... + \gamma^{T-t-1} R_{T} \\
    G_{t+1} =       & R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + ... + \gamma^{T-t-2} R_{T}
\end{align*}
$$

Hence by multiplying $G_{t+1}$ by $\gamma$ and adding R_{t+1} we get

$$
\begin{equation}
    G_t = R_{t+1} + \gamma G_{t+1}
\end{equation}
$$

**The above equation is the most important equation that we will build all of our incremental updates in RL on it.**

In the video below we talk more about this important concept.

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=e5a9acea-f258-4952-8e05-46f5ffb0c576&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200" frameborder="0" scrolling="no" allowfullscreen title="3. Returns 1"></iframe>

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=8a1a8b63-58be-45ce-86b1-eedb4bc133c4&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200"frameborder="0" scrolling="no" allowfullscreen title="3. Returns 2"></iframe>



### $G_t$ Monotonicity for MDP Rewards
Let us see how the return develops for an MDP with a reward of 1 or -1 for each time step.
To calculate $G_t$ we will go backwards, i.e. we will need to calculate $G_{t+1}$ to be able to calculate $G_t$ due to the incremental form of $G_t$ where we have that $G_t = R_{t+1} + \gamma G_{t+1}$.

- Mathematically, we can prove that $G_t$ is monotonically decreasing iff(if and only if) $\frac{R_{t}}{1 - \gamma} >  G_{t}$ $\forall t$ and $G_T=R_T > 0$. 
    - Furthermore, when $R_t=1$ $\forall t$ and $\gamma=.9$ then $G_t$ converges in the limit to 10, i.e. 10 will be an upper bound for $G_t$. 
    - Similarly, when $R_t=1$ $\forall t$ and $\gamma=.09$ then $G_t$ converges in the limit to 100
    - More generally, when $1-\gamma = 1/\beta$ then $R_t \beta > G_t$ 
- On the other hand, we can prove that $G_t$ is monotonically increasing iff $\frac{R_{t}}{1 - \gamma} <  G_{t}$.
    - Furthermore, when $R_t=-1$ $\forall t$ and $\gamma=.9$ then $G_t$ converges to -10, i.e. -10 is its lower bound. 
    - More generally, when $1-\gamma = 1/\beta$ then $R_t \beta < G_t$ 

Below we prove the former and leave the latter for you as homework.

$G_t = R_{t+1} + \gamma G_{t+1}$

We start by assuming that $G_t$ is strictly monotonically decreasing (we dropped the word strictly in th above for readability)

$G_t > G_{t+1} > 0$ $\forall t$ (which entails that $G_T=R_T > 0$ when the horizon is finite, i.e. ends at $t=T$) we substitute by the incremental form of $G_t$

$G_t > G_{t+1} > 0$ $\forall t \implies R_{t+1} + \gamma G_{t+1} >G_{t+1} \implies$  
$R_{t+1} >  G_{t+1} - \gamma G_{t+1} \implies$
$R_{t+1} >  (1 - \gamma) G_{t+1} \implies$

$\frac{R_{t+1}}{1 - \gamma} > G_{t+1}$ ( $\gamma \ne 1$)

The inequality $\frac{R_{t+1}}{1 - \gamma} >  G_{t+1}$ (which also can be written as $\frac{R_{t}}{1 - \gamma} >  G_{t}$) must be satisfied whenever $G_t$ is monotonically decreasing, i.e. it is a necessary condition. We can show that this inequality is also a sufficient condition to prove that $G_t$ is monotonically decreasing by following the same logic backwards. Similar things can be proven for the non-strictly monotonically decreasing case i.e. when $G_t\ge G_{t+1} \ge 0$ $\forall t$.

Now when $R_{t+1}=1$ and $\gamma=.9$, then by substituting these values in the inequality, we get that
$\frac{1}{1 - .9} >  G_{t+1} \implies$ $10 > G_{t+1}$ 

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

### $G_t$ Monotonicity for Sparse MDP Rewards

For sparse positive end-of-episode rewards, the above strict inequality is not satisfied since $R_t=0$ $\forall t<T$ and $R_T>0$.
1. In this case, we can show that $G_t \le G_{t+1}$ i.e. $G_t$ it is a monotonically increasing function.
    1. Furthermore, when $\gamma<1$ then $G_t$ is strictly increasing, i.e.  $G_t < G_{t+1}$
1. Furthermore, $G_{t} = \gamma^{T-t-1} R_{T}$.
    1. when $R_T=1$ then $G_{t} = \gamma^{T-t-1}$ 
    1. when $R_T=-1$ then $G_{t} = -\gamma^{T-t-1}$

- To prove the monotonicity we start with our incremental form for the return: 
    $G_t = R_{t+1} + \gamma G_{t+1}$:
    
    Since we have that $R_{t+1} = 0$ $\forall t<T$ then
    
    $G_t = \gamma G_{t+1}$ $\forall t<T$, therefore, since $\gamma \le 1$ then $G_t \le G_{t+1}$ $\forall t<T$.

- To prove that  $G_{t} = \gamma^{T-t-1} R_{T}$ we can also utilise the incremental form and perform a deduction, but it is easier to start with the general form of a return, we have:
    
    $G_t = R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + ... + \gamma^{T-t-1} R_{T}$
    
    Since we have that $R_{t+1} = 0$ $\forall t<T$ then
    
    $G_t = \gamma^{T-t-1} R_{T}$

This gives us guidance on the type of behaviour that we expect our agent to develop when we follow one of these reward regimes (sparse or non-sparse). 

The above suggests that for sparse end-of-episode rewards, decisions near the terminal state(s) have far more important effects on the learning process than earlier decisions. While for non-sparse positive rewards MDPs, earlier states have higher returns and hence more importance than near terminal states. 

If we want our agent to place more importance on earlier states, and near-starting state decisions, then we will need to utilise non-sparse (positive or negative) rewards. Positive rewards encourage repeating certain actions that maintain the stream of positive rewards for the agent. An example will be the pole balancing problem. Negative rewards, encourage the agent to speed up towards ending the episode so that it can minimise the number of negative rewards received.

When we want our agent to place more importance for the decisions near the terminal states, then a sparse reward is more convenient. Sparse rewards are also more suitable for offline learning as they simplify the learning and analysis of the agent's behaviour. Non-sparse rewards suit online learning on the other hand, because they give a quick indication of the agent behaviour suitability and hence speed up the early population of the value function. 



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

## The Expected Return Function V
Once we move form an actul return that comes froma an actual experience at time step $t$ to try to estimate this return, we move to an expectaiton *function*. This function, traditionally called the value function v, is an important function. But now isntead of tying the value of the return to a particular experience at a step t which would be less useful in generalising the lessons an agent can learn from interacting with the environment, it makes more sense to ty this up to a certain state $s$. This will allow the agent to learn a useful expectation of the return(discounted sum of rewards) for a particualr state when the agent follows a policy $\pi$. I.e. we are now saying that a we will get an expected value of the return for a particular state under a policy $\pi$. So we moved from subscripting by a step $t$ into passing a state $s$ to the function and subscripting by a policy.


$$
\begin{equation}
    v_{\pi}(s) = \mathbb{E}_{\pi}(G_t)   \label{eq:v}  %\tag{1}
\end{equation}
$$


Equation \(\eqref{eq:v}\) gives the definition of v function.



In the following video we tackle this idea in more details.

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=7b8178ed-68d1-4335-8ab7-3d81f214f362&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="940" height="200"frameborder="0" scrolling="no" allowfullscreen title="4. Returns Expectation and Sampling.mkv"></iframe>

## Bellman Equation for V and Q

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=6d6d9455-7174-447a-8bcb-eceaa51a4af5&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200" frameborder="0" scrolling="no" allowfullscreen title="5. Bellman v.mkv"></iframe>

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=3a18cbb0-6960-42c1-bcf4-8e0893c09c89&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200"frameborder="0" scrolling="no" allowfullscreen title="6. Bellman q simple.mkv"></iframe>

## Bellman Optimality Equations for V and Q

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=b73eab99-7af2-4b9e-8909-19492615d273&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200" frameborder="0" scrolling="no" allowfullscreen title="7. Bellman Optimality 1.mkv"></iframe>

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=8d89893b-6e99-4380-a31e-93e2974cd04a&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="470" height="200" frameborder="0" scrolling="no" allowfullscreen title="7. Bellman Optimality 2.mkv"></iframe>

You can adjust the video settings in SharePoint (speed up to 1.2 and reduce the noise if necessary)

*Exercise 1*: If you realise there is a missing symbol in the [video: Bellman Equation for v] last equations, do you know what it is and where it has originally come from?

*Exercise 2*: Can you derive Bellman Optimality Equation for $q(s,a)$ from first principles?

[video:  Bellman Optimality for q from first principles](https://leeds365-my.sharepoint.com/:v:/g/personal/scsaalt_leeds_ac_uk/EVBv-P5S4_VKqFt_E0vikIUBdpV1BZX2V-IDM3ROXDDV4A?e=YQQchV)


## Grid World Environments

Ok, so now we are ready to tackle the practicals, please go ahead and download the worksheet and run and experiement with the provided code to build some grid world environments and visualise them and make a simple robot agent takes some steps/actions within these environments!.

You will need to download a python library (Grid.py) that we bespokley developed to help you run RL algorithms on toy problems and be abel to easily visualise them as needed, the code is optimised to run efficiently and you will be able to use these environmnets to test different RL algorithms extensively. Please place the library in the same directory of the worksheet. In general it would be a good idea to place all worksheets and libraries provided in one directory. This will make importing and runing code easier and more streamlined.


In a grid world, we have a set of cells that the agent can move between them inside a box. The agent can move left, right, up and down. We can also allow the agent to move diagonally, but this is uncommon. 

 We needed to be as efficient as possible, and hence we have chosen to represent each state by its count, where we count from the bottom left corner up to the right top corner, and we start with 0 up to nS-1 where nS is the number of states. This will allow us to streamline the process of accessing and storing a state and will be of at most efficiency. We also deal with actions similarly, i.e. each action of the nA actions is given an index 0..nA-1. For the usual grid, this means 0,1,2,3 for actions left, right, up and down. We represent a 2-d grid by a 1-d array, and so care must be taken on how the agent moves between cells. 

Moving left or right seems easy because we can add or subtract from the *current* state. But when the agent is on the edge of the box, we cannot allow for an action that takes it out of the box. So if the agent is on the far right, we cannot allow it to go further to the right. To account for this issue, we have written a valid() function to validate an action. Moving up and down is similar, but we need to add and subtract a full row, which is how many columns we have in our grid. the valid() function checks for the current state and what would be the next state, and it knows that an agent will overstep the boundary as follows: if s%cols!=0, this means that the agent was not on the left edge and executing a right action (s+a)%cols==0 will take it to the left edge. This means it was on the right edge and wanted to move off this right edge. Other checks are similar. We have also accounted for moving diagonally so the agent will not overstep the boundaries.

We have also accounted for different reward schemes that we might want to use later in other lessons. These are formulated as an array of 4 elements [intermediate, goal1, goal2, cliff] the first reward represents the reward the agent obtains if it is on any intermediate cell. Intermediate cells are non-terminal cells. Goals or terminal states are those that a task would be completed if the agent steps into them. By setting the goals array, we can decide which cells are terminal/goals. As we can see, there are two goals, this will allow us to deal with all the classical problems we will tackle in our RL treatments, but we could have set up more. So, our reward array's second and third elements are for the goals. The last entry is for a cliff. A cliff cell is a special type of non-terminal cell where the agent will emulate falling off a cliff and usually is given a high negative reward and then will be hijacked and put in its start position when it steps into these types of cells. These types of cells are non-terminal in the sense that the agent did not achieve the task when it went to them, but they provoke a reset of the agent position with a large negative reward.

The most important function in our class is the step(a) function. An agent will take a specific action in the environment via this function. Via this function, we return the reward from our environment and a flag (done) indicating whether the task is accomplished. This makes our environment compatible with the classic setup of an OpenAI Gym Atari games, which we deal with towards the end of our RL treatment.

## Your turn
Go ahead and play around with some grid world environment by executing and experiementing with the code in the following worksheet.

<!-- <a href="Grid.py" download> Grid world library</a> -->

[worksheet3](../../workseets/worksheet3.ipynb)

## Conclusion
In this lesson, we covered the basics of RL functions and concepts that we will utilise in other lessons. We also provided you with a environment that you can directly utilise to build simple grid world environments. Please note that you are not required to study the Gris.py file or understand how the grid is programmatically built, but you need to understand how it operates.!

