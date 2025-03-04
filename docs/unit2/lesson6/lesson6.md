# Lesson 6-Tabular Methods: Monte Carlo

**Learning outcomes**

By the end of this lesson, you will be able to: 

1. understand the difference between learning the expected return and computing it via dynamic programming
1. understand the strengths and weaknesses of MC methods
1. appreciating that MC methods need to wait till the end of the task to obtain its estimate of the expected return
1. compare MC methods with dynamic programming methods
1. understand the implication of satisfying and not satisfying the explore-start requirement for the MC control and how to mitigate it via the reward function
1. understand how to move from prediction to control by extending the V function to a Q function and make use of the idea of generalised policy iteration-GPI
1. understand how policy gradient methods work and appreciate how they differ from value function methods

## Overview
In this lesson, we develop the ideas of Monte Carlo methods. Monte Carlo methods are powerful and widely used in settings other than RL. You may have encountered them in a previous module where they were mainly used for sampling. We will also use them here to sample observations and average their expected returns. Because they average the returns, Monte Carlo methods have to wait until *all* trajectories are available to estimate the return. Later, we will find out that Temporal Difference methods do not wait until the end of the episode to update their estimate and outperform MC methods.

Note that we have now moved to *learning* instead of *computing* the value function and its associated policy. This is because we expect our agent to learn from *interacting with the environment* instead of using the dynamics of the environment, which is usually hard to compute except for a simple lab-confined environment. 

Remember that we are dealing with *expected return*, and we are either finding an *exact solution for this expected return* as when we solve the set of Bellman equations or finding an *approximate solution for the expected return* as in DP or MC.
Remember also that the expected return for a state is the future cumulative discounted rewards given that the agent follows a specific policy.

One pivotal observation that summarises the justification for using MC methods over DP methods is that it is often the case that we are able to interact with the environment instead of obtaining its dynamics due to its complexity and intractability. In other words, interacting with the envoronment is often more direct and easier than obtaining the model of the environment. Therefore, we say that MC are model-free methods.

## Plan

As usual, in general, there are two types of RL problems that we will attempt to design methods to deal with 

1. Prediction problem
For These problems, we will design Policy Evaluation Methods that attempt to find the best estimate for the value function given a policy.


2. Control problems 
For These problems, we will design Value Iteration methods that utilise Generalised Policy Iteration. They attempt to find the best policy by estimating an action-value function for a current policy and then moving to a better and improved one by often choosing a greedy action. They minimise an error function to improve their value function estimate, used to deduce a policy.
We will then move to Policy Gradient methods that directly estimate a useful policy for the agent by maximising its value function.


We start by assuming that the policy is fixed. This will help us develop an algorithm that predicts the state space's value function (expected return). Then we will move to the policy improvement methods, i.e. these methods that help us to compare and improve our policy with respect to other policies and move to a better policy when necessary. Then we move to the control case (policy iteration methods).


<!-- ## Monte Carlo Methods -->




### Bellman Equations(reminder)


As we have seen in a previous lesson, the Bellman equations form the foundation of many RL methods. They define the relationship between the value of a state and the values of successor states.

For a given policy \( \pi \), the Bellman Equation for the state-value function is:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s \right]
$$

For the action-value function \( Q^\pi(s, a) \):

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \right]
$$

### Dynamic Programming is Model based

As we have seen in the last lesson, Dynamic Programming uses Bellman equations as update rules to iteratively compute value functions and policies using two key steps:

1. **Policy Evaluation**: Uses the Bellman equation to obtain an evaluation for a given policy.
2. **Policy Improvement**: Uses the Bellman optimality equation to update the policy towards a better policy.

However, DP has a major limitation: it requires complete knowledge of the dynamics model \( p(s',r | s, a) \), or transition model \( p(s'| s, a) \) which is often unavailable in real-world scenarios. In other words, it is a model-based methods.

## Monte Carlo: Model-Free Methods

Monte Carlo (MC) methods are a class of RL algorithms used to estimate value functions and optimise policies by using **sampled episodes** instead of full models of the environment. Unlike Dynamic Programming (DP), which requires a known dynamics model \( p(s',r | s, a) \), Monte Carlo methods learn **directly from experience** by averaging observed rewards.

Monte Carlo (MC) methods estimate value functions without requiring a dynamics model. Instead, they rely on sampled episodes of experience. The key idea is to approximate value functions by averaging observed returns over multiple episodes.

The fundamental principle behind MC methods is that averaging samples from a distribution approximates its expectation. Since state-value and action-value functions are defined as expected returns, averaging returns over a sufficiently large number of episodes provides a reliable estimate of these functions.

There are some technical conditions that must be met for this approximation to hold, which we will reference when necessary.

<!-- ### Return Definition -->

For an episode consisting of states, actions, and rewards:

\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
\]

The Monte Carlo estimate of \( V(s) \) is the average return from all episodes where state \( s \) was visited.

\[
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)
\]

Where:

- $i$ refers to the index over the episodes, not the time steps.
- \(G_i(s)\) is the return (sum of discounted rewards) observed from episode $i$, starting at state $s$.
- $N$ is the total number of episodes used to average the returns.


## First-Visit Monte Carlo Policy Evaluation (prediction) 
Because MC methods depend entirely on experience, a natural way to approximate the cumulative future discounted reward is by taking their average once they become available through experience. So we must collect the cumulative *future* discounted reward once this experience has elapsed. In other words, we need to take the sum *after* the agent has finished an episode for all the rewards obtained from the current state to the end of the episode. Then we average those returns over all of the available episodes. Note that MC methods only apply for episodic tasks, which is one of its limitations in addition to having to wait until the episode is finished.

Note also that the agent can visit the same state more than once inside the same episode. One question that arises from the above averaging is: which visit should be counted? 

We can take the sum starting from the first visit, or every visit, each yields a different algorithm. The first-visit algorithm is more suitable for tabular methods, while the every-visit algorithm is more suitable when using function approximation methods (such as neural networks).

One could argue that since we want to estimate the value of a state based on the full horizon of rewards obtained until the end of an episode, we should include only the *first visit of the state* in the average. This is the basis of the First-visit Monte Carlo (MC) policy evaluation method. First-visit MC estimates the value of a state by averaging the returns from the first time that state is encountered in each episode. Below we show the pseudocode for this algorithm.

\[
\begin{array}{ll}
\textbf{Algorithm: }  \text{First-Visit Monte Carlo Policy Evaluation} \\
\textbf{Input: } \text{Episodes generated under policy } \pi \\
\textbf{Initialize: }  V(S) \leftarrow 0, N(S) \leftarrow 0, \forall S \in \mathcal{S} \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, A_0, R_1, S_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{If } S_t \text{ appears first in the episode:} & \\
\quad \quad \quad N(S_t) \leftarrow N(S_t) + 1 & \\
\quad \quad \quad V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}(G - V(S_t)) & \\
\textbf{Return: } V(S), \forall S \in \mathcal{S} \\
\end{array}
\]


### Random Walk Problem
The Random Walk problem is a simple **Markov Reward Process (MRP)** used to illustrate value estimation methods. It consists of a finite, linear chain of states, where an agent moves randomly left or right until reaching one of the two terminal states.

#### Problem Setup
- Typically, there are five non-terminal states labeled \( A, B, C, D, E \) on a 1-d grid world. Another variaiton use 21 states.
- Two terminal states exist at both ends.
- The agent starts in the center and moves *randomly* left or right with equal probability.
- The episode ends when the agent reaches a terminal state.
- A *reward of +1* is received upon reaching the right terminal state, while the left terminal state gives *0 reward*.
<!-- - The discount factor \( \gamma \) is used to determine state values. -->

This random walk problem is often used to demonstrate **Monte Carlo** and **Temporal-Difference (TD)** learning methods for estimating state-value function. Below we show this problem.

![png](output_9_0.png)  
  
            A     B     C     D     E

Let’s now run MC1st with a useful visualization. The plot displays the true state values of the random walk as black points connected by a black line. The agent’s estimated values are represented by blue points and a blue line, allowing us to visually assess how closely the learned values for states A–D match their actual values. This is acheived by passing plotV=True.

Next to this, an error plot tracks the total error per episode, providing insight into the learning process. Each episode begins in the middle state (C) and ends upon reaching either the far-left or far-right terminal states, which are unnamed since they do not have values to estimate. This is acheived by passing plotE=True.

```python
mc = MC1st(env=randwalk(), episodes=100, plotV=True, plotE=True, seed=1).interact()
```
![png](output_40_0.png)

We use **demoV to implicitly pass plotE=True, plotV=True, animate=True, whenever we want to demo a prediction algorithm.


## Policies
Before we move into control, we need to breifly discuss types of policies that balance exporation and exploitation.

### $\epsilon$-Greedy Policy

The $\epsilon$-greedy policy is a popular action selection strategy that balances exploration and exploitation. The policy chooses the action with the highest estimated value most of the time, but with probability \(\epsilon\), it selects an action randomly to encourage exploration.

Mathematically, the $\epsilon$-greedy policy can be defined as:

\[
\pi(a|s) = 
\begin{cases} 
\frac{\epsilon}{|A|}, & \text{with probability } \epsilon \\
1 - \epsilon + \frac{\epsilon}{|A|}, & \text{for the action with the highest value} \\
0, & \text{for all other actions}
\end{cases}
\]

Where:
- \(\epsilon\) is the probability of exploring (random action).
- \(|A|\) is the total number of actions available.
- The action with the highest value \(Q(s,a)\) is selected with probability \(1-\epsilon + \frac{\epsilon}{|A|}\).

### Softmax Policy

The softmax policy selects actions based on a probability distribution that is a function of the action values. The policy assigns a higher probability to actions with higher expected returns, and the probability of selecting action \(a\) in state \(s\) is proportional to the exponential of its action-value \(Q(s, a)\).

Mathematically, the softmax policy is given by:

\[
\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{b \in A} e^{Q(s,b)/\tau}}
\]

Where:
- \(\tau\) is the temperature parameter that controls the level of exploration. A high \(\tau\) encourages more exploration (more uniform distribution), and a low \(\tau\) leads to more exploitation (choosing the highest value action).
- \(Q(s,a)\) is the action-value function.

In this way, the softmax policy ensures that actions with higher values are more likely to be chosen, but there is always a chance to explore other actions.


## First-visit MC control 
Now let us extend our first-visit MC prediction to control by updating the Q action-value function instead of the state-value function V.

\[
\begin{array}{ll}
\textbf{Algorithm: }  \text{First-Visit Monte Carlo Control (Exploring Starts)} \\
\textbf{Input: } \text{Episodes generated under an exploring-starts policy} \\
\textbf{Initialize: }  Q(S, A) \leftarrow 0, N(S, A) \leftarrow 0, \forall S \in \mathcal{S}, A \in \mathcal{A}(S), \pi(S) \leftarrow \text{arbitrary policy}, \forall S \in \mathcal{S} \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, A_0, R_1, S_1, A_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{If } (S_t, A_t) \text{ appears first in the episode:} & \\
\quad \quad \quad N(S_t, A_t) \leftarrow N(S_t, A_t) + 1 & \\
\quad \quad \quad Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}(G - Q(S_t, A_t)) & \\
\quad \quad \text{Update policy: } \pi(S_t) \leftarrow \arg\max_A Q(S_t, A) & \\
\textbf{Return: } Q(S, A), \pi(S), \forall S \in \mathcal{S}, A \in \mathcal{A}(S) \\
\end{array}
\]


### Applying MC on a control problem

Similar to what we did for prediction, we get help from a dictionary that stores a set of useful configurations that we use often. In the case of control, the most useful is plotting the number of steps the agent took to reach a terminal state in each episode or the sum of rewards the agent collected in each episode. Each one of these plots can be useful for certain tasks. Bear in mind that if the reward is given only for reaching the goal location or terminal state, the sum of the rewards plot would be a constant line that does not convey useful information. Below we show each.


Unfortunately, applying the MC control algorithm with the default reward function will not yield a useful policy. This is because the explore-start condition is not satisfied (refer to section 5.4 of our book). In addition, averaging solutions may not perform well because they do not track a changing policy well for non-stationary problems (most of the control problems are non-stationary). To see this, uncomment the lines in the cell below and run it. (Note that we have set up the priorities of the actions in a way that will show this issue (right comes before left and down before up). demoQ is a dicitonary that passes visualisaiton values to the MDP control algorithm.


```python
mc = MC1stControl(env=grid(), γ=1, episodes=200,  seed=10, **demoQ()).interact()
```
![png](output_67_0.png)

### Exploration-Exploitation and Exploring Starts

- Exploration-Exploitation Tradeoff: In reinforcement learning, the agent must balance exploration (trying new actions to discover better long-term rewards) with exploitation (choosing known actions that yield high rewards based on current knowledge).
  
In exploration, the agent tries actions that it has not explored enough yet. This is necessary to gather more information about the environment. In exploitation, the agent uses its current knowledge to select actions that maximize the immediate reward based on its existing value estimates. Both of these needs to happen for the agent to learn effectively. $\epsilon$-Greedy and softmax policies has a built-in exploraiton-exploitation capabilities.

<!-- - $\epsilon$-Greedy Policy: A commonly used policy to handle exploration-exploitation:
    - With probability \( \epsilon \), the agent chooses a random action (exploration).
    - With probability \( 1 - \epsilon \), the agent chooses the action that maximizes the value estimate (exploitation).

- Softmax Policy: Instead of choosing actions randomly, the softmax policy selects actions based on a probability distribution derived from the action values. Actions with higher values are more likely to be chosen, but there is always a non-zero chance of selecting less optimal actions (exploration). -->

- Exploring Starts: This is a method used to ensure complete exploration of the state-action space. In exploring starts, each episode begins with a random state and a random action. This guarantees that over many episodes, every state-action pair will eventually be visited, ensuring unbiased value estimation. This condition dictates that all states must be randomly started with to gurantee convergance. It is needed for 1stMCC since it relies on the getting enough coverage for all of the states.

Exploring starts guarantees initial exploration and helps ensure all state-action pairs are visited, making it easier for the agent to discover good policies without biases. Exploration-exploitation policies (like $\epsilon$-greedy or softmax) are methods to balance exploration and exploitation during the agent's learning process. These policies allow for exploration to continue throughout the agent's learning, which is particularly useful when the environment is unknown. Exploration-exploitation balance plays an important role when we move into online methods. But they are still important to achieve an end of episode, and $\epsilon$-greedy is useful to avoid the exploring-start condition. 

In short, exploring starts ensures full exploration at the beginning of episodes, while exploration-exploitation policies control the balance between exploration and exploitation throughout the learning process.


### The role of the discount factor $\gamma$ for delayed reward
**Important Note**
It is always the case that when we use a *delayed reward* (which is the default reward for our Grid class), the discount factor $\gamma$ **must not be set to 1**. This is because the sum of the discounted rewards of each visited state will be equal to the delayed reward itself, which will not give any particular advantage to follow a shorter path, yielding a useless policy. Therefore, we can solve this issue 
1. either by providing a discounted value for $\gamma$ that < 1.
1. or by changing the reward to have intermediate steps reward, which, when accumulated, will provide distinguished sums for the different paths and hence help distinguish the shortest path or the policy that will yield an optimal reward.

### Solution 1
Below we show how we can simply reduce $\gamma$ to solve this issue.

```python
mc = MC1stControl(env=grid(), γ=.99, episodes=30, seed=10, **demoTR()).interact()
``` 
![png](output_70_0.png)

### Solution 2
Also we can compensate for the above issue, we would need to set up a reward function that allows the agent to quickly realise when it stuck in some not useful policy.

```python
env1 = grid(reward='reward_1')
mcc = MC1stControl(env=env1, episodes=30, seed=0, **demoTR()).interact()
```    
![png](output_72_0.png)


Compare the above policy with the one produced by the DP solution in lesson 2. You will notice that the MC solution does not give a comprehensive solution from all states because we do not start from different cells. The starting position is fixed. The exploration nature of the policy allowed the agent to develop an *understanding* through its Q function of where it should head if it finds itself in a specific cell. The Markovian property is essential in guaranteeing that this can be safely assumed.

You might have noticed that although the task is very straightforward, the agent detoured a bit from the simplest straight path that leads to the goal. Bear in mind that we are adopting an εgreedy policy by default, which means that the agent will take some explorative actions 10% of the time. But this should not have prevented the maxQ policy from pointing towards the goal. This is because of the nature of MC itself and its sampling averages. The next section demonstrates how we can overcome this difficulty.

We can play with the exploration but that needs lots of trail and is not straightforward.

```python
mc = MC1stControl(env=grid(), γ=.97, episodes=50, ε=.5, dε=.99, seed=20, **demoTR()).interact()
```

![png](output_76_0.png)


```python
mcc = MC1stControl(env=grid(reward='reward_1'), γ=.97, episodes=100, ε=0.9, dε=.999, seed=20, **demoTR()).interact()
```
    
![png](output_78_0.png)
    
## Every-visit MC Prediction
In every-visit Monte Carlo, the value of a state is updated based on the average of returns observed from *all* visits to that state, not just the first visit. For a state \( s \), let \( G_j(s) \) be the return from the \( j \)-th occurrence of \( s \) in some episode. The every-visit MC estimate of \( V^\pi(s) \) is given by: 

\[
V^\pi(s) \approx \frac{1}{N_s} \sum_{j=1}^{N_s} G_j(s)
\]

where:  
- \( N_s \) is the total number of times state \( s \) was visited across all episodes.  
- \( G_j(s) \) is the return obtained from the \( j \)-th visit to \( s \), computed as the sum of rewards from that visit to the end of the episode.  

### First-Visit vs. Every-Visit MC
|   | **First-Visit MC** | **Every-Visit MC** |
|---|------------------|------------------|
| **Update Rule** | Uses the return from the first visit to a state in each episode | Uses returns from **all** visits to a state in each episode |
| **Bias/Variance** | Lower bias but higher variance | Higher bias but lower variance |
| **Suitability** | Works well when visits to a state within an episode are correlated | Suitable when visits are independent and representative of overall state behavior |

<!-- ### When to Use Which? -->
<!-- - First-Visit MC is generally preferred when state occurrences within an episode are highly correlated, as it prevents bias due to repeated visits with varying returns.
- Every-Visit MC is computationally simpler and can work well if state visits are independent and uncorrelated.   -->

## Constant-α MC: Incremental every-visit MC Prediction
We move now into developing the every-visit MC prediction algorithm further, by allowing it to be incremental. Incrementality is a key aspect in RL, since it allows an algorithm to interact and learn from the environment at the same time in each step, without having to wait until the end of the experience. However, for MC since we have to obtain the returns Gt, we will wait until the end of the episode to obtain the full experience rewards, but then we will *apply* the updates incrementally, instead of accumulating the full set in one go by averaging. 

The first step towards realising this target is to use look into how the averaging can be adjusted gradually (incrementally). Let us assume that the we observed the state $s$ at time step $t$ with a return $G_t$ and the number of times $s$ has been visited is $N_s$. We want now to obtain a new estimate $V^\pi_{k+1}(s)$ based on our previous estimate $V^\pi_{k}(s)$. The number of ties $s$ was observed when we had the estimate $V^\pi_{k}(s)$ is $N_s-1$. From the estimate definition we have that: 

$$
    \begin{align*}
        V^\pi_{k+1}(s) =& \frac{1}{N_s} \left(\sum_{j=1}^{N_s-1} G_j(s) + G_t(s)\right) \\
                        =& \frac{1}{N_s} \left((1-N_s)V^\pi_{k}(s) + G_t(s)\right) \\
                        =& V^\pi_{k} (s) + \frac{1}{N_s} \left(G_t -  V^\pi_{k}(s) \right)
    \end{align*}
$$

This idea is similar to what we have discussed in the simple bandit algorihtm.
One issue we have we the averaging is that when $N_s$ becomes large the effect of the newly obtained $G_t$ diminishes with time since $\frac{1}{N_s}$ tends to 0 with time. One way to overcome this issue is by replacing $\frac{1}{N_s}$ with a constant $\alpha$ that is fixed.
This is much better for when the policy is not stationary (during learning an obtimal policy).

In other words, when we obtian the $G_t$ for each time step $t$, we can adjust the estimation of $s$ as follow:


\[
V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)
\]

This method updates the state-value function incrementally using a constant step-size parameter \( \alpha \) for each visit, allowing the agent to continuously refine its estimates as it observes more returns.

Where:

- \( V(s) \) is the value of state \(s\)
- \( G_t \) is the return (sum of discounted rewards) from state \( s \) onwards
- \( \alpha \) is the constant step-size parameter
  
This method is useful in scenarios where multiple visits to the same state provide valuable updates, and it works well for episodic tasks. Below we show the full algorithm in pseudocode.

\[
\begin{array}{ll}
\textbf{Algorithm: }  \text{Incremental Constant-}\alpha \text{ Monte Carlo Prediction} \\
\textbf{Input: } \text{Episodes generated under policy } \pi \\
\textbf{Initialize: }  V(s) \leftarrow 0, \forall s \in \mathcal{S}, 0 < \alpha \leq 1 \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, R_1, S_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{Update state-value estimate:} & \\
\quad \quad \quad V(S_t) \leftarrow V(S_t) + \alpha (G - V(S_t)) & \\
\textbf{Return: } V(s), \forall s \in \mathcal{S} \\
\end{array}
\]


<!-- ### Monte Carlo Control with \(\epsilon\)-Greedy Exploration

To optimize policies, we extend MC to **control**, using the **Generalized Policy Iteration (GPI)** framework. The key idea is:

1. **Policy Evaluation:** Estimate \( Q^\pi(s, a) \) using MC.
2. **Policy Improvement:** Use an **\(\epsilon\)-greedy** strategy to gradually improve the policy.

### Algorithm:

\[
\begin{array}{ll}
\textbf{Algorithm:} & \text{Monte Carlo Control with } \epsilon \text{-greedy policy} \\
\textbf{Initialize:} & Q(s, a) \leftarrow 0, N(s, a) \leftarrow 0, \forall s, a \\
\textbf{For each episode:} & \\
\quad \text{Generate an episode using } \pi_{\epsilon} & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step t from T to 1:} & \\
\quad \quad G \leftarrow \gamma G + r_{t+1} & \\
\quad \quad \text{If } (s_t, a_t) \text{ appears first in episode:} & \\
\quad \quad \quad N(s_t, a_t) \leftarrow N(s_t, a_t) + 1 & \\
\quad \quad \quad Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{1}{N(s_t, a_t)}(G - Q(s_t, a_t)) & \\
\quad \quad \text{Update policy } \pi \text{ using } \epsilon \text{-greedy:} & \\
\quad \quad \quad \pi(s) \leftarrow \arg\max_a Q(s, a) \text{ with probability } (1-\epsilon) & \\
\quad \quad \quad \text{Choose random action with probability } \epsilon & \\
\textbf{Return:} & Q(s, a), \pi(s) \forall s, a \\
\end{array}
\] -->






## MRP, MDP and PG classes
We will evaluate the effectiveness of prediction methods by applying them to a random walk problem. This setup isolates the prediction aspect of an algorithm, focusing purely on estimating the state-value function without involving decision-making. By doing so, we can assess whether a given update rule or algorithm works effectively in the prediction setting.

Once we understand the prediction process, we can extend these methods to control by modifying the update rule to incorporate action-value estimates (Q-values). This transition is typically done within the MDP framework and is a key step in value-based reinforcement learning methods.

Beyond value-based approaches, there is another class of methods called policy-based or *policy gradient (PG)* methods, which optimise the policy directly instead of using a value function. Unlike value-based methods, policy gradient approaches are typically applied to control problems, such as grid-world mazes, rather than random walk problems.

**We use a parent class MRP for any prediciton model-free method which has access to V. We use MDP parent class for any control model-free method, which has access to Q as well as to an $\epsilon-$greedy policy, while the MRP have no policy (it is actually an arbitrary policy). We use PG class for any policy-gradient method, PG has access to both V and Q as well as to a softmax policy.**


### Incremental constant-α MC (prediction) with Python
Below we show a direct *interpretaiton* of the above pseudocode prediction model-free method into a python code. We use MRP as its parent class.

```python
class MC(MRP):
    def init(self):
        self.store = True # stores the trajectory of the episode, always needed for offline
    # ------------------ 🌘 offline, MC learning: end-of-episode learning ------------------
    def offline(self): # called at the end of the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):          # go throgh experience backwards as per Gt in update
            s, rn = self.s[t], self.r[t+1]       # retrieve the state and reward for past step t 
            Gt = self.γ*Gt + rn                  # calculate the return for past time step t
            self.V[s] += self.α*(Gt - self.V[s]) # update the state-value function
```

This type of algorithmic design is more flexible and will be used in general in RL instead of the implementation that requires storing the sums or averages.

Let us try our new shiny prediction algorithm on the random walk problem.

```python
mc = MC( α=.02, episodes=50, **demoV()).interact()
``` 
![png](output_101_0.png)
    

## Incremental MCC: Every-visit MC Control

**Incremental MCC (Monte Carlo Control)** for **Every-visit MC** Control is an approach used for estimating optimal policies through interaction with the environment. It incrementally updates both the state-action value function \( Q\) and the policy using every visit to a state-action pair. The method involves updating the action-value function based on the observed returns, and it uses a constant step-size parameter \( \alpha \) to control the magnitude of updates:

\[
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( G_t - Q(S_t, A_t) \right)
\]

Where:

- \( Q(s, a) \) is the action-value function,
- \( G_t \) is the return from state-action pair \( (s, a) \),
- \( \alpha \) is the constant step-size parameter.

This method is applied iteratively to improve the policy, choosing actions greedily with respect to the estimated \( Q \)-values. Below we show the pseudocode for thsi algorithm.

\[
\begin{array}{ll}
\textbf{Algorithm: }  \text{Incremental Constant-}\alpha \text{ Monte Carlo Control} \\
\textbf{Input: } \text{Episodes generated under an } \varepsilon\text{-greedy policy } \pi \\
\textbf{Initialize: }  Q(S, A) \leftarrow 0, \forall S \in \mathcal{S}, A \in \mathcal{A}(S), 0 < \alpha \leq 1 \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, A_0, R_1, S_1, A_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{Update action-value estimate:} & \\
\quad \quad \quad Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G - Q(S_t, A_t)) & \\
\quad \quad \text{Update policy: } \pi(S_t) \leftarrow \arg\max_A Q(S_t, A) \text{ (with } \varepsilon\text{-greedy exploration)} & \\
\textbf{Return: } Q(S, A), \pi(S), \forall S \in \mathcal{S}, A \in \mathcal{A}(S) \\
\end{array}
\]


### Incremental constant-α MC with Python
Below we show a direct *interpretaiton* of the above pseudocode into a python code. We use a parent class MDP parent class for this control model-free method.

```python
class MCC(MDP()):
    def init(self):
        self.store = True
    # ------------------ 🌘 offline, MC learning: end-of-episode learning 🧑🏻‍🏫 ---------------
    def offline(self):  # called at the end of the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s, a, rn = self.s[t], self.a[t], self.r[t+1] # retrieve the state, action and reward for past step t 
            Gt = self.γ*Gt + rn                          # update Gt incrementally
            self.Q[s,a] += self.α*(Gt - self.Q[s,a])     # update the action-value function
```


```python
mcc = MCC(env=grid(reward='reward1'), α=.2, episodes=1, seed=0, **demoQ()).interact()
```
    
![png](output_105_0.png)
    

<!-- ### Apply incremental MC on control problem -->

As we can see, although we solved the issue of tracking a non-stationary policy when we used a constant learning rate α, and we tried to use a reward function that gives immediate feedback to each step instead of a delayed reward, but still the performance is not as good as we wished for. This is due to our final issue, which is the action precedence that we set up to prefer left over right. If we change this precedence, it will help the agent to immediately find the goal, however, we set it up this way to make the problem more challenging. Consider changing this precedence to see the effect.


## REINFORCE: MC for Policy Gradient

So far, we have only seen how to estimate a value function to deduce a policy from this value function and then improve the policy by preferring a greedy action with a bit of exploration (as in ε-greedy policy). When we allow the agent to act according to this new policy, its value function might change, so we must re-estimate the value function. We go into iterations of this process until the policy and value function are both stable (converge). We also saw that we could integrate both operations seamlessly into one iteration, as in the value-iteration algorithm in Dynamic Programming. We can even do both stages in one *step* as in Q-learning or Sarsa, as we shall see in the next lesson. The policy improvement theorem and the Generalised Policy Iteration process guarantee all of this. The primary approach we took to achieve learning for an **action-value** method is to **minimise an error function** between our estimate of a value function and the actual value function. Since the real value function is unavailable, we replaced it with some samples (unbiased as in MC and biased as in TD that we will see later).

**Policy gradient** algorithms, on the other hand, attempt to **maximise an objective function** instead of minimising an error function. 
Can you think of a function that, if we maximise, will help us solve the RL problem...? pause for a moment and think.

As you might have guessed, the value function can be used as an objective function. The objective here is to change the policy to maximise the value function. 

Directly estimating the policy means we are not using a value function to express the policy as in the e-greedy. Instead, we are using the value function to learn the policy directly. So, our algorithm does not need to learn the value function explicitly; it can learn a set of parameters that will maximise the value function without knowing what the value function is. It will come as a consequence of learning a policy. In the same way that we did not need to learn a policy in the value-function approach, we learned a value function, and as a consequence of minimising the error, we can deduce a policy from the learned value function. This is the fundamental difference between value function approaches and policy gradient approaches.

Estimating the policy directly means we do not need to restrict the policy parameters to value function estimates and their ranges. The policy parameters that represent the preferences to select an action are free to take on any range of values as long as they comparatively form a cohesive policy that maximises the value function by dictating which action to choose in a specific state. This is a major advantage because the value function is strictly tied to the sum of rewards values, while a policy need not have this coupling. This will give us more freedom in using classification architectures when we use function approximation which excels in deducing the best action for a state, instead of using a regression architecture to regress a value function which is usually more prone to initial condition issues and are harder to train.

The best policy representation in a policy gradient method is the action selection softmax policy we came across in our last few lessons. This is a smooth function that, unlike ε-greedy, allows the changes in the probabilities to be continuous and integrates very well with policy gradient methods. One of the significant advantages of policy gradient methods (the policy is differentiable everywhere, unlike stepwise ε-greedy functions) is that it provides better guarantees of convergence than ε-greedy due to this smoothness (ε-greedy can change abruptly due to small changes in the action-value functions, while softmax just smoothly increases or decrease the probability of selecting ana action when its action-value function changes).

We start our coverage for policy gradient methods with an offline method; REINFORCE. REINFORCE is an algorithm that takes a *policy gradient* approach instead of an action-value function approach. The idea is simple, given that an episode provides a sample of returns for the visited states, at the end of an episode, we will take the values of the states and use them to guide our search to find the optimal policy that maximises the value function. 

<!-- **Note** that policy gradient sections in this lesson, and the next are based on chapter 13 of our book. They can be read as they appear in the notebook or delayed until the end of lesson 9. -->

<!-- ### Policy Gradient Class -->
The softmax is the default policy selection procedure for Policy Gradient methods. $\tau$ acts like an exploration factor (more on that later) and we need to one-hot encoding for the actions.

Ok, so now we are ready to define our REINFORCE algorithm. This algorithm and other policy gradient algorithm always have two updates, one for V and one for Q. In other words, the action-value function update will be guided by the state-value update. We usually call the first update deals that with V, the critic and the second update that deals with Q the actor. Below we show the python code for this algorithm.


```python
class REINFORCE(PG()):
    def init(self):
        self.store = True
    # -------------------- 🌘 offline, REINFORCE: MC for policy gradient methdos ----------------------
    def offline(self):
        π, γ, α, τ = self.π, self.γ, self.α, self.τ
        # obtain the return for the latest episode
        Gt = 0
        γt = γ**self.t                           # efficient way to calculate powers of γ backwards
        for t in range(self.t, -1, -1):          # backwards
            s, a, rn = self.s[t], self.a[t], self.r[t+1]
            Gt = γ*Gt + rn                        # update Gt incrementally
            δ = Gt - self.V[s]                    # obtain the error
            self.V[s]   += α*δ                    # update V as per the error
            self.Q[s,a] += α*δ*(1 - π(s,a))*γt/τ  # update Q as per the erro with complement of the policy π
            γt /= γ

```

## The Role of Discount Factor $\gamma$ in Policy Gradient Methods
$\gamma$ seems to play a more important role in policy gradient methods than in action-value methods.
The next few examples show how $\gamma$ can make the difference between convergence and divergence.
The main issue is, as usual, whether the *reward* is delayed or there is an intermediate reward. If the reward is delayed, we would need to assign $\gamma$ values that are < 1 so that the sum of the rewards is discounted, which helps the agent differentiate between longer and shorter paths solution. However, $\gamma$ also plays a role in convergence when the reward is not delayed. It complements the role that $\tau$ plays in the SoftMax policy. Therefore, instead of tuning $\tau$ we can reduce $\gamma$ specifically when the goal reward is 0, and the intermediate reward is -1 (reward_0) function. Let us see some examples:

Below we increase the value of $\tau$ to deal with this issue of diveregnce.


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=2, γ=1, episodes=100, seed=10 , **demoQ()).interact()
```
    
![png](output_135_0.png)
    

As we can see REINFORCE converged when we increase $\tau$ which helped the values in SoftMax to become appropriatly smaller to help the algorithm to converge.

Let us now decrease the value of $\gamma<1$ and keep $\tau=1$


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=1, γ=.98, episodes=100, seed=10, **demoQ()).interact()
```
    
![png](output_140_0.png)
    


As we can see decreasing $\gamma$ helped REINFORCE immensely to converge. Although the reward that we used is 'reward_1' which is not delayed, but discounting the return helped the value function to be more meaningful for the problem in hand which helped in turn the policy to be more appropriate for the problem in hand.  

Let us now increase $\tau$ and keep $\gamma<1$ this will reveal another role for $\tau$.


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=2, γ=.98, episodes=100, seed=10, **demoQ()).interact()
```
    
![png](output_144_0.png)
    


As we can see increasing $\tau$ while using $\gamma <1$ did not help. We will mostly therefore use $\gamma <1$ for our policy gradient methods.  


Note how exploration lead to a fully covered environment but to a slower convergence.

## Conclusion
In this lesson, we studied the properties of Monte Carlo algorithms for prediction and control. We started by covering a basic first visit MC method that averages the returns similar to what we did in lesson 1, this time for the associative problem (i.e., when we have states that we select specific actions for, un-associated problems do not have states and have been studied in lesson 1). We have then created an incremental MC algorithm that allows us to average the returns in a step-by-step manner. To that end, we have developed an essential MRP class that will carry the step-by-step and episode-by-episode interaction with an MRP environment, and then we added a useful set of visualisation routines. We have further inherited the MRP class in an MDP class that defines policies that depend on the Q function to obtain a suitable policy for an agent (i.e., control).
We noted that MC needed to wait until the episode was finished to carry out updates. In the next unit, we will study full online algorithms that mitigate this shortcoming of MC with the cost of bootstrapping. We will be using the MRP and MDP classes that we developed here.


Monte Carlo methods provide a powerful alternative to Dynamic Programming for RL problems where the environment’s transition model is unknown. They estimate value functions from sampled episodes and improve policies using exploration strategies like \(\epsilon\)-greedy. We covered two types of methods:

- Policy Evaluation: Use MC to estimate state values.
- Policy Control: Improve policies using MC-based action-value estimates.

**Advantages**

- Model-free: No need to know the environment’s transition probabilities.  
- Simple and intuitive: Works by averaging sampled returns.  
- Works well for episodic tasks.  

**Limitations**

- Requires complete episodes, which may not always be feasible.  
- Convergence can be slow compared to Temporal-Difference (TD) methods.  


In the next unit, we will explore Temporal-Difference (TD) learning methods, which update state-value estimates without requiring complete episodes. Instead, TD methods rely on one-step updates, much like Value Iteration incrementally improves policies. We will also introduce TD-based action-value methods for control, including SARSA and Q-learning, which use the Q action-value function to learn optimal policies.

**Reading**:
For further reading you can consult chapter 5 from the Sutton and Barto [book](http://incompleteideas.net/book/RLbook2020.pdf). The policy gradient sections in this lesson, and the next are based on chapter 13 of our book. They can be read as they appear in the notebook or delayed until the end of lesson 9.



## Your turn
Now it is time to experiment further and interact with code in [worksheet6](../../workseets/worksheet6.ipynb).
