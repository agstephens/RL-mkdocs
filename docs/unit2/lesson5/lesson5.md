Author: Abdulrahman Altahhan, March 2024.

The notebook uses a library of functionality in RL that aims for simplicity and general insight into how algorithms work, these libraries are written from scratch using standard Python libraries (numpy, matplotlib etc.).
Please note that you will need permission from the author to use the code for research, commercially or otherwise.

# Lesson 5- Dynamic Programming: Model-Based Approach

**Unit 2: Learning Outcomes**  
By the end of this unit, you will be able to:  

1. **Compute** the value function for a given policy in tabular settings.  
2. **Implement** control methods that infer an agent’s policy from an action-value function.  
3. **Explain** the concept of Generalized Policy Iteration (GPI) and how it underpins many RL methods.  
4. **Compare** full-backup action-value-based control methods with direct policy estimation control methods.  
5. **Evaluate** how Monte Carlo (MC) methods provide unbiased but high-variance estimates through interaction with the environment.  
6. **Analyze** how REINFORCE achieves unbiased but high-variance policy gradient estimation through interaction with the environment.  

---

In the first lesson, you looked at a basic RL problem, the k-arm bandit, which involves only actions and no states (non-associative problem). In general, in RL, we are faced with different situations, and we need to take different actions in each situation in order to achieve a certain goal. This general type of environment with states and actions imposes a different flavour to the solution we can design. From now on, we will tackle associative problems. For associative problems, there are two approaches:

1. Model-based approach
2. Model-free approach

In this lesson, we will take the first approach. We will learn how to use a model of the environment to solve an RL problem. The model is given in the form of the dynamics of the environment. These usually come in the form of 4 dimensions of conditional probability involving an answer to the following question: what is the probability of obtaining a certain reward r in a certain state s' given that the agent was previously in a state s and applied action a.

We will assume that there is already a model for the environment and try to take advantage of this model to come up with the best policy. Nevertheless, we will see simple ways to build such models and come back to this question later when we tackle planning algorithms in RL.

**Reading**:
The accompanying reading of this lesson are **chapters 3 and 4** from our textbook by Sutton and Barto available online [here](http://incompleteideas.net/book/RLbook2020.pdf).


**Plan**
As usual, in general there are two types of RL problems that we will attempt to design methods to deal with 
1. Prediction problem
For These problems we will design Policy Evaluation Methods that attempt to find the best estimate for the value-function given a policy.


2. Control problems 
For These problems we will design Value Iteration methods which utilise the idea of Generalised Policy Iteration. They attempt to find the best policy, via estimating an action-value function for a current policy then moving to a better and improved policy by choosing a greedy action often.


To help you, Abdulrahman recorded a set of videos that covers the material

## Basics Dynamic Programming

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=55e740c0-8e7f-4735-b5cd-3a8911e9fc16&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="8. Dynamic Programming 1.mkv"></iframe>


<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=da8f61f0-2c2d-4ec2-9ac1-ef73d31de388&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="8. Dynamic Programming 2.mkv"></iframe>

## Policy Evaluation and Value Iteration

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=7523c957-a9cb-4e87-b55c-1c449702ba9c&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="9. policy evaluation.mkv"></iframe>

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=7523c957-a9cb-4e87-b55c-1c449702ba9c&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="9. policy evaluation.mkv"></iframe>


```python
%matplotlib inline
from Grid import *
```

We start by generating conditional probabilities from random joint probabilities, then we move into generating them from a model.

## 1D Probability
Let us start by defining a 1D probability function 


```python
def P(nS):
    p = np.zeros(nS)
    for s in range(nS):
        p[s] = rand()
        
    return p/p.sum()
p = P(5)
print(p, p.sum())

```

    [0.37548211 0.24436176 0.03478805 0.01064735 0.33472073] 1.0


## 2D Probability: Joint Probability
Let us move to defining a joint probability function 


```python
def P(nS,nA):
    p = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            p[s,a] = rand()
            
        
    # /p.sum() to make sure that this is a joint probability density, i.e. p.sum()==1
    return p/p.sum() 
p = P(5,2)
print(p, p.sum())
```

    [[0.17464052 0.12339209]
     [0.10853332 0.1073298 ]
     [0.03067933 0.03055074]
     [0.06509827 0.06226649]
     [0.15006026 0.14744918]] 0.9999999999999999


Note that p[s,a] in the above is interpreted as the probability of being in state *s* and taking action *a* at the same time

## 2D Probability: Conditional Probability
Let us move to defining a conditional probability function. Remember that a conditional probability must satisfy that Bayes rule.

In the below pr[sn,a] is interpreted as pr[sn|a] the probability of moving to state sn given that we took action a 


```python
def P(nS,nA):
    pr = np.zeros((nS,nA)) # joint
    p  = np.zeros((nS,nA)) # conditional
    
    # first create a joint probability
    for sn in range(nS):
        for a in range(nA):
            pr[sn,a] = rand()

    # to make sure that this is a joint probability density
    pr=pr/pr.sum() 
    
    # now create a conditional probability via Bayes rule
    for a in range(nA):
        p[:,a] = pr[:,a]/pr[:,a].sum()
            
            
    return p

p = P(5,2)
print(p)
print(p.sum(0))
```

    [[0.16261181 0.29707477]
     [0.19121227 0.1445241 ]
     [0.41842684 0.19879146]
     [0.14277596 0.06602533]
     [0.08497312 0.29358434]]
    [1. 1.]


## 4D Probability: Random Dynamic

Our goal now is to be able to obtain a probability that can represent a random environment dynamics. Such a probability has the form of p[sn,rn | s,a] which represents the probability of moving to next state sn and obtaining the reward rn given that the agent was in state s and took action a.

We generalise the above method of obtaining conditional probabilities from random joint probability density to 4-d. We have not dealt with 3-d cases for brevity. This time we will make it a bit more efficient by avoiding the four for loops and utilising the vectorised version of the rand() function in numpy and we convert to a conditional in-place to avoid having two probabilities p and pr.


```python
def dynrand(nS, nA, nR): # states, actions, rewards dimensions 
    #first joint: p[sn,rn, s,a]
    p  = np.random.rand(nS,nR,  nS,nA)
    p /= p.sum()
    
    #convert it to conditional: p[sn,rn| s,a] 
    for s in range(nS):
        for a in range(nA):
            p[:,:, s,a] /= p[:,:, s,a].sum()
        
    return p

p = dynrand(3,2,2)
print(p)
print(p.sum(0).sum(0))
```

    [[[[0.33929711 0.0608615 ]
       [0.24024123 0.02720279]
       [0.19929774 0.23603235]]
    
      [[0.28847357 0.25786912]
       [0.20711515 0.15895581]
       [0.1864589  0.03933985]]]
    
    
     [[[0.09390373 0.23816355]
       [0.09103142 0.22100381]
       [0.01013988 0.14884069]]
    
      [[0.05826466 0.13310033]
       [0.14192641 0.17366847]
       [0.2242753  0.32088934]]]
    
    
     [[[0.06008514 0.28096335]
       [0.14680999 0.2805682 ]
       [0.28111859 0.14183835]]
    
      [[0.15997578 0.02904214]
       [0.1728758  0.13860091]
       [0.09870959 0.11305943]]]]
    [[1. 1.]
     [1. 1.]
     [1. 1.]]


## Inducing the dynamics by interacting with the environment

Now we move into obtaining the dynamics from an actual environment instead of generating the dynamics randomly as we did earlier.

We will use mainly the random walk environment and the grid world environment and generate their dynamics. These are deterministic simple environments. Nevertheless, they are very useful to demonstrate the ideas of RL. 

Note that when we move to the real world the dynamics become much more complex and building or obtaining the dynamic becomes impractical in most cases. Therefore, towards that end instead of dealing directly with the environment's dynamics, we will see later how we can substitute this requirement by having to *interact* with the environment to gain *experience* which will help us *infer* a good *estimate* of the *expected* value function (discounted sum of rewards) which in turn will help us to *infer* a close to *optimal policy* for the task in hand. 

The exercise of dealing with probabilities and then using them in designing a Dynamic Programming solution is valuable since most of the other solutions utilise the basic ideas (policy iteration, value iteration algorithms and policy improvements theorem) that we cover here and will mainly show us that we can devise a form of Bellman equation that is suitable for interaction, where we use sampling, model-free algorithms instead of using probabilities(dynamics), model-based algorithms. 

Dynamic programming suffers from what Bellman described as the curse of dimensionality which indicates that the computational resources required to solve a problem grow exponentially with the dimensionality of the problem. So in our case, the dimensionality is the number of states (as well as actions and rewards). So for example if the dynamic programming solution computational complexity is $2^{|S|}$ and the number of states $|S|=10$ then it costs $2^{10}=1024$ but when the number of states $|S|$ grows to 100 the cost becomes $2^{100}=1267650600228229401496703205376$.


```python
print(2**10)
print(2**100)
```

    1024
    1267650600228229401496703205376


## Sources Stochasticity- Dynamics and Policy

One important point to make is that stochasticity comes from different elements of the MDP and from the policy itself.

1. There might be stochasticity in the dynamics at the state transition level,
   where applying action $a$ in a state $s$ may cause the agent to transition to different states, each with a different probability
1. There might be stochasticity in the dynamics at the reward level,
   where applying action $a$ in a state $s$ may result in different rewards, each with a certain probability
  
1. There might be stochasticity in the policy itself,
   where the policy applies different actions in a state $s$  with different probabilities
   
1. There might be stochasticity or randomness in observing the current state due to the complexity of the state space. 
   For example, when a robot moves around in the environment, after a while, we cannot reliably designate its position from its motor encoders even when we know the start position due to dead-reckoning. This is called partial observability, and there is a framework called BOMDP or partially observable MDP to tackle this problem. However, we will not study this branch. The field is divided about the necessity of BOMDP with a line of thought that considers that we can overcome this difficulty by encoding our states differently but staying in the MDP framework.

These sources of stochasticity dictate using suitable techniques to obtain the dynamics and to evaluate or improve stochastic and deterministic policy.


```python
def dynamics(env=randwalk(), show=False, stoch=True, repeat=1000 ): #, maxjump=1
    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    p  = np.zeros((nS,nR,  nS,nA))
    
    for i in range(repeat if not stoch else 1): # in case the env is stochastic (non-deterministic)
        for s in range(nS):
            # if s in env.goals: continue # uncomment to explicitly make pr of terminal states=0
            for a in range(nA):
                if not i and show: env.render() # render the first repetition only
                env.s = s
                sn,rn,_,_ = env.step(a)
                rd = np.where(rewards==rn)[0][0] # get which reward index we need to update
                p[sn,rd, s,a] +=1
            
    # making sure that it is a conditional probability that satisfies Bayes rule
    for s in range(nS):
        for a in range(nA):
            sm=p[:,:, s,a].sum()
            if sm: p[:,:, s,a] /= sm
            
    return p
```

Note that we excluded the terminal states so that their dynamics are always 0. This is not entirely necessary, since in our algorithm implementations we always initiate the terminal states' values to 0 and avoid updating them. 

Note also that states and actions can be immediately interpreted as indexes. Rewards on the other hand are allowed to be real values. Therefore, we need to obtain the index of the reward value that we need to update since p[sn, rn, s,a] expects an index in rn.

Let us now test our dynamics extraction procedure on a deterministic random walk environment:


```python
p = dynamics(env=randwalk(), show=True)
print(p.shape)
print(p.sum(0).sum(0))
# print(p)
```


    
![png](output_26_0.png)
    


    (7, 2, 7, 2)
    [[1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]]


The above shows that the probabilities of agent having been in state 0:6 (rows) and have gone left (column 0) all sum up to 1 except for the last terminal state 6 because once the agent is there it stays there and cannot move left. Also it shows equally that the probabilities of agent having been in state 0:6 and have gone right (column 1) all sum up to 1 except for the first terminal state 0 because once the agent is there it stays there and cannot move right. 

To further explain and dissect the generated probability dynamics we examine different states probabilities with different rewards.


```python
def print_dynamcis(s):
    print('-----------------------------Agent was in non-terminal state %d---------------------------------'%s)
    print('Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state %d and moved left(0)]'%s)
    print(p[:,0, s,0])

    print('Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state %d and moved right(1)]'%s)
    print(p[:,0, s,1])

    print('Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state %d and moved left(0)]'%s)
    print(p[:,1, s,0])

    print('Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state %d and moved right(1)]'%s)
    print(p[:,1, s,1])
```


```python
print_dynamcis(s=2)
```

    -----------------------------Agent was in non-terminal state 2---------------------------------
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 2 and moved left(0)]
    [0. 1. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 2 and moved right(1)]
    [0. 0. 0. 1. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 2 and moved left(0)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 2 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 0.]



```python
print_dynamcis(s=5)
```

    -----------------------------Agent was in non-terminal state 5---------------------------------
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 5 and moved left(0)]
    [0. 0. 0. 0. 1. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 5 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 5 and moved left(0)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 5 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 1.]


Note that the probability of the penultimate state is a bit different than a mid-state such as 2. The reason is that when the agent move left it will get a 0 reward while when it moves right it will get a reward of 1.


```python
print_dynamcis(s=0)
```

    -----------------------------Agent was in non-terminal state 0---------------------------------
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 0 and moved left(0)]
    [1. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 0 and moved right(1)]
    [0. 1. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 0 and moved left(0)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 0 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 0.]



```python
print_dynamcis(s=6)
```

    -----------------------------Agent was in non-terminal state 6---------------------------------
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 6 and moved left(0)]
    [0. 0. 0. 0. 0. 1. 0.]
    Pr[of moving to states 0:6 and obtaining reward 0 |give agent was in state 6 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 6 and moved left(0)]
    [0. 0. 0. 0. 0. 0. 0.]
    Pr[of moving to states 0:6 and obtaining reward 1 |give agent was in state 6 and moved right(1)]
    [0. 0. 0. 0. 0. 0. 1.]


Uncomment the xor relationship and see the effect on the resultant policy.

## Accommodating for jumps in the environment
Some environments might allow the agent to jump over some obstacles or simply skip cells. For these, we define slightly altered dynamics to take the jumps into account. Below we show the definition.


```python
def dynamics(env=randwalk(), stoch=False, show=False, repeat=1000): # , maxjump=1

    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    p  = np.zeros((nS,nR,  nS,nA))
    randjump = env.randjump
    env.randjump = False # so that probability of all intermed. jumps is correctly calculated
    for i in trange(repeat if stoch else 1): # in case the env is stochastic (non-deterministic)
        for s in range(nS):
            if s in env.goals: continue # uncomment to explicitly make pr of terminal states=0
            for a in range(nA):
                for jump in (range(1,env.jump+1) if randjump else [env.jump]):
                    if not i and show: env.render() # render the first repetition only
                    env.s = s
                    env.jump = jump
                    rn = env.step(a)[1]
                    sn = env.s
                    rn_ = np.where(rewards==rn)[0][0] # get reward index we need to update
                    p[sn,rn_, s,a] +=1
                    
    env.randjump = randjump
    # making sure that it is a conditional probability that satisfies Bayes rule
    for s in range(nS):
        for a in range(nA):
            sm=p[:,:, s,a].sum()
            if sm: p[:,:, s,a] /= sm
            
    return p
```

### State-Transition probability: reward marginalisation

Now that we have induced the dynamics from the *environment*, we are ready to infer the state-transition probability from the *dynamics*.  Note that many older papers of RL refer to the 3-d state-transition probability without referring to the 4-d dynamics. The state-transition probability p[sn | s,a] is a conditional probability that specifies the probability of moving to the next state sn given that the agent was in state s and applied action a regardless of the rewards. In other words, the state-transition probability does not refer to the reward altogether. Therefore, all we have to do to infer it from the dynamics is to marginalise the dynamics with respect to the reward, which is what the below code snippet is doing.


```python
# state-transition probability induced from the dynamics
def ssa(p): 
    # states dim, action dim
    nS, nA = p.shape[0], p.shape[3]
    tr = np.zeros((nS, nS, nA))
    for s in range(nS):
        for a in range(nA):
            for sn in range(nS):
                tr[sn, s,a] = p[sn,:,s,a].sum()
    return tr
```

Let us now apply this function on the obtained random walk dynamics.


```python
tr = ssa(p)
print(tr.sum(0))
# print(tr)
```

    [[1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]
     [1. 1.]]


Again the state-transition probability satisfies similar properties of the dynamics.

Finally, we can apply a similar logic to obtain the reward function, see page 49 of the book.


```python
# reward function induced from the dynamics
def rsa(p, rewards):
    # state dim, reward dim
    nS, nA, nR = p.shape[0], p.shape[3], p.shape[1]
    r = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
             for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                # print(rn_, rn)
                r[s,a] += rn*p[:,rn_, s,a].sum()
    return r

```

Note: that in the book authors use $r +\gamma v(s)$ to mean the reward at time step $t+1$, in fact for consistency they should have used $r'$ instead as they do for $s'$. In our code treatment we have chosen to use $rn$ in place of $r'$.


```python
rewards = randwalk().rewards_set()
r = rsa(dynamics(env=randwalk(),  show=True), rewards)
print(r)
```


    
![png](output_46_0.png)
    


    100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  2.05it/s]

    [[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 1.]
     [0. 0.]]


    


As we can see, the expected reward of the state-action pair (5, right)=1 because the agent obtains a reward of 1 when it is at state 5 and moves right (as it will end up in the far right terminal state with a reward of 1). The state-action pairs (6, right)=0 because the return/value for terminal states is always 0 (to be able to apply the recursive rule for the return, i.e. once the agent is at the terminal state 6 the process terminates and the agent will not receive any further reward).

We can do the same for the longer randwalk_()


```python
rewards = randwalk_().rewards_set()
r = rsa(dynamics(env=randwalk_(), show=True), rewards)
print(r)
```


    
![png](output_49_0.png)
    


    100%|█████████████████████████████████████████████| 1/1 [00:02<00:00,  2.54s/it]

    [[ 0.  0.]
     [-1.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  0.]
     [ 0.  1.]
     [ 0.  0.]]


    


Can you explain the numbers we get above.

# Dynamic Programming Methods

Ok so we are ready now to move to Dynamic programming algorithms to solve the RL problem of finding a best estimate of a value function and or finding an optimal policy.

## Policy evaluation

The first step to improving any policy is to evaluate how good or bad the policy is for the given task. This fundamental question can be addressed by tying up the task with a reward function that basically rewards the agent for achieving the task or a subtask that leads to the final goal. The agent's aim then becomes to collect as many rewards as possible (or to incur as few losses as possible), which should help the agent achieve the given task. One example is when a robot is moving in an environment, and we want it to reach a specific location, then we can reward/punish the robot for each step that is taking it close to the goal or away from it. But this awareness of the goal location is usually difficult to attain in real environments. Hence it is replaced by rewarding the agent when it reaches the goal or punishing the agent for each step taken without reaching the goal location.

We can devise an evaluation strategy based on the discounted sum of rewards the agent is *expected* to collect while executing the task. The strategy depends on the dynamics of the environment. You may want to read section 4.1 and come back here to continue reading the code for the policy evaluation algorithm to get an insight into how it works.


```python
def Policy_evaluation(env=randwalk(), p=None, V0=None, π=None, γ=.99, θ=1e-3, show=False): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.zeros(nS,int) if π  is None else np.array(π) # policy to be evaluated **stochastic/deterministic**

    i=0
    # policy evaluation --------------------------------------------------------------
    while True:
        Δ = 0
        i+= 1
        # show indication to keep us informed
        if show: clear_output(wait=True); rng = trange(nS) # 
        else: rng = range(nS)
        for s in rng:
            if s in env.goals: continue # only S not S+
            v, V[s] = V[s], 0            
            for sn in range(nS): # S+
                for rn_, rn in enumerate(rewards): # get the next reward rn and its index rn_
                    if π.ndim == 1: # deterministic policy
                        V[s] += p[sn,rn_, s,π[s]]*(rn + γ*V[sn])
                    else:           # stochastic policy 
                        V[s] += sum(π[s,a]*p[sn,rn_, s,a]*(rn + γ*V[sn]) for a in range(nA))
            Δ = max(Δ, abs(v-V[s]))
        if Δ<θ: break
    if show: 
        env.render(underhood='V', V=V)
        print('policy evaluation stopped @ iteration %d:'%i)

    return V
```

Note that we assume that when the policy is deterministic, it takes the shape (nS,) and its entries are actions ex. π[s]=1 means take right(1) when in state s.

On the other hand if the policy is probabilistic, it has a shape of (nS, nA) and its entries are probabilities of each action given a state s, i.e π[a|s] written as π[s,a] in the code.

Note that $\gamma$ must be $< 1$ to guarantee convergence of the Bellman equation because, in general, we do not know whether the policy guarantees reaching a terminal(goal) state; if we do, then $\gamma=1$ is ok. 

Refer to section 4.1 in the book: 'The existence and uniqueness of $v_\pi$ guaranteed as long as either  $\gamma< 1$ or eventual termination is guaranteed from all states under the policy $\pi$'.

This condition can be relaxed when we move to sampling instead of dynamic programming in the next consequent lessons.

Ok, let us test our policy evaluation method on a policy where the agent goes right always.


```python
Policy_evaluation(π=[-1,1,1,1,1,1,-1], show=True)
```


    
![png](output_59_0.png)
    


    policy evaluation stopped @ iteration 6:





    array([0.        , 0.96059601, 0.970299  , 0.9801    , 0.99      ,
           1.        , 0.        ])



As we can see, the values for the deterministic policy that always move the agent to the right produce a set of state values of 1 for the non-terminal states 1:5. This is what we expect since the accumulated return of this policy, given that γ=1, is 1 for these states since the agent is rewarded with 1 only when it reaches the far right terminal state, while when it is terminated at the far left state, it is rewarded with 0 (all other intermediate states also has a reward of 0).

To further test our policy evaluation method, we will pass a different policy to it. This time we pass a simple probabilistic policy that gives left and right actions the same probability of 0.5. In this case, since γ=1, the expected analytical values of the intermediate states are given as 1/6, 2/6, 3/6, 4/6, 5/6, which represent the probabilities of ending up in the far right state given the agent has started in them (so pr(s=1)=1/6).


```python
env = randwalk()
π = np.ones((env.nS,env.nA))*.5 # probability of moving left and right are equal
Policy_evaluation(env=env, π=π, show=True)
```


    
![png](output_62_0.png)
    


    policy evaluation stopped @ iteration 19:





    array([0.        , 0.14840421, 0.30070568, 0.45997221, 0.62919208,
           0.81145008, 0.        ])




```python
np.array([1/6, 2/6, 3/6, 4/6, 5/6])
```




    array([0.16666667, 0.33333333, 0.5       , 0.66666667, 0.83333333])



As we can see the values of this optimal policy (moving always right) progressively increase from left to right towards the goal that is most rewarding and the obtained values are exactly what we expected which is reassuring.

Let us try to evaluate a random policy


```python
env = randwalk()
π = np.random.randint(env.nA, size=env.nS, dtype=np.uint32)
print(π)
```

    [1 1 1 1 1 0 0]


Now we call our policy evaluaiton subroutine.


```python
Policy_evaluation(env=env, π=π, show=True)
```


    
![png](output_68_0.png)
    


    policy evaluation stopped @ iteration 1:





    array([0., 0., 0., 0., 0., 0., 0.])



As we can see the set of values are different than the above optimal policy and they are in harmony with the policy.

### Policy Evaluation for 2d MDP Grid World
Let us test our policy evaluation algorithm on the following simple 3x3 grid world


```python
env3x3 = Grid(gridsize=[3, 3], s0=0, goals=[8], figsize=[10,1])
π = [3, 1, 3, 1, 1, 3, 2, 2, -1] # guarantee to reach the terminal state hence γ=1 is ok
π_ = [3, 0, 0, 1, 2, 0, 0, 0, -1] # no way to terminal state hence γ=1 leads to an infinite loop
env3x3.render(underhood='π', π=π)
```


    
![png](output_71_0.png)
    



```python
V0 = [.6, .5, .5, .2, .1, .2, .1, .5, 0]
V = Policy_evaluation(env=env3x3, π=π, V0=V0, γ=.9, show=True)
env3x3.render(underhood='V', V=V)
```


    
![png](output_72_0.png)
    



```python
V = Policy_evaluation(env=env3x3, π=π, V0=V0, γ=1, show=True)
# V = Policy_evaluation(env=env3x3, π=π_, V0=V0, γ=1) #finite 
```


    
![png](output_73_0.png)
    


    policy evaluation stopped @ iteration 5:


Let us now try this on a slightly more complex environment such as a the maze world


```python
env = maze()
π = np.ones(env.nS, dtype=np.uint32)*3 # always go up
V = Policy_evaluation(env=env, π=π, show=True)
```


    
![png](output_75_0.png)
    


    policy evaluation stopped @ iteration 6:


As we can see moving up yeild some benefits mainly in the cells that lead to the goal.

Let us now generate a random policy and evaluate it for a maze environment.


```python
env = maze()
π = np.random.randint(env.nA, size=env.nS, dtype=np.uint32)
print(π)
```

    [1 0 1 2 1 3 2 1 0 1 1 2 3 1 0 3 1 2 3 3 3 3 3 2 0 0 2 1 0 1 0 3 0 0 3 3 0
     2 1 0 2 0 1 1 2 0 2 1 1 1 1 0 3 1]



```python
env.render(underhood='π', π=π)
```


    
![png](output_79_0.png)
    



```python
V = Policy_evaluation(env=env, π=π, show=True)
```


    
![png](output_80_0.png)
    


    policy evaluation stopped @ iteration 1:


As we can see, the randomly generated policy is chaotic and carry little value for the agent. However, evaluating different policies is highly important for an agent since it can guide the improvement of its adopted policy (based on this ability). One example is to keep evaluating random policies until some computational resources are consumed and pick the best. Below we show such a strategy of searching for an optimal policy. You can apply all other search algorithms that you have come across before in conventional AI (breadth-first etc.).


```python
env = maze()
Vmax= np.zeros((env.nS))
p = dynamics(env)
print('Vmax before search', Vmax.sum())
```

    100%|████████████████████████████████████████████| 1/1 [00:00<00:00, 945.09it/s]

    Vmax before search 0.0


    



```python
import random
random.seed(0)
np.random.seed(0)

for _ in trange(1000):
    # π = np.random.randint(env.nA, size=env.nS)
    π = choices(range(env.nA), k=env.nS)
    V = Policy_evaluation(env=env, π=π, p=p, show=False)
    if V.sum() > Vmax.sum(): 
        Vmax = V
        πmax = π

env.render(underhood='π', π=πmax)
print('Vmax after search', Vmax.sum())
```


    
![png](output_83_0.png)
    


    Vmax after search 10.704460219301


As we can see, finding the optimal policy by random search is difficult since the space of policies is huge, making exhaustive or random search infeasible (dimensionality problem). We need a way to take and maintain a step in the right direction of improving the policy. As you have already seen in another module, a greedy search can often lead to a good result. The next section shows a simple but effective strategy to gradually improve a policy by taking a greedy step towards the solution.

## Policy Iteration 
Now that we know how to evaluate a policy, it is time to improve it. Policy iteration is a basic and simple algorithm. It explicitly and iteratively tries first to reach a highly accurate estimate of the value function of the current policy, then it tries to improve the policy by maximising the probability of greedy actions as per the current value function. Evaluating the current policy fully and then improving it via policy iteration is inefficient, but it shows the fundamental ideas behind reinforcement learning. Please spend some time comprehending the code and reading the corresponding section 4.3 in the book.


```python
def Policy_iteration(env=randwalk(), p=None, V0=None, π0=None, γ=.99, θ=1e-3, show=False, epochs=None): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters 
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.zeros(nS,int) if π0 is None else np.array(π0); # initial **deterministic** policy 
    Q = np.zeros((nS,nA))  # state action values storage
    # π = randint(0,nA,nS)
    
    j=0
    while True if epochs is None else j<epochs:
        j+=1
        # 1. Policy evaluation---------------------------------------------------
        i=0
        while True if epochs is None else i<epochs:
            Δ = 0
            i+= 1
            for s in range(nS): 
                if s in env.goals: continue # S not S+
                v, V[s] = V[s], 0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        V[s] += p[sn,rn_,  s, π[s]]*(rn + γ*V[sn])

                Δ = max(Δ, abs(v-V[s]))
            if Δ<θ: print('policy evaluation stopped @ iteration %d:'%i); break
        
        # 2. Policy improvement----------------------------------------------------
        policy_stable=True
        for s in range(nS):
            if s in env.goals: continue # S not S+
            πs = π[s]
            for a in range(nA):
                Q[s,a]=0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V[sn]) 
            
            π[s] = Q[s].argmax() # simple greedy step
            if π[s]!=πs: policy_stable=False
           
        if policy_stable: print('policy improvement stopped @ iteration %d:'%j); break
        if show: env.render(underhood='π', π=π)
        
    return π, Q
```

We can now apply policy iteration to a proper Environments. We start by a random walk and then we move into grid world.


```python
env=randwalk()
π, Q = Policy_iteration(env, show=True)
```


    
![png](output_88_0.png)
    


    policy evaluation stopped @ iteration 2:
    policy improvement stopped @ iteration 6:


To visualise the policy we would need to pass the environment explicitly and then render the environment.


```python
print(π)
```

    [0 1 1 1 1 1 0]



```python
env.render(underhood='π', π=π)
```


    
![png](output_91_0.png)
    



```python
env.render(underhood='π', π=π)
```


    
![png](output_92_0.png)
    


Note that we do not need to break ties randomly for the policy (we jsu use Q[s].argmax()) because we are not sampling.

### 2-d Grid World MDP Examples
Let us take another example, of a simple 3x3 grid world and run the policy iteration algorithm on.
We start with a deterministic policy. The reward is 0 everywhere except for the goal where the agent will be rewarded by 1.



```python
env3x3 = Grid(gridsize=[3, 3], reward='reward_1', s0=0, goals=[8], figsize=[10,1])
π0 = [3, 1, 3, 1, 1, 3, 2, 2, -1] # guarantee to reach the terminal state only if γ<1
π0_ = [3, 0, 0, 1, 2, 0, 0, 0, -1] # guarantee to reach the terminal state only if γ<1
```

Whether the initial policy is guaranteed to reach the terminal state or not, γ=1 leads to an infinite loop, because subsequent policies are not guaranteed to reach the terminal state.



```python
 env3x3.render(underhood='π', π=π0)
```


    
![png](output_97_0.png)
    


Let us show the values before applying the agorithms


```python
V0 = [.6, .5, .5, .2, .1, .2, .1, .5, 0]
env3x3.render(underhood='V', V=V0)
```


    
![png](output_99_0.png)
    



```python
π, Q = Policy_iteration(env=env3x3, V0=V0, π0=π0, show=True)
π, Q
```


    
![png](output_100_0.png)
    


    policy evaluation stopped @ iteration 3:
    policy improvement stopped @ iteration 2:





    (array([ 1,  1,  3,  1,  1,  3,  1,  1, -1]),
     array([[-4.90099501, -3.940399  , -4.90099501, -3.940399  ],
            [-4.90099501, -2.9701    , -3.940399  , -2.9701    ],
            [-3.940399  , -2.9701    , -2.9701    , -1.99      ],
            [-3.940399  , -2.9701    , -4.90099501, -2.9701    ],
            [-3.940399  , -1.99      , -3.940399  , -1.99      ],
            [-2.9701    , -1.99      , -2.9701    , -1.        ],
            [-2.9701    , -1.99      , -3.940399  , -2.9701    ],
            [-2.9701    , -1.        , -2.9701    , -1.99      ],
            [ 0.        ,  0.        ,  0.        ,  0.        ]]))



Let us now check the values after applying the algorithm


```python
env3x3.render(underhood='V', V=Q.max(1))
```


    
![png](output_102_0.png)
    


Note that if we set γ=1 we need then to set epochs, otherwise, the algorithm enters an infinite loop because all state values converge to 1.

Let us try it on a slightly more complex environment such as the maze.


```python
env=maze()
π = Policy_iteration(env, show=True)[0]
```


    
![png](output_105_0.png)
    


    policy evaluation stopped @ iteration 2:
    policy improvement stopped @ iteration 16:



```python
env.render(underhood='π', π=π)
```


    
![png](output_106_0.png)
    


## Stochastic Policy Iteration
We can also create a stochastic policy iteration method. It only differs from the deterministic policy iteration in two places (in the if ε_greedy statements): 
1. in the policy evaluation, we need to marginalize by multiplying by the action probability as we did earlier for the policy evaluation method
2. In calculating the probability of taking an optimal action (to account for the stochastic non-determinism nature of the policy), we allow the agent to take random actions ε% of the time. 

We show the method below.



```python
np.random.seed(1) # change the seed to get a different dynamics
```


```python
# stochast policy iteration
def Policy_iteration_stoch(env=randwalk(), p=None, V0=None, π0=None, γ=.99, θ=1e-4, ε=.1, show=False): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS)      if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.ones ((nS,nA)) if π0 is None else np.array(π0); π=π/π.sum(1)[:,None] # initial **stochastic** policy 
    Q = np.zeros((nS, nA)) # state action values storage
    # π = randint(1,nA,(nS,nA))
    
    j=0
    while True:
        j+=1
        # 1. Policy evaluation---------------------------------------------------
        i=0
        while True:
            Δ = 0
            i+= 1
            for s in range(nS):
                if s in env.goals: continue # S not S+
                v, V[s] = V[s], 0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_ 
                        # stochastic policy
                        # V[s] += sum(π[s,a]*p[sn,rn_, s,a ]*(rn + γ*V[sn]) for a in range(nA)) 
                        V[s] += p[sn,rn_,  s, π[s].argmax()]*(rn + γ*V[sn])
                        
                Δ = max(Δ, abs(v-V[s]))
            if Δ<θ: print('policy evaluation stopped @ iteration %d:'%i); break

        # 2. Policy improvement----------------------------------------------------
        policy_stable=True
        for s in range(nS):
            if s in env.goals: continue # S not S+
            Qs = Q[s]
            for a in range(nA):
                Q[s,a]=0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_, s,a]*(rn + γ*V[sn])
                        
                if abs(Q[s,a]-Qs[a]) > 0: policy_stable=False
            π[s] = Qs*0 + ε/nA
            π[s,Q[s].argmax()] += 1-ε     # greedy step  
    
        if policy_stable: print('policy improvement stopped @ iteration %d:'%j); break
        
    if show: env.render(underhood='maxQ', Q=Q)
    
    return π, Q
```


```python
env=randwalk(figsize=[25,.5])
π, Q = Policy_iteration_stoch(env=env, show=False)
```

    100%|███████████████████████████████████████████| 1/1 [00:00<00:00, 3279.36it/s]

    policy evaluation stopped @ iteration 1:
    policy improvement stopped @ iteration 1:


    



```python
env.render(underhood='Q', Q=π)
```


    
![png](output_111_0.png)
    


As we can see the policy chooses to take right majority of the times  1-ε+ ε/nA=.95 while taking left occasionally  ε/nA.


```python
env3x3 = Grid(gridsize=[3, 3], s0=0, goals=[8], figsize=[10,1])
V0 = [.6, .5, .5, .2, .1, .2, .1, .5, 0]
env3x3.render(underhood='V', V=V0)
```


    
![png](output_113_0.png)
    



```python
Q, π = Policy_iteration_stoch(env3x3, γ=.9, ε=0, θ=1e-5, show=True)
π
```


    
![png](output_114_0.png)
    





    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 0., 0.]])



## Value Iteration Algorithm

Our final step to fully develop the ideas of dynamic programming is to shorten the time it takes for a policy to be evaluated and improved. One simple idea we will follow here is to slightly improve the evaluation and immediately improve the policy. We do these two steps iteratively until our policy has stopped to improve. This is a very effective strategy because we do not wait until the policy is fully evaluated to improve it; we weave and interleave the two loops together in one loop. Below we show this algorithm. Read section 4.4 to further your understanding of this algorithm.


```python
np.random.seed(1) # change the seed to get a different dynamics
```


```python
np.random.seed(1)

def value_iteration(env=randwalk(), p=None, V0=None, γ=.99, θ=1e-4, epochs=None, show=False): 

    # env parameters
    nS, nA, nR, rewards, i = env.nS, env.nA, env.nR, env.rewards_set(), 0
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS) if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA)) # state action values storage

    while True if epochs is None else i<epochs:
        Δ = 0
        i+= 1
        for s in range(nS):
            if s in env.goals: continue
            v, Q[s] = V[s], 0
            for a in range(nA):
                for sn in range(nS):
                    for rn_, rn in enumerate(rewards):            # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V[sn])  # max operation is embedded now in the evaluation
                        
            V[s] = Q[s].max()                                     # step which made the algorithm more concise 
            Δ = max(Δ, abs(v-V[s]))
            
        if Δ<θ: print('loop stopped @ iteration: %d , Δ = %2.f'% (i, Δ)); break
        if show: env.render(underhood='π', π=Q.argmax(1))
        
    return Q
```


```python
env = randwalk()
π = value_iteration(env, show=True).argmax(1)
print('optimal action for state', π)
```


    
![png](output_119_0.png)
    


    loop stopped @ iteration: 6 , Δ =  0
    optimal action for state [0 1 1 1 1 1 0]


As we can see our value iteration algorithm is effective in finding the optimal policy. Note that for the random walk environment the best policy is to move right to get a reward.

### Value Iteration on a Grid World

Now that you understand the value-iteration algorithm, you can apply it on a different and more complex envornment such as the grid world. As we did for the random walk, you have access to a grid world via the Grid class and its subroutines.


```python
env3x3 = Grid(gridsize=[3, 3], reward='reward_1', s0=0, goals=[8], figsize=[10,1])
V0 = [.6, .5, .5, .2, .1, .2, .1, .5, 0]
env3x3.render(underhood='V', V=V0)
```


    
![png](output_123_0.png)
    



```python
Q = value_iteration(env3x3, γ=.5, V0=V0, show=True)
print('optimal action for state', Q.argmax(1))
Q
```


    
![png](output_124_0.png)
    


    loop stopped @ iteration: 6 , Δ =  0
    optimal action for state [1 1 3 1 1 3 1 1 0]





    array([[-1.9375, -1.875 , -1.9375, -1.875 ],
           [-1.9375, -1.75  , -1.875 , -1.75  ],
           [-1.875 , -1.75  , -1.75  , -1.5   ],
           [-1.875 , -1.75  , -1.9375, -1.75  ],
           [-1.875 , -1.5   , -1.875 , -1.5   ],
           [-1.75  , -1.5   , -1.75  , -1.    ],
           [-1.75  , -1.5   , -1.875 , -1.75  ],
           [-1.75  , -1.    , -1.75  , -1.5   ],
           [ 0.    ,  0.    ,  0.    ,  0.    ]])



Let us try on a more complex grid environment.


```python
env=grid()
Q = value_iteration(env=env, show=True)
print('optimal action for state', Q.argmax(1))
# env.render(underhood='π', π=Q.argmax(1))
```


    
![png](output_126_0.png)
    


    loop stopped @ iteration: 10 , Δ =  0
    optimal action for state [1 1 1 1 1 1 3 0 0 0 1 1 1 1 1 1 3 0 0 0 1 1 1 1 1 1 3 0 0 0 1 1 1 1 1 1 0
     0 0 0 1 1 1 1 1 1 2 0 0 0 1 1 1 1 1 1 2 0 0 0 1 1 1 1 1 1 2 0 0 0 1 1 1 1
     1 1 2 0 0 0]


To interpret the policy we provided you with a useful function to render the environemt with its policy as shown above.

As we can see the policy-iteration algorithm successfuly gave us the best policy for this simple environment.

### Value Iteration on a Windy Grid World
Below we show the results of applying the value iteration method on a windy grid world. This is almost identical to the previous simple grid world without any obstacles, the only difference is that there is a wind blowing upwards, which shifts the agent 2 or 1 cell depending on its location. See page 130 of the book.


```python
env=windy()
Q = value_iteration(env, show=True)
print('optimal action for state', Q.argmax(1))
```


    
![png](output_130_0.png)
    


    loop stopped @ iteration: 12 , Δ =  0
    optimal action for state [1 1 1 1 1 1 1 3 0 0 1 1 1 1 1 1 1 2 0 2 1 1 1 1 1 1 1 2 1 2 1 1 1 1 1 1 1
     0 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 2]


### value iteration on a maze
Let us now apply the policy-iteration on the maze env.


```python
env=maze()
policy = value_iteration(env, show=True)
# print('optimal action for state', policy)
```


    
![png](output_132_0.png)
    


    loop stopped @ iteration: 14 , Δ =  0


## Conclusion
In this lesson, we covered the main dynamic programming algorithms. We saw how evaluating a policy was extremely useful in being the key component to allowing us to improve the policy. We then developed a policy iteration algorithm which improves the policy in two main steps:
1. a step that evaluates the policy fully to reach an accurate estimation of the action values of the current policy
2. a step that improves the policy by adopting a greedy action. The usage of an action-value function Q(s,a) was key in allowing us to choose between actions since the state-value function V(s) does not differentiate between the values of actions

We finally saw how the value-iteration algorithm has a similar structure to the policy-iteration algorithm with one important difference; it can arrive at an optimal policy by just taking a step *towards* the optimal policy by slightly refines its estimation of the action-value function without fully evaluating it.  Hence, it improves its policy more concisely and with much less overhead than the full policy iteration method.

In the next lesson, we will take a different approach and move to cover sampling methods that do not use the dynamics of the environment explicitly and instead try to improve its policy by interacting with the environment.


## Your turn
- Try to infer the state transition function tr(sn,s,a) and reward function r(s,a) for the randwalk_()
- Try the policy iteration on the maze8 which allows the agent to move diagonally, it is fun. You should arrive at an optimal policy for this maze environment where an agent can move diagonally and has 8 actions. You should be able to observe how efficient the policy is and how the agent is able to reach the goal location from anywhere in the environment.


## Challenges
1. Alter policy_iteration_stoch to use a softmax policy instead of the ε-greedy policy that we used in the policy_iteration_stoch method.
2. You can challenge yourself also by trying to combine policy_iteration and policy_iteration_stoch() in one method. Use a flag such as ε_greedy to distinguish between deterministic and probabilistic policies.

