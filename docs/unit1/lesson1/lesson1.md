# Lesson 1: Introduction to Tabular Methods in Reinforcement Learning

In this unit, we cover the main framework of reinforcement learning, namely Markov Decision Processes(MDPs). RL has gained a lot of attention in recent years due to its unmatched ability to tackle difficult control problems with a minimal assumption about the setting and the environment that an agent works in. Controlling an agent (simulated robot) is not trivial and can often require a specific setup and strong assumptions about its environment that make the corresponding solution sometimes either difficult to attain or impractical in real scenarios. In RL, we try to minimise these assumptions and require that only the environment adheres to the Markov property. In simple terms, the Markov property assumes that inferring what to do (taking action) in a specific state can be fully specified by looking at this state and does not depend on other states.

In RL, we deal with states, actions and rewards. State space is the space the agent operates in, whether physical or virtual. The state can represent something specific in the environment, an agent configuration or both. The actions are the set of decisions available for the agent to take. An RL agent's main aim is to attain, usually via learning, a cohesive policy that allows it to achieve a specific goal. This policy $π$ can take a simple form $π(s)=a$ or symbolically $s → a$, which means if the agent is in state $s$, then take action $a$. This type of policy is deterministic because the agent will definitely take the action $a$ if it is in state $s$. Another type of policy that we deal with is stochastic policy. A stochastic policy takes the form of $π(a|s)$, which represents the probability of taking action $a$ given that the agent is in state $s$. For such a policy, the agent draws from the set of available actions according to the conditional probability, which we call its policy.

The reward function can take the simple form of $r(s,a)$ or symbolically $(s,a) →r$, which is interpreted as: if the agent is in state s and applied action $a$ it obtains a reward $r$. This reward can be actual or expected. The general setting that we deal with is the probabilistic one with the form of $p(r|s,a)$, which provides us with the probability of the agent obtaining reward r given it was in state s and applied action $a$. Another conditional probability that we deal with takes the form of $p(s’|s,a)$, which is the probability of transitioning to state $s’$ given the agent was in state s and applied action $a$. This is called the transition probability. Both the transition and reward probabilities can be inferred from a more general probability that specifies the dynamics of the environment. This joint conditional probability takes the form of $p(s’,r|s,a)$, which is interpreted as the joint probability of transitioning to state s’ and obtaining reward r given that the agent was in state s and applied action $a$. We will deal mostly with these dynamics in the second lesson of this unit. Bear in mind that obtaining the dynamics is difficult or intractable in most cases except for the simplest environment. Nevertheless, the dynamics are very useful theoretically for understanding the basic ideas of RL.

The reward function is strongly linked to the task that the agent is trying to achieve, this is the minimal information provided for the agent to indicate to it whether it is on the right track or not. We need to be careful not to devise a complicated reward function. This is usually unnecessary, and when doing so, we may be directly solving the problem for the agent. Instead, we would want the agent to solve the problem by utilising its simple reward signal and interacting with its environment to gain experience and sharpen and improve its decision-making policy. Improving its decision-making involves two things: evaluating its current policy and changing it in a way that will improve its performance. This is where the sum of rewards an agent can collect while achieving a task plays a major role. In RL, we link the policy to maximising the discounted sum of the rewards an agent can obtain while moving towards achieving a task. The reward can be negative, and in this case, the agent will be trying to minimise the sum of negative rewards that it is collecting before terminating. The termination happens when the agent achieves the required task, has taken a pre-specified number of steps or consumed pre-set computational resources. An example of a good simple reward is giving a robot a reward of 1 when it reaches a goal location and 0 otherwise. Another example is giving a robot a negative reward of -1 for each step it takes before reaching the goal location. Both of these rewards have their advantages, and we will study them in our exercises. The advantage of the first type is its simplicity and inforced sparsity, but it can take longer to explore the environment. The advantage of the second type is that it provides intermediate information that the agent can immediately utilise to improve its policy before reaching the goal location. The second type is specifically useful for online learning and when we want to alleviate the agent from having to change its starting position to cover all possible starts. The first type is useful for studying the capabilities of a learning algorithm with minimum information.

The sum of rewards from a specific state to the end of the task is called the return. Because we do not know how long the agent may take to achieve the task and to fairly maximise the rewards and allow for variability, we discount the rewards so that more recent rewards have more effect than later rewards. Nevertheless, our aim is to maximise the rewards in the long run. To be more explicit, we call the sum of discounted rewards from the current state to the end of the task the return that the agent will obtain in the future- the whole idea is to predict these rewards and be able to collect as much as possible.

The task we give the agent can be a continuous, infinite interaction with the environment. It can also take the form of a task with a specific goal or termination state, and when it is reached, the task is naturally aborted and repeated or reattempted. The former is called a continuous task, while the latter is called an episodic task. Episodic tasks have a start and end, while continuous tasks have a start but never end. The horizon (the number of steps) of episodic tasks is finite, while the horizon for continuous tasks is infinite. It turns out that discounting is necessary for theoretical guarantees for continuous tasks, which must be strictly < 1, while for episodic tasks, it can be set to 1. We will deal mostly with episodic tasks.

The function that specifies each state's return when the agent *follows a specific policy* is called the **value function** and is denoted as $v(s)$. On the other hand, we call the function that specifies the return of the current state given that the agent takes a specific action $a$ and then just follows a specific policy, the action-value function and is denoted as $q(s,a)$. These two functions provide a great utility for us in guiding our search for an optimal policy. The action-value function can be directly utilised to provide a policy that greedily chooses the action with the maximum value; when we do so, we call the algorithms that follow this pattern a value-function algorithm. Alternatively, we can maximise the policy directly without having to maximise the expected return first. These types of algorithms that do so are called policy-gradient algorithms and depend on approximation.

We will largely deal with two types of algorithms: tabular algorithms, which use a tabular representation of the value function v and the action-value function q. These will be covered in the first three units. In the subsequent units, we cover the second type of algorithms that deal with function approximation, where representing the state and actions in a table is intractable or impractical. In these algorithms, we generalise the tabular algorithms that we cover in the first three units to be able to use the models that we covered in machine learning, such as linear regression models or neural networks


## Tabular Representation and Example
In the first three units, you will learn about the main ideas of reinforcement learning that use lookup tables. These tables identify a certain action that will be taken in a certain state and are called a policy. Our main concern is to design algorithms that *learn* a suitable policy.

**A Policy Lookup Table For Commuting to Work**

| State      | Action
|---         |---
| have energy and have time | walk
| have energy and no   time | cycle
| no  energy  and have time | take a bus
| no  energy  and no   time | take a taxi

More formally, in the first three units, we will study important reinforcement learning algorithms that use *tabular policy representation*. These algorithms learn a lookup table that identifies a suitable action that can be taken for a certain state. Our main concern is to design practical and efficient algorithms that can *learn* an optimal policy either via direct interaction with an environment, via an environment model that captures the dynamics of the environment, or both. An optimal policy is a policy that maximises the sum of discounted rewards obtained by following this policy.

The main underlying framework we assume is a Markov Decision Process (MDP). In a nutshell, this framework assumes that the probability of moving to the next state is only dependent on the current state and not on past states.

We start by covering non-associated problems, such as K-armed Bandit. These problems have no states. This will help us focus on the action space as it will be isolated from the state's effect. We then study how to solve MDP problems using Dynamic Programming (DP). DP assumes that we have a model of the environment that the agent is acting on. By model, we mean the dynamics or probabilities of landing in a state and obtaining a specific reward given an action and a previous state. This kind of conditional probability provides a comprehensive framework to reach an optimal policy, but it is hard to obtain in the real world. We then reside in sample methods, particularly Monte Carlo. This method allows us to gather samples of an agent running, making environmental decisions, and collecting rewards. We use averages to obtain an estimate of the discounted returns of an episode. We conclude our units by developing suitable core classes that allow us to study and demonstrate the advantages and disadvantages of these and other methods.

A policy can be deterministic or stochastic, below we show both for an associative problem. A deterministic policy is a special case of a stochastic policy with all of its actions' probabilities being 0 except one action.

<!-- <table><tr><th>Deterministic Associative Policy</th><th> Stochastic Associative Policy</th><th> Non-Associative Policy</th></tr>
<tr><td> -->

| State | Action
|---    | ---
| $S_1$ | $A_1$ 
| $S_2$ | $A_2$ 
| $S_3$ | $A_2$ 

<!-- </td><td> -->

| State  | Action | Action's Probability
|---     |---     |---
| $S_1$  | $A_1$  | .8
| $S_1$  | $A_2$  | .2
| $S_2$  | $A_1$  | .6 
| $S_2$  | $A_2$  | .4 
| $S_3$  | $A_1$  | 0. 
| $S_3$  | $A_2$  | 1.

<!-- </td><td> -->

| Action | Action's Probability
|---     |---
| $A_1$  | .7
| $A_2$  | .2
| $A_3$  | .1 
<!-- </td></tr></table> -->
