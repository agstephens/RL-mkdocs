# Running Turtlesim in ROS 2 Foxy: A Step-by-Step Tutorial

**learning outcomes:**

By the end of this worksheet, you will be better able to:

- **explain** the purpose and function of the Robot Operating System (ROS)
- **describe** how "messages", "topics", and "message types" work in ROS
- **demonstrate** skills to use ROS messages to make two ROS processes communicate
- 
Turtlesim is a simple simulation tool in ROS 2 that lets you interact with a turtle in a 2D environment. This tutorial will guide you through running the Turtlesim node and interacting with it using the ROS 2 terminal commands.

## Step 1: Setup Your ROS 2 Workspace
Before running Turtlesim, ensure your ROS 2 Foxy environment is set up.

In a terminal, source your ROS 2 Foxy installation:

```bash
source /opt/ros/foxy/setup.bash
```
If you have a ROS 2 workspace, source it as well:

```bash
source ~/ros2_ws/install/setup.bash
```

## Step 2: Launch Turtlesim
Now you are ready to launch the turtlesim_node.

To run Turtlesim, use the following command in a terminal:

```bash
ros2 run turtlesim turtlesim_node
```

This will open a window displaying a turtle in a 2D world.

## Step 3: Interact with the Turtle
Now that Turtlesim is running, you can interact with the turtle using ROS 2 commands and topics.


You can view the active topics in your ROS 2 environment using the following command:

```bash
ros2 topic list
```

This will list all topics being used. You should see something like:

```bash
/clock
/turtle1/cmd_vel
/turtle1/pose
```

- /turtle1/cmd_vel: The topic for controlling the turtle’s velocity.
- /turtle1/pose: The topic for getting the turtle’s position and orientation.

### Move the Turtle Using Velocity Commands
You can use the cmd_vel topic to control the turtle’s movement. First, let’s send a velocity command to move the turtle forward. To send a forward velocity to the turtle, use the following command:

```bash
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0}, angular: {z: 0.0}}"
```

- linear.x = 2.0: Moves the turtle forward with a speed of 2.0.
- angular.z = 0.0: No rotation (straightforward).

You will see the turtle start moving forward in the Turtlesim window.

### Stop the Turtle
To stop the turtle, publish a message with zero velocity:

```bash
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

The turtle will stop moving.

### Rotate the Turtle
To rotate the turtle, you can publish a message that applies angular velocity:

```bash
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 1.0}}"
```

- linear.x = 0.0: No forward movement.
- angular.z = 1.0: Rotates the turtle clockwise.

You will see the turtle start rotating in the Turtlesim window.

3.3. View Turtle’s Pose
To view the turtle’s current position and orientation, you can use the pose topic:

```bash
ros2 topic echo /turtle1/pose
```

This will display real-time information about the turtle, such as its position (x, y) and orientation (theta), as well as its linear and angular velocities.

For example:

```makefile
x: 5.544
y: 5.544
theta: 0.0
linear_velocity: 2.0
angular_velocity: 0.0
```

This information updates every time the turtle moves.

## Step 4: Shutdown the Turtlesim Node
Once you are done, you can stop the Turtlesim node by pressing Ctrl+C in the terminal where the ros2 run turtlesim turtlesim_node command is running.

## Summary
In this tutorial, you learned how to:

- Launch the Turtlesim node in ROS 2 Foxy.
- Interact with the turtle by publishing velocity commands through the terminal.
- View the turtle’s pose by subscribing to the /turtle1/pose topic.
- Stop and control the turtle's movement using the ros2 topic pub command.

This is a simple demonstration of how to control a simulated robot in ROS2 using the terminal. We will provide you with a way to control a simulated robot using python code in the next tutorial. 