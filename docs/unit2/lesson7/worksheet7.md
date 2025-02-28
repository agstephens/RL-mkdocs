# Working with ROS 2 Packages, Nodes, and Topics: A Tutorial Using Turtlesim

This tutorial introduces the basics of working with ROS 2 by guiding you through creating a simple package, developing a Python node, and using topics to control a turtle in the Turtlesim simulation. You'll learn how to publish movement commands, build and run the package, and inspect topics to enhance your understanding of ROS 2 communication and node development.

**Learning Outcomes**

1. Generate a Python-based ROS 2 package using ros2 pkg create.
1. Write a Python node that publishes velocity commands to control a turtle.
1. Build and run the package to control the turtle in Turtlesim.
1. Inspect and interact with ROS 2 topics using ros2 topic list and ros2 topic echo.
1. Modify the node to add more complex controls, such as rotation.

Turtlesim is a simple simulation package that can be used to learn ROS 2 fundamentals. We will walk through creating a ROS 2 package, writing nodes, and using topics to control a simulated turtle.

## Step 1: Create a ROS 2 Package

To create a ROS 2 package, navigate to your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
```

create a new ROS 2 package, run the following command:

```bash
ros2 pkg create --build-type ament_python my_turtle_pkg
```

Once the package is created, navigate to the newly created package:

```bash
cd my_turtle_pkg
```

## Step 2: Create a Simple Node to Control the Turtle

In the my_turtle_pkg package, create a Python script to control the turtle. Start by creating a turtle_control.py file in the my_turtle_pkg/my_turtle_pkg directory.

Create the Python file:

```bash
touch my_turtle_pkg/turtle_control.py
```

Open turtle_control.py and add the following code:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleControl(Node):
    def __init__(self):
        super().__init__('turtle_control')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.timer = self.create_timer(1.0, self.publish_velocity)

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 2.0  # move forward
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)

def main(args=None):
    rclpy.init(args=args)
    turtle_control_node = TurtleControl()

    rclpy.spin(turtle_control_node)
    turtle_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

```

TurtleControl is a node that publishes velocity commands to the /turtle1/cmd_vel topic.
It creates a publisher to send Twist messages to control the turtle.
The publish_velocity function sends commands to move the turtle forward every 1 second.

## Step 3: Build and Run the Package
Now, go back to your ROS 2 workspace and build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_turtle_pkg
```
In a new terminal, launch the Turtlesim simulation:

```bash
ros2 run turtlesim turtlesim_node
```

This will open a window with a turtle in a 2D world.

Now, run your turtle_control.py node:

```bash
ros2 run my_turtle_pkg turtle_control
```

You should see the turtle in the Turtlesim window moving forward.

## Step 4: Use ROS2 Topics to Control the Turtle
You can view the topic being used by your node by running the following command:

```bash
ros2 topic list
```
This will display all active topics. You should see /turtle1/cmd_vel listed.

To see the messages being published to the topic, run:
```bash
ros2 topic echo /turtle1/cmd_vel
```

You can modify the TurtleControl node to add more complex control, such as rotating the turtle. Modify the publish_velocity method:

```python
def publish_velocity(self):
    msg = Twist()
    msg.linear.x = 2.0  # move forward
    msg.angular.z = 1.0  # rotate
    self.publisher_.publish(msg)
    self.get_logger().info('Publishing: "%s"' % msg)

```

To stop the turtle, simply press Ctrl+C in the terminal where the node is running.

## Summary
This tutorial introduces the core concepts of working with ROS 2 by guiding users through the creation of a simple ROS 2 package and node to control a turtle in the Turtlesim simulation. 