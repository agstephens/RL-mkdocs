





# Basic ROS Concepts

![01-04.png](01-04.png)

[picture credit](https://ktiwari9.gitlab.io/ros101/JargonSection.html)


## Table of Contents
- [Basic ROS Concepts](#basic-ros-concepts)
  - [Table of Contents](#table-of-contents)
  - [Basic ROS Terms](#basic-ros-terms)
    - [ROS Node](#ros-node)
    - [Discovery](#discovery)
    - [ROS Package](#ros-package)
    - [ROS Stack](#ros-stack)
    - [ROS Workspace](#ros-workspace)
    - [ROS Topic](#ros-topic)
    - [ROS Publisher](#ros-publisher)
    - [ROS Subscriber](#ros-subscriber)
    - [ROS Service](#ros-service)
    - [URDF File](#urdf-file)
    - [ROS Launch file](#ros-launch-file)
    - [ROS Parameter](#ros-parameter)
    - [ROS Messages](#ros-messages)
    - [Robot Web Tools](#robot-web-tools)
    - [ROS Serial](#ros-serial)
- [VM and Prerequisite Packages](#vm-and-prerequisite-packages)

 ## Basic ROS Terms

This Wiki page will help you understand the basic definitions of the most common ROS terms. ROS 2 is a middleware based on a strongly-typed, anonymous publish/subscribe mechanism that allows for message passing between different processes.

At the heart of any ROS 2 system is the ROS graph. The ROS graph refers to the network of nodes in a ROS system and the connections between them by which they communicate.

These are the concepts that will help you get started understanding the basics of ROS 2.

**These concepts are same for both ROS 1 and ROS 2**

### ROS Node

A [ROS node](https://docs.ros.org/en/iron/Concepts/Basic/About-Nodes.html) is a simple program that publishes or subscribes to a topic or contains a program that enables a ROS service. It should be highly specialized and accomplish a single task. Nodes communicate with other nodes by sending messages, which will be discussed later. Examples of tasks which should be carried out in a specific node are:

-  Controlling motors
-  Interpreting commands from an input source
-  Planning a path to drive on
-  Reading a specific sensor.

A node is a participant in the ROS 2 graph, which uses a client library to communicate with other nodes. Nodes can communicate with other nodes within the same process, in a different process, or on a different machine. Nodes are typically the unit of computation in a ROS graph; each node should do one logical thing.

Nodes can publish to named topics to deliver data to other nodes, or subscribe to named topics to get data from other nodes. They can also act as a service client to have another node perform a computation on their behalf, or as a service server to provide functionality to other nodes. For long-running computations, a node can act as an action client to have another node perform it on their behalf, or as an action server to provide functionality to other nodes. Nodes can provide configurable parameters to change behavior during run-time.

Nodes are often a complex combination of publishers, subscribers, service servers, service clients, action servers, and action clients, all at the same time.

Connections between nodes are established through a distributed discovery process


### Discovery

Discovery of nodes happens automatically through the underlying middleware of ROS 2. It can be summarized as follows:

When a node is started, it advertises its presence to other nodes on the network with the same ROS domain (set with the ROS_DOMAIN_ID environment variable). Nodes respond to this advertisement with information about themselves so that the appropriate connections can be made and the nodes can communicate.

Nodes periodically advertise their presence so that connections can be made with new-found entities, even after the initial discovery period.

Nodes advertise to other nodes when they go offline.

Nodes will only establish connections with other nodes if they have compatible Quality of Service settings.

Take the [talker-listener demo](https://docs.ros.org/en/iron/Installation/Alternatives/Ubuntu-Development-Setup.html#talker-listener) for example. Running the C++ talker node in one terminal will publish messages on a topic, and the Python listener node running in another terminal will subscribe to messages on the same topic.

You should see that these nodes discover each other automatically, and begin to exchange messages.

### ROS Package

A package is a container for closely related nodes and utilities used by those nodes. Include files, message definitions, resources, etc are stored along with nodes inside a package. As an example, imagine a robot with many different sensors, such as IR rangefinders, sonar, laser scanners, and encoders. Each of these types of sensors would ideally have their own node. Because these nodes are all for sensors, you might group them into a package called _my_robots_sensor_drivers_.

### ROS Stack

A stack is a collection of closely related packages. You might name a stack after your robot, with individual packages for sensors, motors, and planning. These packages would contain more specific nodes that do specific tasks. You will often hear about the “Navigation stack” in ROS. This is a collection of packages used for helping robots navigate in the world and is very useful.

_Important note:_ ROS is trying to get rid of the term Stack for Metapackage, but everyone still uses Stack

### ROS Workspace

All of the development you do for ROS must be done in your ROS workspace. This is so ROS knows where to look for all of the programs you write and their respective utilities and resources.

### ROS Topic

[ROS Topics](http://wiki.ros.org/Topics) are named buses that allow nodes to pass messages. A ROS topic can be published to, or subscribed to with many different message types. This setup allows the publishing and subscribing to happen completely independent of each other assuming they have matching message types. For a more detailed information please follow the [link](https://docs.ros.org/en/iron/Concepts/Basic/About-Interfaces.html)

### ROS Publisher

A ROS Publisher is a program or portion of a program that "publishes" or sends data into a ROS topic.

### ROS Subscriber

A ROS subscriber is a program or portion of a program that "subscribes to" a topic in order to receive the messages published to that topic.

### ROS Service

A [ROS service](http://wiki.ros.org/Services) is an object that can be called similar to a function from other nodes. It allows a request and a reply. An example might be an ["add two ints"](http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29) service.

### URDF File
[ROS URDF](http://wiki.ros.org/urdf) files are used to define a robots physical characteristics. The URDF file describes the robot model, Robot sensors, scenery, and objects. These models are displayed in rviz and use the tf (transform) topics to define where the models are. If you are creating a robot you can/will be able to export the URDF file straight from Solidworks.

### ROS Launch file
A [ROS launch](http://wiki.ros.org/roslaunch) file is a file used to execute multiple nodes and allows for remapping of topics, setting of parameters, and will automatically start roscore if it is not already running.
See [roslaunch](http://wiki.ros.org/roslaunch) or [ROS Launch file type](http://wiki.wpi.edu/robotics/ROS_File_Types).

### ROS Parameter
A ROS parameter is a name that has an associated value. A central [[parameter server](http://wiki.ros.org/Parameter%20Server)] keeps track of parameters in one place that can be updated. . As it is not designed for high-performance, it is best used for static, non-binary data such as configuration parameters.

### ROS Messages
[ROS Messages](http://wiki.ros.org/Messages) are the individual sets of data that are published to a topic. Messages can have many [different types](http://wiki.ros.org/common_msgs) but the standard messages can be found [here](http://wiki.ros.org/std_msgs). The fields in messages can be any of the [base types](http://wiki.ros.org/msg) or another message. Common Message types can be found [here](http://wiki.ros.org/common_msgs).

### Robot Web Tools
Robot Web Tools is a suite that allows most ROS information to be sent over the internet. [Robot Web Tools](http://robotwebtools.org/) allows you to build ROS webpages or pass the information directly through the internet.

### ROS Serial
[ROS Serial](http://wiki.ros.org/rosserial) allows ROS to talk with any serial device, primarily embedded controllers such as Arduino's.

# VM and Prerequisite Packages 

<!-- Video Introduction to [Azure VM](https://leeds365-my.sharepoint.com/:v:/g/personal/scsaalt_leeds_ac_uk/EWEwLfWr5w9Mnl0E29LjvJUBlsG-nWbee6RzKHKYSp_D-Q?e=V79IXI) -->

Please see the following video to get started with teh VM. 

<iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=f52d3061-e7ab-4c0f-9e5d-04dbd2e3bc95&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="1- Introduction to Azure VM.mkv"></iframe>


You should recieve an email inviting you to have access to an Azure VM. The VM has [ROS 2 Foxy Fitzroy](https://docs.ros.org/en/foxy/Installation.html) already installed. ROS2 commands need to be run from the terminal not from a conda-activated terminal (due to compatibility), and they use the default system Python 3.8. The VM has the libraries required for ROS2 along with TurtleBot3 installed with the worlds required for assessment.

We have tested the notebooks on Python 3.8, so they should work smoothly for higher versions.
Note that ROS2 code must be run with the default VM Python3.8 kernel. For the best experience, use VScode

The machine has decent cores and memory (according to Azure 4 cores | 8GB RAM | 128GB Standard SSD). The VM has Ubuntu 20 and Xfce (Xubuntu) interface due to its lightweight (to give you the best experience remotely- to come as close as a local machine feeling) and it is tailored to give the same feeling as the usual Ubuntu Genome. You can run hardinfo in the terminal to check the VM specs. I hope you will enjoy it. 

To access the VM, please use the usual remote desktop app available on your system. You will receive an email with access to your VM. The username is rl, and the password is rl@ros2. 


You will have sudo access. Please apply caution when dealing with the system and avoid installing packages so as not to break it, which can be time-consuming. You will have around a 40-hour time limit, so please be mindful not to leave the system running unless necessary so as not to run out of time. Usually, you would want time for running the exercises and save plenty of time (1/2) for your project training (this is where the VM will be most useful).

If the VM becomes corrupted for some reason, then you can reimage it by going to Azure Lab page and selecting the three dots, then reimage. That *will cause all the data you have on the machine to be lost*. You are advised to back up your data, you may want to use OneDrive or other backup methods.



Now go ahead and try doing worksheet4 to experiement more with some of the ros concepts.

[ROS Worksheet 4: Robotics Operating System](../../workseets/worksheet4.ipynb)



