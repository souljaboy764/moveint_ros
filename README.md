# moveint_ros

ROS Package to run the [MoVEInt](https://github.com/souljaboy764/moveint) models on the Pepper Robot and the Kobo Robot for Robot-to-Human Handovers.

## Dependencies

### Python Libraries

The following python libraries need to be installed with pip

- torch
- matplotlib
- numpy
- ikpy ([modified](https://github.com/souljaboy764/ikpy))

Additionally clone and install the repository [`phd_utils`](https://github.com/souljaboy764/phd_utils) if not already done.

For the skeleton tracking, Nuitrack needs to be installed. Follow the [instructions](https://github.com/3DiVi/nuitrack-sdk/blob/master/doc/Install.md) to install nuitrack. Once Nuitrack is installed and the license is activated, download the suitable [python wheel file for Nuitrack](https://github.com/3DiVi/nuitrack-sdk/tree/master/PythonNuitrack-beta/pip_packages/dist) and install it with `pip install /path/to/wheel.whl`

### ROS Packages

The following packages need to be installed to your catkin workspace:

- [naoqi_dcm_driver](https://github.com/souljaboy764/naoqi_dcm_driver) (along with the rest of the Pepper robot ROS stack)
- [tf_dynreconf](https://github.com/souljaboy764/tf_dynreconf)
- [pepper_controller_server](https://github.com/souljaboy764/pepper_controller_server)

## Installation

Once the prerequisites are installed, clone this repository to your catkin workspace and build it.

```bash
cd /path/to/catkin_ws/src
git clone https://github.com/souljaboy764/moveint_ros
cd ..
catkin_make
```

The pretrained models are available with this repository in the `models_final` folder.

## Handovers with Pepper

### Setup

1. Before running any ROS nodes, make sure that the library path is set for Nuitrack.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/lib/nuitrack
    source /path/to/current_ws/devel/setup.bash
    ```

2. Run `roslaunch moveint_ros prepare_pepper.launch` after setting the IP of the Pepper robot and the network interface accordingly to get the setup ready. This launches the robot nodes, the transformation between the camera and the robot, collision avoidance etc.

3. For the external calibration, after starting up the robot with [`naoqi_dcm_driver`](https://github.com/souljaboy764/naoqi_dcm_driver), and `nuitrack_node.py` to start nuitrack, run `rosrun rqt_reconfigure rqt_reconfigure gui:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/nuitrack_pepper_tf.yaml`](config/nuitrack_pepper_tf.yaml).

### Experimental run

Run steps 1 and 2 from above to setup the experiment to start up the robot.

First, reset the robot by running

`rosrun moveint_ros reset_pepper.py 0`

The 0 is to close the hand so that pepper holds the object in question. if the hand needs to be opened, use a suitable positive value less than 1 as 1 is fully open.

For running the controller:

`rosrun moveint_ros rmdvae_pepper_node.py`

This starts the controller node as well as Nuitrack. For the fist 1-2 seconds stand still as the controller calibrates the human's neutral position, which is used for terminating the interaction when the human returns their hand to this neutral position.

## Handovers with Kobo

### Setup

1. Ensure that the computer is connected on the same network as the Kobo control PC that is running the Kobo control ROS stack.
2. Add the following to `~/.bashrc` if not already done:

    ```bash
    export ROS_MASTER_URI=http://10.10.0.8:11311
    export ROS_IP=10.10.0.182
    ```

3. Run `roslaunch moveint_ros prepare_kobo.launch`.

4. External calibration
    4a. For external calibration with Nuitrack, run `nuitrack_node.py` to start nuitrack and then run `rosrun rqt_reconfigure rqt_reconfigure gui:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/kobo-nuitrack-calib.yaml`](config/kobo-nuitrack-calib.yaml).
    4b. For external calibration with Kinect, run `roslaunch azure_kinect_ros_driver driver_with_bodytracking.launch` to start the Kinect node and then run `rosrun rqt_reconfigure rqt_reconfigure gui:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/kobo-kinect-calib.yaml`](config/kobo-kinect-calib.yaml).

### Experimental run

Run steps 1, 2 and 3 from above to setup the experiment to start up the robot.

First, reset the robot by running

`rosrun moveint_ros reset_kobo.py`

This will place the Kobo in a suitable location. Then place the object between its hands

For running the controller, start the ROS node corresponding to the perception system that is being used:

`rosrun moveint_ros rmdvae_kobo_node.py`

This starts the controller node as well as Nuitrack. For the fist 1-2 seconds stand still as the controller calibrates the human's neutral position, which is used for terminating the interaction when the human returns their hand to this neutral position.
