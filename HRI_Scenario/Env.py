import pybullet as p
import pybullet_data
import numpy as np
import itertools
from random import sample
import os
from dataclasses import dataclass
from pybullet_funs import SphereMarker
from pybullet_utils import bullet_client
import time
import fastdtw

#prevent pybullet client from printing out


"""This module interfaces with Pybullet simulation environment and provides functions to effectuate changes in the simulation."""

@dataclass
class Goal:
    """Class that represents the goal area of the task and the colors of the blocks to be placed in the goal area."""
    x: float
    y: float
    color_order: list
    z: float = 0.63
    
    def color2text(self, color_order: list) -> str:
        """Converts the color order to text.
        :param color_order (list): The color order to convert
        :return: (str): The color order in text"""
        c2t_dict = {1: "red", 2: "green", 3: "blue", 4: "yellow"}
        color_list = []
        for i in color_order:
            color_list.append(c2t_dict[i])
        return " ".join(color_list)
    
    def resolve_goal_position(self, object_id: int) -> list:
        """Finds the goal position of the object from the goal and the color order
        :param object_id (int): The id of the object to find the goal position of
        :return: goal (list): The goal position of the object"""
        #1 is red, 2 is green, 3 is blue, 4 is yellow
        
        colors = self.color_order
        #find where the object is in the color order (object_id +1 matches the color order list)
        color_index = colors.index(object_id + 1)
        #find the goal position based on the color index
        goal = [self.x, self.y, self.z + 0.025 * (color_index + 1)]

        return goal

class Env:
    def __init__(self, limits, headless=False) -> None:
        """Environment constructor that initializes the simulation, 2 robots,
        assigns obstacles and constraints.
        :param headless (bool): If true, the simulation will run without a GUI
        :param limits (list): The limits of the robot
        :return: None"""

        self.main_sim = bullet_client.BulletClient(connection_mode=p.GUI) if not headless else bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.currentPath = os.path.dirname(os.path.realpath(__file__))
        self.main_sim.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # hide the debug parameters sidebar
        self.main_sim.setRealTimeSimulation(0)
        self.ll = limits[:,0].tolist()
        self.ul = limits[:,1].tolist()
        self.rest_poses = [0., 0., 1.4, -0.3, 0.17, 0.2, np.pi/6, 0., 0.]
        # Reset the simulation to the correct initial + goal state
        self.reset(self.main_sim, input_goal=Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), [1, 2, 3, 4]))
        self._setup_jac_dummy()
        self._setup_controller()
        

    def _set_objects(self) -> list:
        """Spawns the blocks that will be moved in the task, by computing a grid within bounds of the table, and randomly placing the blocks in the grid.
        :param None
        :return: boxId (list): list of the ids of the blocks in the scene"""

        # Get table boundaries
        boundaries = [[.3, .5], [0., 0.2]] # x lower, x upper, y lower, y upper, z
        # Compute grid
        prd = itertools.product(np.linspace(boundaries[0][0], boundaries[0][1], 10), np.linspace(boundaries[1][0], boundaries[1][1], 10), [0.62])
        grid = [i for i in prd]
        # sample 4 random positions
        boxPos = sample(grid, 4)
        self.main_sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set up the boxes
        self.boxId = []
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]] # red, green, blue, yellow
        for i in range(4):
            self.boxId.append(self.main_sim.loadURDF("cube_small.urdf", boxPos[i], [0, 0, 0, 1], globalScaling=1))
            self.main_sim.changeVisualShape(self.boxId[i], -1, rgbaColor=colors[i])

        return self.boxId
    
    def _get_object_position(self, object_id: int) -> list:
        """Gets the position of the object.
        :param object (int): The object to get the position of
        :return: _position (list): The position of the object"""

        return self.main_sim.getBasePositionAndOrientation(object_id)[0]
    
    def reset(self, sim_id, input_goal) -> None:
        """Resets the simulation to the initial state, with new objects and building area.
        :param new_constraints (bool): If true, the constraints for robot 1 will be reset
        :param sim_id (bool): to determine whether this is applied to the simulation or the collision checker
        :return: None"""
        self.input_goal = input_goal
        sim_id.resetSimulation()
        sim_id.setAdditionalSearchPath(pybullet_data.getDataPath())
        sim_id.setGravity(0, 0, -9.81) 
        self.plane = sim_id.loadURDF("plane.urdf", useFixedBase=True)
        self.tableId = sim_id.loadURDF("table/table.urdf", [0.5, 0, 0])
        self.helper = sim_id.loadURDF("sphere_small.urdf", [0.5, -0.4, 0.8], [0, 0, 1, 0], useFixedBase=True)
        sim_id.setAdditionalSearchPath(self.currentPath)
        #load the humanoid and fix its feet to the ground
        self.human_position = [0.5, 0.55, 0.77]
        self.human_orientation = p.getQuaternionFromEuler([0.,0.,-np.pi/2])
        self.human = sim_id.loadURDF("humanSubject06_48dof_customLimits.urdf", self.human_position, self.human_orientation, useFixedBase=True)
        # Set up the camera
        sim_id.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.5, 0, 0.2])
        # Draw the goal area on the table
        goal = [input_goal.x, input_goal.y, input_goal.z]
        colors = input_goal.color_order
        colors_text = input_goal.color2text(colors)
        #draw an area a bit bigger than the block around the goal
        sim_id.addUserDebugLine([goal[0] + 0.07, goal[1] + 0.07, goal[2]], [goal[0] + 0.07, goal[1] - 0.07, goal[2]], [0, 1, 0], lineWidth=5)
        sim_id.addUserDebugLine([goal[0] + 0.07, goal[1] - 0.07, goal[2]], [goal[0] - 0.07, goal[1] - 0.07, goal[2]], [0, 1, 0], lineWidth=5)
        sim_id.addUserDebugLine([goal[0] - 0.07, goal[1] - 0.07, goal[2]], [goal[0] - 0.07, goal[1] + 0.07, goal[2]], [0, 1, 0], lineWidth=5)
        sim_id.addUserDebugLine([goal[0] - 0.07, goal[1] + 0.07, goal[2]], [goal[0] + 0.07, goal[1] + 0.07, goal[2]], [0, 1, 0], lineWidth=5)

        #add text saying what the goal is
        sim_id.addUserDebugText('Goal: ' + colors_text, [0.2, -1.5, 1.1], textSize=2, textColorRGB=[1., 0., 0.])

        self.box_id = self._set_objects()

        #get number of non-fixed joints
        self.nonfixed_indices = [p.getJointInfo(self.human, i)[0] for i in range(p.getNumJoints(self.human)) if p.getJointInfo(self.human, i)[2] != p.JOINT_FIXED]
        self.ee_link_index = self.nonfixed_indices[-1] # Apparently pybullet has link_no == joint_no
        self.ee_joint_index = self.nonfixed_indices[-1]

        for i, jointIndex in enumerate(self.nonfixed_indices):
            sim_id.resetJointState(self.human, jointIndex, self.rest_poses[i])

        sim_id.stepSimulation()

    def set_to_state(self, human: list, boxes: list, helper: tuple) -> None:
        """Sets the simulation to a certain state.

        Args:
            human (list or None): The joint state of the human. If None, the human is not moved.
            boxes (list(tuple(list)) or None): Per box: a list of len 3 for x,y,z position, and a list of len 4 for quaternion orientation, in a tuple per box. If None, the boxes are not moved.
            helper (tuple(list) or None): The position and orientation of the helper sphere. If None, the helper sphere is not moved.
        """
        if human is not None:
            for i, jointIndex in enumerate(self.nonfixed_indices):
                self.main_sim.resetJointState(self.human, jointIndex, human[i])
        if boxes is not None:
            for i, box in enumerate(boxes):   
                self.main_sim.resetBasePositionAndOrientation(self.box_id[i], box[0], box[1])
        if helper is not None:
            self.main_sim.resetBasePositionAndOrientation(self.helper, helper[0], helper[1])
        
        self.main_sim.stepSimulation()

    def get_state(self) -> tuple:
        """Gets the current state of the simulation.

        Returns:
            tuple: human joint states, box positions, helper sphere position, goal position
        """
        #get the joint positions of the person
        joint_states = [self.main_sim.getJointState(self.human, i)[0] for i in self.nonfixed_indices]
        #get the position of the boxes
        box_states = [self.main_sim.getBasePositionAndOrientation(i) for i in self.box_id]
        #get the position of the helper sphere
        helper_state = self.main_sim.getBasePositionAndOrientation(self.helper)
        #get the position of the goal
        goal_state = self.input_goal

        return (joint_states, box_states, helper_state, goal_state)
    
    def forward_kinematics(self, q: list) -> list:
        """Computes the forward kinematics of the robot.
        :param q (list): The joint state of the robot
        :return: fk (list): 3D position of the end effector"""
        for i, jointIndex in enumerate(self.nonfixed_indices):
            self.dummy.resetJointState(self.dummy_human, jointIndex, q[i])
        fk = self.dummy.getLinkState(self.dummy_human, self.ee_link_index, computeForwardKinematics=1)[0]
        return fk
    
    def ee_velocity(self, q: list) -> list:
        """Computes the EE velocity.
        :param q (list): The joint state of the robot
        :return: vel (list): 3D velocity of the end effector"""
        for i, jointIndex in enumerate(self.nonfixed_indices):
            self.dummy.resetJointState(self.dummy_human, jointIndex, q[i])
        fk = self.dummy.getLinkState(self.dummy_human, self.ee_link_index, computeForwardKinematics=1, computeLinkVelocity=1)[6]
        return fk
    
    def _setup_controller(self):
        """Helper function to set up the controller parameters"""
        self.P = np.eye(3) * 10.
        self.D = np.eye(3) * 0.5
        self.KN = 10
        self.dt = 0.01
        self.lim_buffer = 0.1
        self._setup_jac_dummy()

    def _setup_jac_dummy(self):
        """Helper function to set up the dummy simulation for the jacobian calculations"""
        self.dummy = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.dummy.setGravity(0, 0, -9.81)
        self.dummy.setAdditionalSearchPath(self.currentPath)
        self.dummy_human = self.dummy.loadURDF("humanSubject06_48dof_customLimits.urdf", self.human_position, self.human_orientation, useFixedBase=True)
        for i, jointIndex in enumerate(self.nonfixed_indices):
            self.dummy.resetJointState(self.dummy_human, jointIndex, self.rest_poses[i])
        
    def _calculate_jacobian(self, q):

        for i, jointIndex in enumerate(self.nonfixed_indices):
            self.dummy.resetJointState(self.dummy_human, jointIndex, q[i])

        joint_states = self.dummy.getJointStates(self.dummy_human, range(self.dummy.getNumJoints(self.dummy_human)))
        joint_infos = [self.dummy.getJointInfo(self.dummy_human, i) for i in range(self.dummy.getNumJoints(self.dummy_human))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        mpos = [state[0] for state in joint_states]

        result = self.dummy.getLinkState(self.dummy_human,
                                self.nonfixed_indices[-1],
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

        x_rot = np.array(p.getEulerFromQuaternion(link_rot)).reshape(3, 1)

        zero_vec = [0.0] * len(mpos)
        jac_t, jac_r = self.dummy.calculateJacobian(self.dummy_human, self.ee_link_index, com_trn, mpos, zero_vec, zero_vec)
        
        return jac_t, jac_r, x_rot

    def _idk_controller(self, state: list, goal: list, pos: list, vel: list) -> list:
        """Inverse Differential Kinematics Controller

        Args:
            state (list): Joint state of the robot.
            goal (list): Cartesian Goal for the robot.
            pos (list): Current cartesian position of the robot.
            vel (list): Current cartesian velocity of the robot.
            goal_vel (list): Goal cartesian velocity of the robot.
        
        Returns:
            new_state (list): New joint state of the robot.
        """
        # joint space
        q = np.array(state).reshape(len(state), 1)
        q_l = np.array(self.ll).reshape(len(state), 1)
        q_u = np.array(self.ul).reshape(len(state), 1)
        q_i = np.array(self.rest_poses).reshape(len(state), 1)

        # task space
        x = np.array(pos).reshape(len(pos), 1)
        x_dot = np.array(vel).reshape(len(vel), 1)
        x_g = np.array(goal).reshape(len(goal), 1)
        x_dot_g = np.zeros_like(x_dot)

        jac_t, jac_r, x_rot = self._calculate_jacobian(state)
        # jacobian, pinv, nullspace
        jac = np.array(jac_t)
        J_pinv = self._get_weighted_pseudo_inverse(jac, [1., 1., 1., 1., 1., 1., 1., 1., 1.])
        null = np.eye(len(state)) - J_pinv @ jac
        # tracking errors
        e_x = x_g - x   # position error
        e_x_dot = x_dot_g - x_dot

        # joint limit penalties (projected into nullspace)
        penalty_lower_N = self.KN * np.clip(self.lim_buffer + q_l - q, 0, 0.01)   # if q > q_l, no penalty
        penalty_upper_N = self.KN * np.clip(q - q_u - self.lim_buffer, 0, 0.01)   # if q < q_u, no penalty

        # joints close to rest position
        penalty_rp =  q - q_i

        # commands
        q_new = q
        q_new += self.dt * (J_pinv @ self.P @ e_x + J_pinv @ self.D @ e_x_dot)    # task space errors
        q_new += self.dt * (null @ penalty_lower_N + null @ penalty_upper_N)      # nullspace joint limits
        q_new += self.dt * null @ penalty_rp # Rest pose penalty
        new_state = q_new.T.tolist()[0]

        # clip for joint limits, as done in ROS
        for i in range(len(new_state)):
            if new_state[i] < self.ll[i]:
                new_state[i] = self.ll[i]
            
            elif new_state[i] > self.ul[i]:
                new_state[i] = self.ul[i]
        
        return new_state, e_x
    
    def _get_weighted_pseudo_inverse(self, J, weights):
        # Ensure J and weights are NumPy arrays
        J = np.array(J)
        weights = np.array(weights)

        # Calculate the weighted pseudo-inverse
        J_W = np.dot(J, np.diag(weights))
        J_W_J_T = np.dot(J_W, J.T)
        inv_J_W_J_T = np.linalg.inv(J_W_J_T)
        pseudo_inverse = np.dot(J.T, inv_J_W_J_T)

        return pseudo_inverse

    def simulate_p2p(self, start: list, goal: list):
        """Simulates a point to point trajectory from start to goal.
        :param start (list): The start position of the trajectory
        :param goal (list): The goal position of the trajectory
        :return: cartesian_traj (list): The cartesian trajectory
        :return: joint_traj (list): The joint trajectory
        :return: err (list): The error of the trajectory"""

        state = start
        for i, jointIndex in enumerate(self.nonfixed_indices):
            p.resetJointState(self.human, jointIndex, state[i])
        x = self.forward_kinematics(state)
        vel = self.main_sim.getLinkState(self.human, self.ee_link_index, computeLinkVelocity=1)[6]
        cartesian_traj = [x]
        joint_traj = [state]
        iters = 100
        i = 0
        err = []
        while np.linalg.norm(np.array(x) - np.array(goal)) > 0.05 and i < iters:
            new_state, e_x= self._idk_controller(state, goal, x, vel)
            joint_traj.append(new_state)
            new_x = self.forward_kinematics(new_state)
            vel = np.subtract(new_x, x) / self.dt
            cartesian_traj.append(new_x)
            state = new_state
            x = new_x
            err.append(e_x)
            i += 1

        if i == iters:
            return None, None, None
        
        #goal_marker = SphereMarker(goal, radius=0.05, rgba_color=(0, 1, 0, 0.8), p_id=self.main_sim._client)
        for k in range(len(joint_traj)):
            #pseudo_point = np.array(p.getLinkState(self.human, 22)[0]) + np.array((0.01, 0.01, 0.01))
            #p.addUserDebugLine(p.getLinkState(self.human, 22)[0], pseudo_point, [1,0,0], 10, 0)
            for i, jointIndex in enumerate(self.nonfixed_indices):
                p.resetJointState(self.human, jointIndex, joint_traj[k][i])
            p.stepSimulation()
            time.sleep(1./240.)
        
        return cartesian_traj, joint_traj, err
    
    def generate_assignment_and_task_trajectories(self, goal: Goal) -> tuple:
        """Generates a set of human trajectories for the task.
        :param goal (Goal): The goal of the task
        :return (Tuple): assignment (list): The assignment of the objects to the robot or human, trajs (list): The trajectories of the human"""
        assignment = []
        trajs = []
        j_trajs = []
        #get the start and goal position for each object
        start_positions = [self._get_object_position(i) for i in goal.color_order]
        goal_positions = [goal.resolve_goal_position(i) for i in range(len(self.box_id))]
        #generate a trajectory for each object
        for s, g in zip(start_positions, goal_positions):
            traj_s, jt_s, _ = self.simulate_p2p(self.rest_poses, s)
            if traj_s is None:
                assignment.append("Robot")
                trajs.append(None)
                j_trajs.append(None)
            else:
                traj_g, jt_g, _ = self.simulate_p2p(jt_s[-1], g)
                if traj_g is None:
                    assignment.append("Robot")
                    trajs.append(None)
                    j_trajs.append(None)
                else:
                    assignment.append("Human")
                    trajs.append(traj_g)
                    j_trajs.append(jt_g)

        return assignment, trajs, j_trajs
    
# UNIT TESTS

def environment_test(limits):
    limits_np = np.array(limits).reshape(9,2)
    #test the setup of the environment and the reset, set state and get state functions
    sim = Env(limits_np, Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), [1, 2, 3, 4]))
    #create a random goal and test the reset function
    goal = Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), [1, 2, 3, 4])
    sim.reset(sim.main_sim, goal)
    #test the get state function
    joint_states, box_states, helper_state, goal_state = sim.get_state()
    print("Current state:", joint_states, box_states, helper_state, goal_state)
    #test the set state function
    desired_joint_states = [np.random.uniform(limits_np[i,0], limits_np[i,1]) for i in range(9)]
    desired_box_states = []
    for i in range(4):
        #append a tuple of list length 3 for position and list length 4 for orientation
        desired_box_states.append(([np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), 0.63], [0, 0, 0, 1]))
    desired_helper_state = None
    sim.set_to_state(desired_joint_states, desired_box_states, desired_helper_state)
    #test the get state function
    joint_states, box_states, helper_state, goal_state = sim.get_state()
    print("Desired state:", desired_joint_states, desired_box_states, desired_helper_state, goal_state)
    print("Current state:", joint_states, box_states, helper_state, goal_state)



def controller_test(limits):
        from itertools import product
        sim = Env(np.array(limits).reshape(9,2), Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), [1, 2, 3, 4]))
        #l = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "test_env.mp4")
        # sample 10 random joint configurations within the joint limits
        x_g = np.linspace(0.3, .6, 5).tolist()
        y_g = np.linspace(-0.2, 0.2, 5).tolist()
        z_g = [0.75]
        goals = list(product(x_g, y_g, z_g))
        completed = 0
        for i in range(len(goals)):
            goal = goals[i]
            print("Goal:", goal)
            traj, traj_states, err = sim.simulate_p2p(sim.rest_poses, goal)
            
            if traj is None:
                print("Controller Failed")
                continue
            else:
                print("Controller Succeeded")
                print("Final:", traj[-1])
                completed += 1
        print("Total goals: ", len(goals))
        print("Completed: ", completed) 

        #p.stopStateLogging(l)
        p.disconnect()
        del sim

def assignment_test(limits):
    #create 10 random goals
    goals = []
    for i in range(10):
        goals.append(Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), [1, 2, 3, 4]))
    sim = Env(np.array(limits).reshape(9,2), headless=True)
    for goal in goals:
        sim.reset(sim.main_sim, input_goal=goal)
        assignment, trajs, jtrajs = sim.generate_assignment_and_task_trajectories(goal)
        print(assignment)
        print(len(assignment))
        print(len(trajs))
        [print(type(traj)) for traj in trajs]
    
    p.disconnect()
    del sim

def trajectory_test(limits1, limits2, limits3):
    """Tests 3 different trajectories with the same goal and then plots the cartesian trajectories.

    Args:
        limits1 (np.array): The joint limits of the first robot
        limits2 (np.array): The joint limits of the second robot
        limits3 (np.array): The joint limits of the third robot
    """

    from matplotlib import pyplot as plt

    goal = [0.3, -0.1, .65]

    sim1 = Env(np.array(limits1).reshape(9,2), headless=True)
    print(sim1.ll)
    print(sim1.ul)
    traj1, traj_states1, err1 = sim1.simulate_p2p(sim1.rest_poses, goal)
    del sim1

    sim2 = Env(np.array(limits2).reshape(9,2), headless=True)
    print(sim2.ll)
    print(sim2.ul)
    traj2, traj_states2, err2 = sim2.simulate_p2p(sim2.rest_poses, goal)
    del sim2

    sim3 = Env(np.array(limits3).reshape(9,2), headless=True)
    print(sim3.ll)
    print(sim3.ul)
    traj3, traj_states3, err3 = sim3.simulate_p2p(sim3.rest_poses, goal)
    del sim3
    
    #plot 3D cartesian trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([i[0] for i in traj1], [i[1] for i in traj1], [i[2] for i in traj1], label="Robot 1", color="red", linestyle="dashed")
    ax.plot([i[0] for i in traj2], [i[1] for i in traj2], [i[2] for i in traj2], label="Robot 2", color="green")
    ax.plot([i[0] for i in traj3], [i[1] for i in traj3], [i[2] for i in traj3], label="Robot 3", color="blue", linestyle="dotted")
    ax.legend()
    plt.show()

    #plot joint trajectories in one big subplot
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('Joint Trajectories')
    axs[0, 0].plot([i[0] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[0, 0].plot([i[0] for i in traj_states2], label="Robot 2", color="green")
    axs[0, 0].plot([i[0] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[0, 0].set_title('Torso')
    axs[0, 0].legend()
    axs[0, 1].plot([i[1] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[0, 1].plot([i[1] for i in traj_states2], label="Robot 2", color="green")
    axs[0, 1].plot([i[1] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[0, 1].set_title('Torso')
    axs[0, 1].legend()
    axs[0, 2].plot([i[2] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[0, 2].plot([i[2] for i in traj_states2], label="Robot 2", color="green")
    axs[0, 2].plot([i[2] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[0, 2].set_title('Shoulder')
    axs[0, 2].legend()
    axs[1, 0].plot([i[3] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[1, 0].plot([i[3] for i in traj_states2], label="Robot 2", color="green")
    axs[1, 0].plot([i[3] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[1, 0].set_title('Shoulder')
    axs[1, 0].legend()
    axs[1, 1].plot([i[4] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[1, 1].plot([i[4] for i in traj_states2], label="Robot 2", color="green")
    axs[1, 1].plot([i[4] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[1, 1].set_title('Shoulder')
    axs[1, 1].legend()
    axs[1, 2].plot([i[5] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[1, 2].plot([i[5] for i in traj_states2], label="Robot 2", color="green")
    axs[1, 2].plot([i[5] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[1, 2].set_title('Elbow')
    axs[1, 2].legend()
    axs[2, 0].plot([i[6] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[2, 0].plot([i[6] for i in traj_states2], label="Robot 2", color="green")
    axs[2, 0].plot([i[6] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[2, 0].set_title('Wrist')
    axs[2, 0].legend()
    axs[2, 1].plot([i[7] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[2, 1].plot([i[7] for i in traj_states2], label="Robot 2", color="green")
    axs[2, 1].plot([i[7] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[2, 1].set_title('Wrist')
    axs[2, 1].legend()
    axs[2, 2].plot([i[8] for i in traj_states1], label="Robot 1", color="red", linestyle="dashed")
    axs[2, 2].plot([i[8] for i in traj_states2], label="Robot 2", color="green")
    axs[2, 2].plot([i[8] for i in traj_states3], label="Robot 3", color="blue", linestyle="dotted")
    axs[2, 2].set_title('Wrist')
    axs[2, 2].legend()
    plt.show()

    loss_1 = fastdtw.fastdtw(np.array(traj1), np.array(traj2), dist=2)[0]
    loss_2 = fastdtw.fastdtw(np.array(traj1), np.array(traj3), dist=2)[0]
    loss_3 = fastdtw.fastdtw(np.array(traj2), np.array(traj3), dist=2)[0]
    loss_4 = fastdtw.fastdtw(np.array(traj1), np.array(traj1), dist=2)[0]
    j_loss_1 = fastdtw.fastdtw(np.array(traj_states1), np.array(traj_states2), dist=2)[0]
    j_loss_2 = fastdtw.fastdtw(np.array(traj_states1), np.array(traj_states3), dist=2)[0]
    j_loss_3 = fastdtw.fastdtw(np.array(traj_states2), np.array(traj_states3), dist=2)[0]
    j_loss_4 = fastdtw.fastdtw(np.array(traj_states1), np.array(traj_states1), dist=2)[0]
    print("Loss 1:", loss_1)
    print("Loss 2:", loss_2)
    print("Loss 3:", loss_3)
    print("Self loss:", loss_4)
    print("Joint Loss 1:", j_loss_1)
    print("Joint Loss 2:", j_loss_2)
    print("Joint Loss 3:", j_loss_3)
    print("Self Joint Loss:", j_loss_4)


if __name__ == "__main__":

    full_limits = [-.0523, 1.309,
                   -0.61, 0.61,
                   -2.35, 1.57,
                -np.pi/2, np.pi/2,
                -.785, np.pi,
                -np.pi/2, np.pi/2,
                0, 2.53,
                -0.873, 1.047,
                -0.524, 0.349]
    
    blocked_shoulder = [-.0523, 1.309,
                   -0.61, 0.61,
                   1., 1.52,
                -np.pi/2, np.pi/2,
                -.785, np.pi,
                -np.pi/2, np.pi/2,
                0, 2.53,
                -0.873, 1.047,
                -0.524, 0.349]
    
    blocked_elbow = [-.0523, 1.309,
                   -0.61, 0.61,
                   -2.35, 1.57,
                -np.pi/2, np.pi/2,
                -.785, np.pi,
                0.2 -0.02, 0.2+0.02,
                0, 2.53,
                -0.873, 1.047,
                -0.524, 0.349]

    environment_test(full_limits)
    controller_test(full_limits)
    assignment_test(full_limits)
    environment_test(blocked_shoulder)
    controller_test(blocked_shoulder)
    assignment_test(blocked_shoulder)
    environment_test(blocked_elbow)
    controller_test(blocked_elbow)
    assignment_test(blocked_elbow)
    trajectory_test(full_limits, blocked_shoulder, blocked_elbow)