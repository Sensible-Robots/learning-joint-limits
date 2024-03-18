import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import time

#function timing decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time() - ts
        print(f"{method.__name__} took {te} seconds")
        return result
    return timed

NORMAL_LIMITS = [0, np.pi,
                -np.pi/3, np.pi,
                -np.pi/4, 4.5*np.pi/3,
                -np.pi/2, np.pi/2,
                0, np.pi,
                0, np.pi,
                -np.pi/4, np.pi/4]
#Shoulder Roll Lower, Shoulder Roll Upper, Shoulder Pitch Lower, Shoulder Pitch Upper, Shoulder Yaw Lower, Shoulder Yaw Upper, Elbow Pitch Lower, Elbow Pitch Upper, Wrist Yaw Lower, Wrist Yaw Upper, Wrist Pitch Lower, Wrist Pitch Upper, Wrist Roll Lower, Wrist Roll Upper
class World:
    def __init__(self, limits):
        self.links = [40, 41, 10]
        self.robot_limits = limits
        self.rest_position = [np.pi/2, np.pi/6, np.pi/6, 0., np.pi/2, np.pi/2, 0.]
        self.robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(a=0, alpha=np.pi/2, qlim=self.robot_limits[0:2]),
                rtb.RevoluteDH(a=0, alpha=-np.pi/2, qlim=self.robot_limits[2:4]),
                rtb.RevoluteDH(d=self.links[0], alpha=np.pi/2, qlim=self.robot_limits[4:6]),
                rtb.RevoluteDH(a=self.links[1], alpha=0, qlim=self.robot_limits[6:8]),
                rtb.RevoluteDH(a=0, alpha=-np.pi/2, qlim=self.robot_limits[8:10]),
                rtb.RevoluteDH(a=0, alpha=-np.pi/2, qlim=self.robot_limits[10:12]),
                rtb.RevoluteDH(d=self.links[2], alpha=np.pi/2, qlim=self.robot_limits[12:14]),
            ], name="robot")
        self.start = self.rest_position
        self.start_cartesian = self.forward_kinematics(self.start)
    
    def forward_kinematics(self, joint_values):
        self.T = self.robot.fkine(joint_values)
        return [self.T.t[0], self.T.t[1], self.T.t[2]]

    def inverse_kinematics(self, ee_position):
        q = self.robot.ik_lm_chan(SE3(ee_position[0], ee_position[1], ee_position[2]), q0=np.array(self.rest_position), ilimit=500, reject_jl=True)
        if q[1]==1 and q[0] is not None:
            return [q[0][0], q[0][1], q[0][2], q[0][3], q[0][4], q[0][5], q[0][6]], True
        else:
            return None, False

class Simulator:
    def __init__(self, robot_limits):
        self.env = World(robot_limits)
        self.P = np.eye(3) * 5
        self.D = np.eye(3) * 0.5
        self.KN = 10
        self.dt = 0.02
        self.lim_buffer = 0.01
        self.ll = np.array(robot_limits[0::2])
        self.ul = np.array(robot_limits[1::2])

    def generate_trajectory(self, center, radius, tilt, resolution):
        path = []
        dpath = [[0,0,0]]
        for i in np.linspace(-np.pi, np.pi, resolution):
            path.append([center[0] + radius*np.cos(i), center[1] + radius*np.sin(i), center[2] + tilt*np.sin(i)])
        
        for j in range(len(path)-1):
                dpath.append([(path[j+1][0]-path[j][0])/10*self.dt, (path[j+1][1]-path[j][1])/10*self.dt, (path[j+1][2]-path[j][2])/10*self.dt])

        return path, dpath

    def controller(self, state, goal, goal_vel, pos, vel):
        # joint space
        q = np.array(state).reshape(len(state), 1)
        q_l = np.array([self.env.robot_limits[i] for i in range(0, 14, 2)]).reshape(len(state), 1)
        q_u = np.array([self.env.robot_limits[i + 1] for i in range(0, 14, 2)]).reshape(len(state), 1)

        # task space
        x = np.array(pos).reshape(len(pos), 1)
        x_dot = np.array(vel).reshape(len(vel), 1)
        x_g = np.array(goal).reshape(len(goal), 1)
        x_dot_g = np.array(goal_vel).reshape(len(goal_vel), 1)

        # jacobian, pinv, nullspace
        J = self.env.robot.jacob0(state, half="trans")
        J_pinv = np.linalg.pinv(J)
        N = np.eye(7) - J_pinv @ J

        # tracking errors
        e_x = x_g - x   # position error
        e_x_dot = x_dot_g - x_dot

        # joint limit penalties (projected into nullspace)
        penalty_lower_N = 2 * self.KN * np.clip(self.lim_buffer + q_l - q, 0, 0.2)   # if q > q_l, no penalty
        penalty_upper_N = 2 * self.KN * np.clip(q - q_u - self.lim_buffer, 0, 0.2)   # if q < q_u, no penalty

        # joint limit penalties (unprojected, hard limits)
        penalty_lower = self.KN * np.clip(q_l - q, 0, None)  # if q > q_l, no penalty
        penalty_upper = self.KN * np.clip(q - q_u, 0, None)  # if q < q_u, no penalty

        # commands
        q_new = q
        q_new += self.dt * (J_pinv @ self.P @ e_x + J_pinv @ self.D @ e_x_dot)    # task space errors
        q_new += self.dt * (N @ penalty_lower_N + N @ penalty_upper_N)      # nullspace joint limits
        q_new += self.dt * (penalty_lower + penalty_upper)              # hard joint limits
        new_state = q_new.T.tolist()[0]

        # clip for joint limits, as done in ROS
        for i in range(len(new_state)):
            if new_state[i] < self.ll[i]:
                new_state[i] = self.ll[i]
            
            elif new_state[i] > self.ul[i]:
                new_state[i] = self.ul[i]
        
        return new_state, e_x, e_x_dot

    def simulate(self, center, radius, tilt, resolution=100):
        path, velocity = self.generate_trajectory(center, radius, tilt, resolution)
        state = self.env.start
        x = self.env.start_cartesian
        traj = [x]
        traj_states = [state]
        vel = [0, 0, 0]
        error = []
        error_v = []
        for goal, goal_vel in zip(path, velocity):
            new_state, e, e_v = self.controller(state, goal, goal_vel, x, vel)
            new_x = self.env.forward_kinematics(new_state)
            vel = np.subtract(new_x, x) / self.dt
            traj.append(new_x)
            traj_states.append(new_state)
            state = new_state
            x = new_x
            error.append(e)
            error_v.append(e_v)
        
        return traj, traj_states, path, error, error_v
    
    def simulate_p2p(self, start, goal, resolution=100):
        state = start
        x = self.env.forward_kinematics(state)
        goal = self.env.forward_kinematics(goal)
        traj = [x]
        traj_states = [state]
        vel = [0, 0, 0]
        error = []
        error_v = []
        for _ in range(resolution):
            new_state, e, e_v = self.controller(state, goal, [0.]*3, x, vel)
            new_x = self.env.forward_kinematics(new_state)
            vel = np.subtract(new_x, x) / self.dt
            traj.append(new_x)
            traj_states.append(new_state)
            state = new_state
            x = new_x
            error.append(e)
            error_v.append(e_v)
        
        return traj, traj_states