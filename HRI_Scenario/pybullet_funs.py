import pybullet as p
import numpy as np

def getJointStates(robot, sim=0):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot), physicsClientId=sim), physicsClientId=sim)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot, sim=0):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot, physicsClientId=sim)), physicsClientId=sim)
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot, physicsClientId=sim))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques

def is_valid(q: list, limits: list) -> bool:
    """Checks if the joint state is within the joint limits.
    :param q (list): The joint state to check
    :param limits (list): The joint limits to check against
    :return: True if the joint state is within the joint limits, False otherwise"""
    if q is None:
        return False

    for i in range(len(q)):
        if q[i] < limits[i,0] or q[i] > limits[i,1]:
            return False
    return True

def create_contact(robot, object, sim, ee_idx) -> int:
        """Creates a contact between the robot and the object.
        :param robot (int): The robot to create the contact with
        :param object (int): The object to create the contact with
        :return: None"""
        
        _constraint = sim.createConstraint(robot, ee_idx, object, -1,
        p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition= [0, 0, 0], childFramePosition=[0, 0, -0.11],
        parentFrameOrientation= [0, 0, 0], childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        return _constraint

def remove_contact(constraint, sim) -> None:
    """Removes a contact between the robot and the object.
    :param constraint (int): The constraint to remove
    :return: None"""
    
    sim.removeConstraint(constraint)
class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.p_id)

        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                p.addUserDebugText(text, position + radius)
            )
        
        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        p.removeBody(self.marker_id, physicsClientId=self.p_id)
        for debug_item_id in self.debug_item_ids:
            p.removeUserDebugItem(debug_item_id)