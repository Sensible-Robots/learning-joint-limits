from Learner import Learner
from Env import Env, Goal
import numpy as np
from random import shuffle
import argparse
import pickle as pkl
import time
import os

"""Main script to run the HRI scenario, where the learner tries to estimate the joint limits of the human by observing the human's joint trajectory across the tasks within the block stacking domain"""

#parse command line arguments
parser = argparse.ArgumentParser(description='Run the HRI scenario')
parser.add_argument("--samples", type=int, help="Number of samples", default=5000, required=False)
parser.add_argument("--thresh", type=float, help="Threshold for ABC", default=1.5, required=False)
parser.add_argument("--cartesian_thresh", type=float, help="Threshold for ABC in Cartesian Space", default=2.5, required=False)
parser.add_argument("--kernel", type=float, help="Kernel bandwidth for KDE", default=0.5, required=False)
parser.add_argument("--runs", type=int, help="Number of runs", default=1, required=False)
parser.add_argument("--observations", type=int, help="Number of observations", default=10, required=False)
parser.set_defaults(verbose=False)

class Experiment:
    
    def __init__(self, limits):
        samples = parser.parse_args().samples
        thresh = parser.parse_args().thresh
        cartesian_thresh = parser.parse_args().cartesian_thresh
        kernel = parser.parse_args().kernel

        self.env = Env(limits, headless=True)
        self.learner = Learner(samples, thresh, cartesian_thresh, kernel)
        self.limits = limits
        self.static_goal_set = [self._get_random_goal() for _ in range(20)]

        # create results directory if it doesn't yet exist
        if not os.path.exists("HRI_Scenario/results"):
            os.mkdir("HRI_Scenario/results")
        else:
            pass

    def _get_random_goal(self) -> None:
        """Sets the goal of the domain to be a random block stacking configuration"""
        order = [1, 2, 3, 4]
        shuffle(order)
        return Goal(np.random.uniform(0.3, 0.6), np.random.uniform(-0.2, 0.2), order)
    
    def run(self, num_traj) -> None:
        tic = time.time()
        """Runs the HRI scenario, where the learner tries to estimate the joint limits of the human by observing the human's joint trajectory across the tasks within the block stacking domain"""
        #get observations and ground truth from the environment
        observations = []
        cartesian = []
        box_states = []
        i = 0
        while i < num_traj:
            goal = self._get_random_goal()
            self.env.reset(self.env.main_sim, input_goal=goal)
            box_states.append(self.env.get_state()[1])
            assignments, trajs, jtrajs = self.env.generate_assignment_and_task_trajectories(goal)
            if jtrajs is not None:
                [observations.append(ob) for ob in jtrajs if ob is not None]
                [cartesian.append(ob) for ob in trajs if ob is not None]
                t = [ob for ob in jtrajs if ob is not None]
                i += len(t)
        #update the learner with the observations
        print("Number of observations: ", len(observations))

        res = []


        for case in zip(cartesian, observations, box_states):
            self.learner.learn(*case)
            print("Best params: ", self.learner.best_params)
            print("Best cartesian params: ", self.learner.c_best_params)
            observed_limits = np.array(list(zip(self.learner.min[2:], self.learner.max[2:])))
            print("Observed min max: ", observed_limits)
            if self.learner.best_params is not None and self.learner.c_best_params is not None:
                result = self.assess_performance(self.limits, len(self.static_goal_set))
                res.append(result)
            else:
                res.append(None)

        #get the learner's estimate of the joint limits
        estimate = np.array(self.learner.best_params).reshape(7,2) if self.learner.best_params is not None else None
        cartesian_estimate = np.array(self.learner.c_best_params).reshape(7,2) if self.learner.c_best_params is not None else None
        updates = self.learner.learner_updates
        cartesian_updates = self.learner.c_learner_updates
        toc = time.time() - tic

        print("Ground truth: ", self.limits)
        print("Estimate: ", estimate) if estimate is not None else None
        print("Cartesian estimate: ", cartesian_estimate) if cartesian_estimate is not None else None
        print("Number of updates: ", updates)
        print("Number of cartesian updates: ", cartesian_updates)
        print("Time: ", toc)

        with open("results/results_" + str(time.time()) +".pkl", "wb") as f:
            pkl.dump([self.limits, estimate, cartesian_estimate, res, updates, cartesian_updates], f)

    def assess_performance(self, gt_limits, num_goals) -> float:
        """Assesses the performance of the learner by comparing the learner's estimate of the joint limits to the ground truth joint limits"""
        gt_sim = Env(np.array(gt_limits).reshape(9,2), headless=True)
        est_sim = Env(np.concatenate((np.array([-.0523, 1.309, -0.61, 0.61]), np.array(self.learner.best_params))).reshape(9,2), headless=True)
        #join self.min and self.max into one array that interchanges between the two
        naive_limits = np.hstack((np.array(self.learner.min[2:]).T, np.array(self.learner.max[2:]).T)).reshape(7,2)
        naive_sim = Env(np.concatenate((gt_limits[:2], naive_limits)), headless=True)
        cartesian_sim = Env(np.concatenate((np.array([-.0523, 1.309, -0.61, 0.61]), np.array(self.learner.c_best_params))).reshape(9,2), headless=True)

        i = 0
        gt_labels = []
        est_labels = []
        naive_labels = []
        cartesian_labels = []

        while i < num_goals:
            print("Goal: ", i)
            goal = self.static_goal_set[i]
            gt_sim.reset(gt_sim.main_sim, input_goal=goal)
            est_sim.reset(est_sim.main_sim, input_goal=goal)
            naive_sim.reset(naive_sim.main_sim, input_goal=goal)
            cartesian_sim.reset(cartesian_sim.main_sim, input_goal=goal)
            gt_assignments, gt_trajs, gt_jtrajs = gt_sim.generate_assignment_and_task_trajectories(goal)
            est_assignments, est_trajs, est_jtrajs = est_sim.generate_assignment_and_task_trajectories(goal)
            naive_assignments, naive_trajs, naive_jtrajs = naive_sim.generate_assignment_and_task_trajectories(goal)
            cartesian_assignments, cartesian_trajs, cartesian_jtrajs = cartesian_sim.generate_assignment_and_task_trajectories(goal)
            gt_labels.extend(gt_assignments)
            est_labels.extend(est_assignments)
            naive_labels.extend(naive_assignments)
            cartesian_labels.extend(cartesian_assignments)
            i += 1

        return {"labels": gt_labels, "est_labels": est_labels, "c_labels": cartesian_labels, "n_labels": naive_labels}
    

if __name__ == "__main__":
    blocked_shoulder = [-.0523, 1.309,
                    -0.61, 0.61,
                    .8, 1.52,
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



    exp = Experiment(np.array(blocked_shoulder).reshape(9,2))
    for i in range(parser.parse_args().runs):
        exp.run(parser.parse_args().observations)