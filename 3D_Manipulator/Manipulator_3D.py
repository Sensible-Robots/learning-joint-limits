import numpy as np
from Simulator_rtb import Simulator
import time
import random
from itertools import product as prd
from sklearn.neighbors import KernelDensity
from pathos.multiprocessing import ProcessingPool
import pathos.helpers as ph
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, help="Number of samples", default=2000, required=False)
parser.add_argument("--thresh", type=float, help="Threshold for ABC", default=0.02, required=False)
parser.add_argument("--cartesian_thresh", type=float, help="Threshold for ABC in CartesianSpace", default=0.01, required=False)
parser.add_argument("--kernel", type=float, help="Kernel bandwidth for KDE", default=0.5, required=False)
parser.add_argument("--verbose", action="store_true", help="Verbose mode", required=False)
parser.add_argument("--runs", type=int, help="Number of runs", default=30, required=False)
parser.add_argument("--goals", type=int, help="Number of goals", default=10, required=False)
parser.set_defaults(verbose=False)

SAMPLE_BANK = parser.parse_args().samples
THRESH = parser.parse_args().thresh
KERNEL = parser.parse_args().kernel
THRESH_CARTESIAN = parser.parse_args().cartesian_thresh
CORES = ph.cpu_count()

NUM_GOALS = parser.parse_args().goals
RUNS = parser.parse_args().runs
WRITE_DATA = True
RUN_STATS = True
VERBOSE = parser.parse_args().verbose
random.seed()
KDE_SEED = False

NORMAL_LIMITS = [0, np.pi,
                -np.pi/3, np.pi,
                -np.pi/4, 4.5*np.pi/3,
                -np.pi/2, np.pi/2,
                0, np.pi,
                0, np.pi,
                -np.pi/4, np.pi/4]

START_STATE = np.array([np.pi/2, np.pi/2,
                        np.pi/6, np.pi/6,
                        np.pi/6, np.pi/6,
                        0., 0.,
                        np.pi/2, np.pi/2, 
                        np.pi/2, np.pi/2, 
                        0., 0.])

test_ = [np.pi/2-0.5, np.pi/2+0.5,
                -np.pi/3, np.pi,
                -np.pi/4, 4.5*np.pi/3,
                -np.pi/2, np.pi/2,
                0, np.pi,
                0, np.pi,
                -np.pi/4, np.pi/4]

tests = [test_]

samples = []
s_1 = np.linspace(NORMAL_LIMITS[0], NORMAL_LIMITS[1], 5).tolist()
s_2 = np.linspace(NORMAL_LIMITS[2], NORMAL_LIMITS[3], 5).tolist()
s_3 = np.linspace(NORMAL_LIMITS[4], NORMAL_LIMITS[5], 5).tolist()
s_4 = np.linspace(NORMAL_LIMITS[6], NORMAL_LIMITS[7], 5).tolist()
s_5 = np.linspace(NORMAL_LIMITS[8], NORMAL_LIMITS[9], 5).tolist()
s_6 = np.linspace(NORMAL_LIMITS[10], NORMAL_LIMITS[11], 5).tolist()
s_7 = np.linspace(NORMAL_LIMITS[12], NORMAL_LIMITS[13], 5).tolist()

c_1 = [s for s in s_1 if s <= START_STATE[0]]
c_2 = [s for s in s_1 if s > START_STATE[1]]
c_3 = [s for s in s_2 if s <= START_STATE[2]]
c_4 = [s for s in s_2 if s > START_STATE[3]]
c_5 = [s for s in s_3 if s <= START_STATE[4]]
c_6 = [s for s in s_3 if s > START_STATE[5]]
c_7 = [s for s in s_4 if s <= START_STATE[6]]
c_8 = [s for s in s_4 if s > START_STATE[7]]
c_9 = [s for s in s_5 if s <= START_STATE[8]]
c_10 = [s for s in s_5 if s > START_STATE[9]]
c_11 = [s for s in s_6 if s <= START_STATE[10]]
c_12 = [s for s in s_6 if s > START_STATE[11]]
c_13 = [s for s in s_7 if s <= START_STATE[12]]
c_14 = [s for s in s_7 if s > START_STATE[13]]

for i in prd(c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14):
    samples.append(i)
sample_bank = random.sample(samples, SAMPLE_BANK)

def generate_trajectory_from_ground_truth(start, ground_truth, goal):
    sim = Simulator(ground_truth)
    cartesian_trajectory, joint_trajectory = sim.simulate_p2p(start, goal, resolution=20)
    return cartesian_trajectory, joint_trajectory

def rejection_sampling(sample, start, observation, goal):
        sim = Simulator(sample)
        _, joint_trajectory = sim.simulate_p2p(start, goal, resolution=20)
        loss = np.linalg.norm(np.array(observation) - np.array(joint_trajectory))
        if loss < THRESH:
            return sample
        else:
            return None
        
def rejection_sampling_cartesian(sample, start, observation, goal):
        sim = Simulator(sample)
        cart_trajectory, _ = sim.simulate_p2p(start, goal, resolution=20)
        loss = np.linalg.norm(np.array(observation) - np.array(cart_trajectory))
        if loss < THRESH_CARTESIAN:
            return sample
        else:
            return None

def sample_goal(constraints, fixed_grid=False):
    goal = []
    if fixed_grid:
        pass
    else:
        for i in range(0, len(constraints), 2):
            r = random.uniform(constraints[i], constraints[i+1])
            goal.append(r)
    
    return goal

def dump_run_data(run_data, name):
    with open("run_data_test_" + name + ".pkl", "wb") as f:
        pkl.dump(run_data, f)

def get_task_stats(min_obs, max_obs, ml_constraints, ground_truth, goal):
    
    run_data = {}
    obs = []
    for i in range(len(min_obs)):
        obs.append(min_obs[i])
        obs.append(max_obs[i])
    run_data["observed"] = obs
    run_data["ml_constraints"] = ml_constraints
    run_data["ground_truth"] = ground_truth
    run_data["goal"] = goal
    run_data["mlc_loss"] = np.sum(np.linalg.norm(np.array(ml_constraints) - np.array(ground_truth), axis=0))
    run_data["obs_loss"] = np.sum(np.linalg.norm(np.array(obs) - np.array(ground_truth), axis=0))
    
    return run_data

def get_task_stats_cartesian(ml_constraints, ground_truth, goal):
    
    run_data = {}
    run_data["ml_constraints_cartesian"] = ml_constraints
    run_data["ground_truth"] = ground_truth
    run_data["goal"] = goal
    run_data["mlc_loss_cartesian"] = np.sum(np.linalg.norm(np.array(ml_constraints) - np.array(ground_truth), axis=0))
    
    return run_data

def do_kde_calculations(KDE_collection, possible_constraints):
    #find max likelihood constraints
    kde_scores = np.zeros(len(possible_constraints))
    for i in range(len(KDE_collection)):
        kde_14D = KDE_collection[i]
        score = kde_14D.score_samples(np.array(possible_constraints))
        kde_scores = kde_scores + np.array(score)

    max_likelihood_index = np.argmax(kde_scores)
    mlc_14D = possible_constraints[max_likelihood_index]
    return mlc_14D

def test_estimated_parameters(mlc, obs, gt, cartesian, num_goals):
    stats = []
    sim_gt = Simulator(gt)
    sim_mlc = Simulator(mlc)
    sim_obs = Simulator(obs)
    sim_cartesian = Simulator(cartesian)
    sim_fixed = Simulator(NORMAL_LIMITS)

    for _ in range(num_goals):
        stat = {}
        goal = sample_goal(gt)
        trajectory, joint_trajectory = sim_gt.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_mlc, joint_trajectory_mlc = sim_mlc.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_obs, joint_trajectory_obs = sim_obs.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_fixed, joint_trajectory_fixed = sim_fixed.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_cartesian, joint_trajectory_cartesian = sim_cartesian.simulate_p2p(START_STATE[::2], goal, resolution=20)
        stat["gt"] = joint_trajectory
        stat["mlc"] = joint_trajectory_mlc
        stat["obs"] = joint_trajectory_obs
        stat["fixed"] = joint_trajectory_fixed
        stat["goal"] = goal
        #calculate the losses in joint and cartesian space
        mlc_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_mlc), axis=0))
        obs_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_obs), axis=0))
        fixed_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_fixed), axis=0))
        cartesian_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_cartesian), axis=0))

        c_mlc_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_mlc), axis=0))
        c_obs_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_obs), axis=0))
        c_fixed_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_fixed), axis=0))
        c_cartesian_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_cartesian), axis=0))

        stat["mlc_loss"] = mlc_loss
        stat["obs_loss"] = obs_loss
        stat["fixed_loss"] = fixed_loss
        stat["cartesian_loss"] = cartesian_loss
        stat["c_mlc_loss"] = c_mlc_loss
        stat["c_obs_loss"] = c_obs_loss
        stat["c_fixed_loss"] = c_fixed_loss
        stat["c_cartesian_loss"] = c_cartesian_loss

        stats.append(stat)

    return stats

def test_estimated_parameters_single_run(mlc, obs, gt, cartesian, goal_samples):
    stats = []
    sim_gt = Simulator(gt)
    sim_mlc = Simulator(mlc)
    sim_obs = Simulator(obs)
    sim_cartesian = Simulator(cartesian)
    sim_fixed = Simulator(NORMAL_LIMITS)

    for _ in range(len(goal_samples)):
        stat = {}
        goal = goal_samples[_]
        trajectory, joint_trajectory = sim_gt.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_mlc, joint_trajectory_mlc = sim_mlc.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_obs, joint_trajectory_obs = sim_obs.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_fixed, joint_trajectory_fixed = sim_fixed.simulate_p2p(START_STATE[::2], goal, resolution=20)
        trajectory_cartesian, joint_trajectory_cartesian = sim_cartesian.simulate_p2p(START_STATE[::2], goal, resolution=20)
        stat["gt"] = joint_trajectory
        stat["mlc"] = joint_trajectory_mlc
        stat["obs"] = joint_trajectory_obs
        stat["fixed"] = joint_trajectory_fixed
        stat["goal"] = goal
        #calculate the losses in joint and cartesian space
        mlc_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_mlc), axis=0))
        obs_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_obs), axis=0))
        fixed_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_fixed), axis=0))
        cartesian_loss = np.sum(np.linalg.norm(np.array(joint_trajectory) - np.array(joint_trajectory_cartesian), axis=0))

        c_mlc_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_mlc), axis=0))
        c_obs_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_obs), axis=0))
        c_fixed_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_fixed), axis=0))
        c_cartesian_loss = np.sum(np.linalg.norm(np.array(trajectory) - np.array(trajectory_cartesian), axis=0))

        stat["mlc_loss"] = mlc_loss
        stat["obs_loss"] = obs_loss
        stat["fixed_loss"] = fixed_loss
        stat["cartesian_loss"] = cartesian_loss
        stat["c_mlc_loss"] = c_mlc_loss
        stat["c_obs_loss"] = c_obs_loss
        stat["c_fixed_loss"] = c_fixed_loss
        stat["c_cartesian_loss"] = c_cartesian_loss

        stats.append(stat)

    return stats

if __name__ == "__main__":
    k = 1
    run_data = {"kernel": KERNEL, "threshold": THRESH, "goals": NUM_GOALS, "runs": RUNS, "data": [], "cartesian_data": []}
    iterations = {}
    all_stats = []
    #sample 20 goals for the learning test
    goal_samples = [sample_goal(tests[0]) for i in range(20)]
    for test in tests:
        GOAL = []
        random.seed(1)
        for i in range(NUM_GOALS):
            g = sample_goal(test)
            GOAL.append(g)
        print("Test ", k) if VERBOSE else None
        random.seed()
        GROUND_TRUTH = test
        TEST = k
        k += 1
        p = ProcessingPool(CORES)
        for _ in range(RUNS):
            possible_constraints = []
            c_possible_constraints = []
            trajectories_stack = []
            KDE_collection = []
            c_KDE_collection = []
            test_data = []
            c_test_data = []
            sample_bank = random.sample(samples, SAMPLE_BANK)
            c_sample_bank = sample_bank
            st = time.time()
            for j in range(len(GOAL)):
                print("----------------------------------") if VERBOSE else None
                print("ABC for Task " + str(j+1)) if VERBOSE else None
                c_trajectory, trajectory = generate_trajectory_from_ground_truth(start=START_STATE[::2], goal=GOAL[j], ground_truth=GROUND_TRUTH)
                
                # JOINT SPACE ABC
                trajectories_stack.extend(trajectory)
                min_obs = np.array(trajectories_stack).min(axis=0)
                max_obs = np.array(trajectories_stack).max(axis=0)
                min_obs = np.clip(min_obs, GROUND_TRUTH[::2], None)
                max_obs = np.clip(max_obs, None, GROUND_TRUTH[1::2])
                trajectories = [trajectory for i in range(SAMPLE_BANK)]
                goals = [GOAL[j] for i in range(SAMPLE_BANK)]
                starts = [START_STATE[::2] for i in range(SAMPLE_BANK)]
                results = list(p.map(rejection_sampling, sample_bank, starts, trajectories, goals))
                results = [r for r in results if type(r) != type(None)]
                print("Number of samples (JS): ", len(results)) if VERBOSE else None
                #find best matching constraints
                if len(results) > 0:
                    possible_constraints.extend(results)
                    kde_14D = KernelDensity(kernel='gaussian', bandwidth=KERNEL).fit(np.array(results))
                    KDE_collection.append(kde_14D)
                    mlc_14D = do_kde_calculations(KDE_collection, possible_constraints)
                    test_data.append(get_task_stats(min_obs, max_obs, mlc_14D, GROUND_TRUTH, GOAL[j]))
                    sample_bank = kde_14D.sample(SAMPLE_BANK, random_state=0 if KDE_SEED else None)
                    #clip samples to maximum and minimum observed so far
                    for sample in sample_bank:
                            sample[0] = np.clip(sample[0], NORMAL_LIMITS[0], min_obs[0])
                            sample[1] = np.clip(sample[1], max_obs[0], NORMAL_LIMITS[1])
                            sample[2] = np.clip(sample[2], NORMAL_LIMITS[2], min_obs[1])
                            sample[3] = np.clip(sample[3], max_obs[1], NORMAL_LIMITS[3])
                            sample[4] = np.clip(sample[4], NORMAL_LIMITS[4], min_obs[2])
                            sample[5] = np.clip(sample[5], max_obs[2], NORMAL_LIMITS[5])
                            sample[6] = np.clip(sample[6], NORMAL_LIMITS[6], min_obs[3])
                            sample[7] = np.clip(sample[7], max_obs[3], NORMAL_LIMITS[7])
                            sample[8] = np.clip(sample[8], NORMAL_LIMITS[8], min_obs[4])
                            sample[9] = np.clip(sample[9], max_obs[4], NORMAL_LIMITS[9])
                            sample[10] = np.clip(sample[10], NORMAL_LIMITS[10], min_obs[5])
                            sample[11] = np.clip(sample[11], max_obs[5], NORMAL_LIMITS[11])
                            sample[12] = np.clip(sample[12], NORMAL_LIMITS[12], min_obs[6])
                            sample[13] = np.clip(sample[13], max_obs[6], NORMAL_LIMITS[13])
                else:
                    test_data.append(get_task_stats(min_obs, max_obs, mlc_14D, GROUND_TRUTH, GOAL[j]))

                # CARTESIAN SPACE ABC
                c_trajectories = [c_trajectory for i in range(SAMPLE_BANK)]
                c_goals = [GOAL[j] for i in range(SAMPLE_BANK)]
                c_starts = [START_STATE[::2] for i in range(SAMPLE_BANK)]
                c_results = list(p.map(rejection_sampling_cartesian, sample_bank, c_starts, c_trajectories, c_goals))
                c_results = [r for r in c_results if type(r) != type(None)]
                print("Number of samples (CS): ", len(c_results)) if VERBOSE else None
                #find best matching constraints
                if len(c_results) > 0:
                    c_possible_constraints.extend(c_results)
                    c_kde_14D = KernelDensity(kernel='gaussian', bandwidth=KERNEL).fit(np.array(c_results))
                    c_KDE_collection.append(c_kde_14D)
                    c_mlc_14D = do_kde_calculations(c_KDE_collection, c_possible_constraints)
                    c_test_data.append(get_task_stats_cartesian(c_mlc_14D, GROUND_TRUTH, GOAL[j]))
                    c_sample_bank = c_kde_14D.sample(SAMPLE_BANK, random_state=0 if KDE_SEED else None)
                else:
                    c_test_data.append(get_task_stats_cartesian(c_mlc_14D, GROUND_TRUTH, GOAL[j]))

                obs = []
                for i in range(0, len(min_obs)):
                    if max_obs[i] - min_obs[i] < 0.01:
                        obs.append(min_obs[i]- 0.05)
                        obs.append(max_obs[i] + 0.05)
                    else:
                        obs.append(min_obs[i])
                        obs.append(max_obs[i])

                iteration_test_data = test_estimated_parameters_single_run(mlc_14D, obs, GROUND_TRUTH, c_mlc_14D, goal_samples)
                iterations[str(j)] = iteration_test_data
            
            dump_run_data(tuple(test_data), str(time.time()) + "_" + str(TEST)) if WRITE_DATA else None
            dump_run_data(tuple(c_test_data), str(time.time()) + "_" + str(TEST) + "_cartesian") if WRITE_DATA else None

            rounded_ml = [round(elem, 2) for elem in mlc_14D]
            rounded_gt = [round(elem, 2) for elem in GROUND_TRUTH]
        
            obs = []
            for i in range(0, len(min_obs)):
                if max_obs[i] - min_obs[i] < 0.01:
                    obs.append(min_obs[i]- 0.05)
                    obs.append(max_obs[i] + 0.05)
                else:
                    obs.append(min_obs[i])
                    obs.append(max_obs[i])

            rounded_obs = [round(elem, 2) for elem in obs]
            
            print("Ground Truth: ", rounded_gt) if VERBOSE else None
            print("----------------------------------") if VERBOSE else None
            print("ML Constraints: ", rounded_ml) if VERBOSE else None
            print("Difference: ", np.array(rounded_ml) - np.array(rounded_gt)) if VERBOSE else None
            print("Total difference: ", np.linalg.norm(np.array(rounded_ml) - np.array(rounded_gt))) if VERBOSE else None
            print("----------------------------------") if VERBOSE else None
            print("ML Constraints (Cartesian): ", c_mlc_14D) if VERBOSE else None
            print("Difference: ", np.array(c_mlc_14D) - np.array(rounded_gt)) if VERBOSE else None
            print("Total difference: ", np.linalg.norm(np.array(c_mlc_14D) - np.array(rounded_gt))) if VERBOSE else None
            print("----------------------------------") if VERBOSE else None
            print("Observed: ", rounded_obs) if VERBOSE else None
            print("Difference: ", np.array(rounded_obs) - np.array(rounded_gt)) if VERBOSE else None
            print("Total difference: ", np.linalg.norm(np.array(rounded_obs) - np.array(rounded_gt))) if VERBOSE else None


            end = time.time()
            print("Time taken: ", end - st) if VERBOSE else None

            print("Testing estimated parameters...") if VERBOSE else None
            random.seed(1)
            test_stats = test_estimated_parameters(mlc_14D, obs, c_mlc_14D, GROUND_TRUTH, 20)
            dump_run_data(test_stats, str(time.time()) + "_test_trajectories_" + str(TEST)) if WRITE_DATA else None
            dump_run_data(iterations, str(time.time()) + "_test_trajectories_" + str(TEST) + "_iterations") if WRITE_DATA else None
