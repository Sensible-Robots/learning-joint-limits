from Env import Env
import numpy as np
import random
import pathos.helpers as ph
from pathos.multiprocessing import ProcessingPool
import random
from itertools import product as prd
from sklearn.neighbors import KernelDensity
import fastdtw
from scipy.stats.qmc import LatinHypercube
from scipy.stats.qmc import scale

"""Learner class that takes environment observations and estimates the parameters of the human by implementing ABC in joint trajectory space"""

class Learner:
    def __init__(self, samples=2000, threshold=0.01, cartesian_threshold=0.2, kernel=0.5):
        self.best_params = None
        self.c_best_params = None

        self.normal_limits = [-2.35, 1.57,
                -np.pi/2, np.pi/2,
                -.785, np.pi,
                -np.pi/2, np.pi/2,
                0, 2.53,
                -0.873, 1.047,
                -0.524, 0.349]
        self.start_state = [1.4, -0.3, 0.17, 0.2, np.pi/6, 0., 0.]
        self.env = Env(np.concatenate((np.array([-.0523, 1.309, -0.61, 0.61]), np.array(self.normal_limits))).reshape(9,2), headless=True)

        # algo hyperparameters
        self.sample_bank = samples
        self.threshold = threshold
        self.c_threshold = cartesian_threshold
        self.kernel_width = kernel
        self.cores = ph.cpu_count()
        self.kde_seed = True
        self.random_seed = True
        self.learner_updates = 0
        self.c_learner_updates = 0
        self.DEBUG = False

        # ABC parameters
        self.j_kde_collection = []
        self.c_kde_collection = []
        self.observations = []
        self.possible_constraints = []
        self.min = None
        self.max = None
        self.truncate = 500

    def learn(self, cartesian_observation, observation, state) -> None:
        """Learns the parameters of the human by implementing ABC in joint trajectory and cartesian trajectory space, and updates the best_params attribute in the learner class.
        :param env (Env): The environment to learn the parameters of the human in, class that interfaves with the pybullet simulation
        :param observation (list): The observation of the human's joint trajectory with unknown joint limits across the tasks within the block stacking domain
        :param state (list): The state of the boxes at the start of the trajectory"""

        #store observation in the stack of observations for clipping purposes
        self.observations.extend(observation)
        self.min, self.max = self._get_min_max()

        if len(self.j_kde_collection) == 0:
            #samples = self._biased_sampling()
            #samples = self._strong_prior_sampling()
            samples = self._latin_hypercube_sampling()
            #samples = self._lhs_strong_prior()
            #samples = self._dummy_sampling()
        else:
            #samples = self._sample_kde()
            #samples = self._biased_sampling()
            #samples = self._strong_prior_sampling()
            samples = self._latin_hypercube_sampling()
            #samples = self._lhs_strong_prior()
            #samples = self._dummy_sampling()

        if len(self.c_kde_collection) == 0:
            #c_samples = self._biased_sampling()
            #c_samples = self._strong_prior_sampling()
            c_samples = self._latin_hypercube_sampling()
            #c_samples = self._lhs_strong_prior()
            #c_samples = self._dummy_sampling()
        else:
            #c_samples = self._sample_kde()
            #c_samples = self._biased_sampling()
            #c_samples = self._strong_prior_sampling()
            c_samples = self._latin_hypercube_sampling()
            #c_samples = self._lhs_strong_prior()
            #c_samples = self._dummy_sampling()

        starts = [observation[0] for i in range(self.sample_bank)]
        goals = [observation[-1] for i in range(self.sample_bank)]
        states = [state for i in range(self.sample_bank)]
        cartesian_trajectories = [cartesian_observation for i in range(self.sample_bank)]
        observations = [observation for i in range(self.sample_bank)]
        flags_joint = [0 for i in range(self.sample_bank)]
        flags_cartesian = [1 for i in range(self.sample_bank)]
        results = []
        
        p = ProcessingPool(self.cores)
        results = list(p.map(self._rejection_sampling, samples, states, starts, observations, cartesian_trajectories, goals, flags_joint))
        c_results = list(p.map(self._rejection_sampling, c_samples, states, starts, observations, cartesian_trajectories, goals, flags_cartesian))
        results = [r for r in results if type(r[0]) != type(None)]
        c_results = [r for r in c_results if type(r[0]) != type(None)]
        sorted_results = [sort[0] for sort in sorted(results, key=lambda x: x[1])]
        sorted_c_results = [sort[0] for sort in sorted(c_results, key=lambda x: x[1])]

        if len(sorted_results) > 0:
            print("Found " + str(len(sorted_results)) + " samples for JS")
            #if there are more than 2000 samples, take the best 2000
            if len(sorted_results) > self.truncate:
                sorted_results = sorted_results[:self.truncate]
            self.possible_constraints.extend(sorted_results)
            kde_14D = KernelDensity(kernel='gaussian', bandwidth=self.kernel_width).fit(np.array(sorted_results))
            self.j_kde_collection.append(kde_14D)
            self.learner_updates += 1
        else:
            print("No samples found for JS")

        if len(sorted_c_results) > 0:
            print("Found " + str(len(sorted_c_results)) + " samples for CS")
            #if there are more than 2000 samples, take the best 2000
            if len(sorted_c_results) > self.truncate:
                sorted_c_results = sorted_c_results[:self.truncate]
            self.possible_constraints.extend(sorted_c_results)
            kde_14D = KernelDensity(kernel='gaussian', bandwidth=self.kernel_width).fit(np.array(sorted_c_results))
            self.c_kde_collection.append(kde_14D)
            self.c_learner_updates += 1
        else:
            print("No samples found for CS")

        if len(self.j_kde_collection) > 0:
            self.best_params = self._do_kde_calculations(0)

        if len(self.c_kde_collection) > 0:
            self.c_best_params = self._do_kde_calculations(1)

    def _sample_kde(self, flag: int):
        """Samples the KDEs to generate a sample bank of size bank_size
        :return: The sample bank"""
        #get last KDE
        kde = self.j_kde_collection[-1] if  flag == 0 else self.c_kde_collection[-1]
        sample_bank = kde.sample(self.sample_bank, random_state=0 if self.kde_seed else None)
        #clip samples to maximum and minimum observed so far
        if len(self.observations) > 0:
            min_obs = np.min(np.array(self.observations), axis=0)
            max_obs = np.max(np.array(self.observations), axis=0)

        for sample in sample_bank:
                sample[0] = np.clip(sample[0], self.normal_limits[0], min_obs[0])
                sample[1] = np.clip(sample[1], max_obs[0], self.normal_limits[1])
                sample[2] = np.clip(sample[2], self.normal_limits[2], min_obs[1])
                sample[3] = np.clip(sample[3], max_obs[1], self.normal_limits[3])
                sample[4] = np.clip(sample[4], self.normal_limits[4], min_obs[2])
                sample[5] = np.clip(sample[5], max_obs[2], self.normal_limits[5])
                sample[6] = np.clip(sample[6], self.normal_limits[6], min_obs[3])
                sample[7] = np.clip(sample[7], max_obs[3], self.normal_limits[7])
                sample[8] = np.clip(sample[8], self.normal_limits[8], min_obs[4])
                sample[9] = np.clip(sample[9], max_obs[4], self.normal_limits[9])
                sample[10] = np.clip(sample[10], self.normal_limits[10], min_obs[5])
                sample[11] = np.clip(sample[11], max_obs[5], self.normal_limits[11])
                sample[12] = np.clip(sample[12], self.normal_limits[12], min_obs[6])
                sample[13] = np.clip(sample[13], max_obs[6], self.normal_limits[13])
        
        return sample_bank

    def _get_random_sample_bank(self):
        """Returns a random sample bank of size self.sample_bank"""

        samples = []
        s_1 = np.linspace(self.normal_limits[0], self.normal_limits[1], 5).tolist()
        s_2 = np.linspace(self.normal_limits[2], self.normal_limits[3], 5).tolist()
        s_3 = np.linspace(self.normal_limits[4], self.normal_limits[5], 5).tolist()
        s_4 = np.linspace(self.normal_limits[6], self.normal_limits[7], 5).tolist()
        s_5 = np.linspace(self.normal_limits[8], self.normal_limits[9], 5).tolist()
        s_6 = np.linspace(self.normal_limits[10], self.normal_limits[11], 5).tolist()
        s_7 = np.linspace(self.normal_limits[12], self.normal_limits[13], 5).tolist()

        c_1 = [s for s in s_1 if s <= self.start_state[0]]
        c_2 = [s for s in s_1 if s > self.start_state[0]]
        c_3 = [s for s in s_2 if s <= self.start_state[1]]
        c_4 = [s for s in s_2 if s > self.start_state[1]]
        c_5 = [s for s in s_3 if s <= self.start_state[2]]
        c_6 = [s for s in s_3 if s > self.start_state[2]]
        c_7 = [s for s in s_4 if s <= self.start_state[3]]
        c_8 = [s for s in s_4 if s > self.start_state[3]]
        c_9 = [s for s in s_5 if s <= self.start_state[4]]
        c_10 = [s for s in s_5 if s > self.start_state[4]]
        c_11 = [s for s in s_6 if s <= self.start_state[5]]
        c_12 = [s for s in s_6 if s > self.start_state[5]]
        c_13 = [s for s in s_7 if s <= self.start_state[6]]
        c_14 = [s for s in s_7 if s > self.start_state[6]]

        for i in prd(c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14):
            samples.append(i)
        sample_bank = random.sample(samples, self.sample_bank)
        return sample_bank
    
    def _rejection_sampling(self, sample, state, start, joint_traj, cartesian_traj, goal, flag):
        limits = np.concatenate((np.array([-.0523, 1.309, -0.61, 0.61]), np.array(sample))).reshape(9,2)
        sim = Env(limits, headless=True)
        sim.set_to_state(start, state, None)
        cartesian_trajectory, joint_trajectory, _ = sim.simulate_p2p(start, self.env.forward_kinematics(goal))
        
        if cartesian_trajectory is None and joint_trajectory is None:
            return (None, 1000)
        
        if flag == 1:
            loss = fastdtw.fastdtw(np.array(joint_trajectory), np.array(joint_traj), dist=5)[0]
        else:
            loss = fastdtw.fastdtw(np.array(cartesian_trajectory), np.array(cartesian_traj), dist=5)[0]

        if loss < self.threshold and flag == 0:
            del sim
            print("Accepted", loss, flag, sample) if self.DEBUG else None
            return (sample, loss)
        elif loss < self.c_threshold and flag == 1:
            del sim
            print("Accepted", loss, flag, sample) if self.DEBUG else None
            return (sample, loss)
        else:
            del sim
            print("Rejected", loss, flag, sample) if self.DEBUG else None
            return (None, loss)
        
    def _do_kde_calculations(self, flag: int):
        """Returns the maximum likelihood constraints from the possible constraints so far by running through the list of all accepted constraint combinations and finding the one with the maximum sum of log likelihoods
        :return: The maximum likelihood constraints"""
        #find max likelihood constraints
        kde_scores = np.zeros(len(self.possible_constraints))
        for i in range(len(self.j_kde_collection if flag == 0 else self.c_kde_collection)):
            kde_14D = self.j_kde_collection[i] if flag == 0 else self.c_kde_collection[i]
            score = kde_14D.score_samples(np.array(self.possible_constraints))
            kde_scores = kde_scores + np.array(score)

        max_likelihood_index = np.argmax(kde_scores)
        mlc_14D = self.possible_constraints[max_likelihood_index]
        return mlc_14D
    
    def _biased_sampling(self):
        """Sample values between lower_limit and upper_limit with bias towards the lower limit.
        :returns: list of samples of len self.sample_bank"""
        lims = np.array(self.normal_limits).reshape(7,2)

        #for each row in the limits:
        biased_samples = []

        for i, row in enumerate(lims):
             #rescale all samples in the i-th and i+1-th column to be between lower limit and start state, and start state and upper limit respectively
            lower_limit = row[0]
            upper_limit = row[1]
            ss = self.start_state[i]
            #generate samples from exponential distribution clipped between 0 and 1
            exponential_samples_ll = np.random.exponential(scale=2.5, size=self.sample_bank)
            exponential_samples_ul = np.random.exponential(scale=2.5, size=self.sample_bank)

            scaled_samples_ll = exponential_samples_ll / (exponential_samples_ll.max() / (ss - lower_limit))
            biased_samples_ll = lower_limit + scaled_samples_ll

            scaled_samples_ul = exponential_samples_ul / (exponential_samples_ul.max() / (upper_limit - ss))
            biased_samples_ul = upper_limit - scaled_samples_ul
            
            biased_samples.append(biased_samples_ll)
            biased_samples.append(biased_samples_ul)

        biased_samples = np.array(biased_samples).T

        #convert to list of lists
        biased_samples = biased_samples.tolist()

        return biased_samples
    
    def _strong_prior_sampling(self):

        lims = np.array(self.normal_limits[:8]).reshape(4,2)

        #for each row in the limits:
        biased_samples = []

        for i, row in enumerate(lims):
            #rescale all samples in the i-th and i+1-th column to be between lower limit and start state, and start state and upper limit respectively
            lower_limit = row[0]
            upper_limit = row[1]
            ss = self.start_state[i]
            #generate samples from exponential distribution clipped between 0 and 1
            exponential_samples_ll = np.random.uniform(size=self.sample_bank)
            exponential_samples_ul = np.random.uniform(size=self.sample_bank)

            scaled_samples_ll = exponential_samples_ll / (exponential_samples_ll.max() / (ss - lower_limit))
            biased_samples_ll = lower_limit + scaled_samples_ll

            scaled_samples_ul = exponential_samples_ul / (exponential_samples_ul.max() / (upper_limit - ss))
            biased_samples_ul = upper_limit - scaled_samples_ul
            
            biased_samples.append(biased_samples_ll)
            biased_samples.append(biased_samples_ul)

        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[6])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[7])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[8])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[9])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[10])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[11])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[12])
        biased_samples.append(np.ones(self.sample_bank) * self.normal_limits[13])
    
        biased_samples = np.array(biased_samples).T

        #convert to list of lists
        biased_samples = biased_samples.tolist()

        return biased_samples
        
    def _latin_hypercube_sampling(self):
        ll = []
        ul = []
        for i in range(0, 14, 2):
            ll.append(self.normal_limits[i])
            ul.append(self.start_state[i//2])
            ll.append(self.start_state[i//2])
            ul.append(self.normal_limits[i+1])

        sampler = LatinHypercube(d=14)
        samples = sampler.random(n=self.sample_bank)
        scaled_samples = scale(samples, ll, ul)

        return scaled_samples
    
    def _lhs_strong_prior(self):
        ll = []
        ul = []
        for i in range(0, 8, 2):
            ll.append(self.normal_limits[i])
            ul.append(self.start_state[i//2])
            ll.append(self.start_state[i//2])
            ul.append(self.normal_limits[i+1])

        sampler = LatinHypercube(d=8)
        samples = sampler.random(n=self.sample_bank)
        scaled_samples = scale(samples, ll, ul)
        samples_concat = np.concatenate((scaled_samples, np.ones((self.sample_bank, 6)) * self.normal_limits[8:]), axis=1)

        return samples_concat
    
    def _dummy_sampling(self):
        blocked_shoulder = [.8, 1.52,
                            -np.pi/2, np.pi/2,
                            -.785, np.pi,
                            -np.pi/2, np.pi/2,
                            0, 2.53,
                            -0.873, 1.047,
                            -0.524, 0.349]
        return [blocked_shoulder for i in range(self.sample_bank)]
    
    def _get_min_max(self):
        """Returns the minimum and maximum values in the sample bank"""
        min = np.min(np.array(self.observations), axis=0)
        max = np.max(np.array(self.observations), axis=0)
        return min, max
    
if __name__ == "__main__":

    learner = Learner()
    s = learner._strong_prior_sampling()
    print(len(s))
    print(len(s[0]))
    print(s[0])

    bs = learner._biased_sampling()
    #max and min in each column and limit to 2 decimal places
    print(np.min(np.array(bs), axis=0).round(2))
    print(np.max(np.array(bs), axis=0).round(2))
    print(np.mean(np.array(bs), axis=0).round(2))

    lh = learner._latin_hypercube_sampling()
    print("Latin Hypercube")
    print(lh[0])
    print(np.min(np.array(lh), axis=0).round(2))
    print(np.max(np.array(lh), axis=0).round(2))
    print(np.mean(np.array(lh), axis=0).round(2))
    print(len(lh))

    print("LHS Strong Prior")
    lhs = learner._lhs_strong_prior()
    print(lhs[0])
    print(np.min(np.array(lhs), axis=0).round(2))
    print(np.max(np.array(lhs), axis=0).round(2))
    print(np.mean(np.array(lhs), axis=0).round(2))
    print(len(lhs))

    
    