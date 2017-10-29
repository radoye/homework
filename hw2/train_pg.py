import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#

def report(tag, tsr):
    print("[{}] {} : {}".format(tag, tsr.name, tsr.get_shape()))

def flip(arr):
    return arr[::-1] # not gonna spend time on silly utils

def discounts(n, gamma=1.0):
    return np.power(gamma, range(n))

def ret(rs, gamma=1.0):
    vals = [[np.dot(r, discounts(len(r), gamma))]*len(r) for r in rs]
    return np.array(vals).flatten()

def rtg(rs, gamma=1.0):
    vals = np.array([])
    for r in rs:
        vals = np.append(vals, flip(np.cumsum(flip(np.multiply(r, discounts(len(r), gamma))))))
    return np.array(vals).flatten()
    

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #
    # NOTE: interestingly, nothing about the return
    #       probably the output tensor (judging from the way this is used further down)
    #========================================================================================#

    if (n_layers == 0):
        return tf.contrib.layers.fully_connected(input_placeholder, output_size, activation_fn=output_activation)
    else:
        lyr = tf.contrib.layers.fully_connected(input_placeholder, size, activation_fn=activation)
        return build_mlp(lyr,
                         output_size,
                         scope,
                         n_layers-1,
                         size, activation, output_activation)

            
def pathlength(path):
    return len(path["reward"])


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):

    print("[INFO] max_path_length: {}".format(max_path_length))

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # hack according to https://github.com/openai/gym/issues/499
    env._max_episode_steps = max_path_length

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        # WARN: why is it not [None, ac_dim]? this violates the convention _na above!
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)


    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        # the code below is used in policy update
        
        # (1)
        # our network proposes log probabilities
        sy_logits_na = build_mlp(sy_ob_no, ac_dim, "policy_d", size=32, n_layers=3)

        # (3)
        # now we must calculate based on what the sim actually performed
        # as suggesting an action does not mean it will get performed 
        #
        # WARN: we implicitly assume that when sy_logprob_n is used, sy_logits_na is called with
        #       the appropriatelly sized input (essentially with all the actions in the rollout)
        sy_idx_na = tf.one_hot(sy_ac_na, depth=ac_dim, dtype=tf.float32, on_value=1.0, off_value=0.0)
        sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits(labels=sy_idx_na, logits=sy_logits_na) 
        # alternative
        # tf.nn.sparse_softmax_cross_entropy_with_logits

        # the code below is used in simulation stepping

        # (2)
        # we select a control from the play probabilities
        # Hint: Use the tf.multinomial op
        #
        # WARN: implicitly we assume that when this is called the sy_logits_na is called with n = 1
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, num_samples=1), [1]) 


        print("discrete case [ac_dim = {}]".format(ac_dim))
        report("sy_logits_na", sy_logits_na)
        report("sy_idx_na", sy_idx_na)
        report("sy_logprob_n", sy_logprob_n)
        report("sy_sampled_ac", sy_sampled_ac)


    else:
        # YOUR_CODE_HERE
        # 20th minute in the lecture:
        # https://www.youtube.com/watch?v=tWNpiNzWuO8&index=4&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3

        sy_mean = build_mlp(sy_ob_no, ac_dim, "policy_c", size=64, n_layers=2, activation=tf.nn.tanh)
        # logstd should just be a trainable variable not a network output.
        sy_logstd = tf.get_variable("sy_logstd", shape=[], dtype=tf.float32)
        sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal([ac_dim], 0.0, 1.0)
        # Hint: Use the log probability under a multivariate gaussian. 
        sy_diff_n = sy_mean - sy_ac_na
        sy_logprob_n = 0.5*tf.exp(-sy_logstd) * tf.squeeze(tf.square(sy_diff_n))

        report("sy_mean", sy_mean)
        report("sy_sampled_ac", sy_sampled_ac)
        report("sy_diff_n", sy_diff_n)
        report("sy_logprob_n", sy_logprob_n)
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #
    # NOTE: it's super confusing to have this here ...
    #       given that q_n and adv_n is calculated after the rollouts!
    #       that's where the _adv_ stuff comes from!
    #========================================================================================#

    # Loss function that we'll differentiate to get the policy gradient.
    # e.g. "Example: Gaussian policies" slide
    # http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf
    sy_logp_rsum = tf.multiply(sy_logprob_n, sy_adv_n)
    loss = tf.reduce_mean(sy_logp_rsum) 

    sy_lr = tf.placeholder(dtype=tf.float32, shape=(), name="lr")
    update_op = tf.train.AdamOptimizer(sy_lr).minimize(loss)

    report("loss", loss)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1, 
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        # YOUR_CODE_HERE

        # minute 20 in the lecture
        # https://www.youtube.com/watch?v=PpVhtJn-iZI&t=1223s&index=5&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3
        baseline_update_op = TODO

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):

        # a bit of lr scheduling to test
        if (itr < 15):
            current_lr = learning_rate
        elif (itr < 30):
            current_lr = learning_rate / 2.0
        elif (itr < 50):
            current_lr = learning_rate / 4.0
        elif (itr <= 75):
            current_lr = learning_rate / 8.0
        else:
            current_lr = learning_rate / 16.0

            
        print("********** Iteration %i ************"%itr)

        # NOTE: This (the whole thing, actually) is a pretty messy.
        #       I'm also lazy to check stuff that should be obvious.
        #       Assuming that this training loop is STEPPING through
        #       the simulation by sending 'sy_ob_no' imputs one at
        #       a time, getting results from the policy + sampling
        #       and then applying that by extracting the number with
        #       ac[0] ... something along those lines
        
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = ac[0] 
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#

        # YOUR_CODE_HERE

        # this is a numeric (numpy) computation (no sym_ prefix)
        if not reward_to_go:
            print(">>>>> reward_to_go == False")
            q_n = ret([p["reward"] for p in paths], gamma)

        else:
            print(">>>>> reward_to_go == True")
            q_n = rtg([p["reward"] for p in paths], gamma)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = TODO
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            pass

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE

        # finally we can train the policy network
        feed = {sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n, sy_lr: current_lr}
        old_loss = sess.run([loss], feed_dict = feed)[0]
        _, new_loss = sess.run([update_op, loss], feed_dict = feed)

        # debugging
        #adv, diff, logprob = sess.run([sy_adv_n, sy_diff_n, sy_logprob_n], feed_dict = feed)


        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("Loss BEFORE", old_loss)
        logz.log_tabular("Loss AFTER", new_loss)
        logz.log_tabular("Max EpLen", max_path_length)
        logz.log_tabular("learning rate", current_lr)
        logz.dump_tabular()
        logz.pickle_tf_vars()
        #print("[DIFF: {}] {}".format(diff.shape, diff))
        #print("[LOGPROB: {}] {}".format(logprob.shape, logprob))
        #print("[ADV: {}] {}".format(adv.shape, adv))
        #print("[XYZ] {}".format(np.mean(np.dot(logprob,adv))))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=-1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    seed = args.seed
    for e in range(args.n_experiments):
        #seed = args.seed + 10*e
        seed = (seed * 1772742 + e * 22774) % 100000
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
