#!/usr/bin/env python

"""
UCBerkeley.DeepRL
Homework1.BC+Dagger
rrs@numericcal

Example usage:

    python rrs.py experts/Humanoid-v1.pkl Humanoid-v1 --render --num_rollouts 5 --dagger_rounds 50

Author of the original script and included expert policies: Jonathan Ho (hoj@openai.com)

Environment setup instruction and problem statement: http://rll.berkeley.edu/deeprlcourse/f17docs/hw1fall2017.pdf
Class website: http://rll.berkeley.edu/deeprlcourse/
Lectures for this exercise: Aug 23 / Aug 28
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

import math
import random as rnd
import matplotlib.pyplot as plt

# for dropping into the repl
import IPython as ipy

def runExpert(args):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:]).flatten()
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action) # third is done
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        return expert_data
    
def queryExpert(args, obs_samples):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        returns = []
        observations = list(obs_samples)
        actions = []
        for obs in obs_samples:
            action = policy_fn(obs[None,:]).flatten()
            actions.append(action)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        return expert_data
 
def runClone(sess, args, net):

    (g, top, ref, loss, cnet_in, cnet_out) = net


    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            feed = {cnet_in: obs[None,:]}
            a = sess.run([cnet_out], feed_dict = feed)

            action = a[0].flatten()
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action) # third is done
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    clone_data = {'observations': np.array(observations),
                  'actions': np.array(actions)}

    return clone_data

def parseInput():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--dagger_rounds', type=int, default=10)
    return parser.parse_args()

###########################################################################
##
## network construction
##

def activation_size(tsr):
    s = tsr.shape
    if (len(s) != 2):
        raise ValueError("Expecting a 2D activation vector, BxN.")
    else:
        return int(s[1])

def mk_fcl(g, name, input, out_size):
    """Expects only simple vectors as input layers, not multi-dim tensors."""
    with g.as_default():
        with tf.name_scope(name):
            ws = tf.Variable(
            tf.truncated_normal([activation_size(input), out_size],
                                stddev=1.0/math.sqrt(float(activation_size(input)))),
            name='weights')
            bs = tf.Variable(tf.constant(1e-4,shape=[out_size]), name='biases')

            return tf.matmul(input, ws) + bs

def nn(g, dim_lst):
    """Create a stack of FC layers. Not sure what else we could do
       give that we're not told anything about the structure of 
       observation data.

       :param dim_lst: a list of activation vector dimensions
    """

    with g.as_default():
        lyr = tf.placeholder(tf.float32, shape=(None, dim_lst[0]))
        inp = lyr

        i = 1
        for lyr_dim in dim_lst[1:]:
            lyr = mk_fcl(g, 'clone_full' + str(i), lyr, lyr_dim)
            if (i < len(dim_lst)-1):
                lyr = tf.nn.tanh(lyr)
            i += 1

        return (inp, lyr)

def add_loss(g, nn_out):

    with g.as_default(): 
        size = activation_size(nn_out)

        ref = tf.placeholder(tf.float32, shape=(None, size))

        loss = tf.losses.mean_squared_error(labels=ref, predictions=nn_out)

        return (ref, loss)

def add_train(g, loss, learning_rate=5e-6, lr_step=500000, lr_drop=0.2):

    with g.as_default():
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        starter_learning_rate = learning_rate
        lr = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        lr_step, lr_drop, staircase=True)
        tf.summary.scalar('lr', lr)
        # Passing global_step to minimize() will increment it at each step.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

###########################################################################
##
## training
##

def cloner(net_ops):
    (top, ref, loss, cnet_in, cnet_out) = net_ops

    def p_fn(inp):
        with tf.Session() as sess:
            feed = {cnet_in: inp}
            pred = sess.run([cnet_out], feed_dict = feed)
            return pred
    #policy_fn = tf_util.function([cnet_in], cnet_out)
    return p_fn

def split_shuffle(data, valid_frac):
    o = data['observations']
    a = data['actions']

    limit = int(valid_frac * len(o))

    o_valid = o[:limit]
    a_valid = a[:limit]

    o_rest = o[limit:]
    a_rest = a[limit:]

    idx = list(range(len(o_rest))) # should be len(o) = len(a)!
    rnd.shuffle(idx)

    return ((o_valid, a_valid), (o[idx], a[idx]))

def train(sess, data, steps, net, valid_frac=0.1, batch=256):

    (g, top, ref, loss, cnet_in, cnet_out) = net

    oa_valid, (o_train, a_train) = split_shuffle(data, valid_frac)

    print("training {} epochs".format(steps*batch / len(o_train)))

    for cnt in range(steps):

        offset = (cnt*batch) % (a_train.shape[0] - batch)
        obs = o_train[offset:offset+batch]
        act = a_train[offset:offset+batch]
        feed = {cnet_in: obs, ref: act}

        _, m, l = sess.run([top, merged, loss], feed_dict = feed)

        if (cnt % 2500 == 0):
            print("[{}] loss: {}".format(cnt, l))
            train_writer.add_summary(m, cnt)

###########################################################################
##
## command line
##

def merge(d1, d2):
    return {'observations': np.concatenate((d1['observations'], d2['observations'])),
            'actions': np.concatenate((d1['actions'], d2['actions']))}

if __name__ == '__main__':
    args = parseInput()
    expert_data = runExpert(args)

    # define the network
    o_size = expert_data['observations'][0].size
    a_size = expert_data['actions'][0].size

    g = tf.Graph()

    (cnet_in, cnet_out) = nn(g, [o_size,128,64,a_size])
    (ref, loss) = add_loss(g, cnet_out)
    top = add_train(g, loss)

    net = (g, top, ref, loss, cnet_in, cnet_out)

    with tf.Session(graph=g) as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        tf.initialize_all_variables().run() # don't reset the weights on each training

        # cloning
        # NOTE: at this point, despite excellent matching of functions (low loss)
        #       the system does not work ... simplation exits early as the robot
        #       falls over
        train(sess, expert_data, 100000, net)
        clone_data = runClone(sess, args, net)
    
        # dagger
        # LECTURE: https://youtu.be/C_LGsoe36I8?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=1653
        # NOTE: you can watch how the robot gets better through dagger
        #         - it tries and fails, but records the actual (under clone policy) state samples
        #         - the expert policy provides instruction on what SHOULD HAVE BEEN done in those states
        #         - we retrain the clone policy using the augmented set of expert samples

        dagger_data = merge(expert_data, clone_data)
        
        for itr in range(args.dagger_rounds):
            clone_obs = clone_data['observations']              # the observed trajectory using clone policy
            print("[DAGGER] adding {} samples".format(len(clone_obs)))
            query_data = queryExpert(args, clone_obs)    # step 3 in the lecture
            dagger_data = merge(expert_data, query_data) # step 4 in the lecture
        
            train(sess, dagger_data, 50000, net)         # step 1' (step 1 from the lecture before the loop)
            clone_data = runClone(sess, args, net)       # step 2 in the lecture


