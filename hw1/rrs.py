#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

import math
import random as rnd

# for dropping into the repl
import IPython as ipy

def exerciseExpert(args):
    print('loading and building expert policy')
    expert_policy = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    return exercisePolicy(args, expert_policy)


# TODO: split this into
#         exercisePolicy: [observation] -> [action]
#         genPath: () -> ([observation], [action])
def exercisePolicy(args, policy_fn):
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
                obs, r, done, _ = env.step(action)
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
    

def parseInput():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    return parser.parse_args()

def cloner(expert_data):
    raise NotImplementedError

def dagger():
    raise NotImplementedError

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

def mk_fcl(name, input, out_size):
    """Expects only simple vectors as input layers, not multi-dim tensors."""
    with tf.name_scope(name):
        ws = tf.Variable(
          tf.truncated_normal([activation_size(input), out_size],
                              stddev=1.0/math.sqrt(float(activation_size(input)))),
          name='weights')
        bs = tf.Variable(tf.zeros([out_size]), name='biases')

        return tf.matmul(input, ws) + bs

def nn(dim_lst):
    """Create a stack of FC layers. Not sure what else we could do
       give that we're not told anything about the structure of 
       observation data.

       :param dim_lst: a list of activation vector dimensions
    """

    lyr = tf.placeholder(tf.float32, shape=(None, dim_lst[0]))
    inp = lyr

    i = 1
    for lyr_dim in dim_lst[1:]:
        lyr = mk_fcl('full' + str(i), lyr, lyr_dim)
        lyr = tf.nn.relu(lyr)
        i += 1

    return (inp, lyr)

def add_loss(nn_out):
    size = activation_size(nn_out)

    ref = tf.placeholder(tf.float32, shape=(None, size))

    loss = tf.losses.mean_squared_error(labels=ref, predictions=nn_out)

    return (ref, loss)

def add_train(loss, learning_rate, lr_step=100000, lr_drop=0.2):

    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = learning_rate
    lr = tf.train.exponential_decay(starter_learning_rate, global_step,
                                    lr_step, lr_drop, staircase=True)
    tf.summary.scalar('lr', lr)
    # Passing global_step to minimize() will increment it at each step.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
    
###########################################################################
##
## training
##


def split_shuffle(data, valid_frac):
    o = data['observations']
    a = data['actions']

    idx = list(range(len(o))) # should be len(o) = len(a)!
    rnd.shuffle(idx)

    brake = int(valid_frac * len(idx))

    valid_idx = idx[:brake]
    train_idx = idx[brake:]

    return ((o[valid_idx], a[valid_idx]), (o[train_idx], a[train_idx]))

def validate(net, valid):
    raise NotImplementedError
    
def train(data, steps, net_ops, valid_frac=0.1, batch=64):

    (top, ref, loss, cnet_in, cnet_out) = net_ops

    oa_valid, (o_train, a_train) = split_shuffle(data, valid_frac)

    print("training {} epochs".format(steps*batch / len(o_train)))

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        tf.initialize_all_variables().run()

        for cnt in range(steps):

            offset = (cnt*batch) % (a_train.shape[0] - batch)
            obs = o_train[offset:offset+batch]
            act = a_train[offset:offset+batch]
            feed = {cnet_in: obs, ref: act}

            _, m, l = sess.run([top, merged, loss], feed_dict = feed)

            if (cnt % 250 == 0):
                print("[{}] loss: {}".format(cnt, l))
                train_writer.add_summary(m, cnt)

        # validate
            

def cloner(net_ops):
    (top, ref, loss, cnet_in, cnet_out) = net_ops

    policy_fn = tf_util.function([cnet_in], cnet_out)

    return policy_fn

        # is dynamic context going to work here?
        
    
###########################################################################
##
## command line
##

if __name__ == '__main__':
    args = parseInput()
    expert_data = exerciseExpert(args)

    #ipy.embed()

    # define the network
    o_size = expert_data['observations'][0].size
    a_size = expert_data['actions'][0].size

    (cnet_in, cnet_out) = nn([o_size,512,1024,512,a_size])
    (ref, loss) = add_loss(cnet_out)
    top = add_train(loss, 0.01)

    net_ops = (top, ref, loss, cnet_in, cnet_out)

    # cloning
    train(expert_data, 500000, net_ops)

    clone_policy = cloner(net_ops)

    clone_data = exercisePolicy(args, clone_policy)

    # test execution

    # dagger
