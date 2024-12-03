"""
@author: Mustafa Kütük
"""

import sys

sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, y0, z0, u0, X_f, layers, lb, ub):
        X0 = np.concatenate((x0, y0, z0, 0 * x0 + lb[3]), 1)  # (x0, y0, t0)
        t_test = lb[3] + (ub[3] - lb[3]) * lhs(1, 50)
        t_t2 = np.vstack((lb[3], t_test[:, 0:1]))
        t_t2 = np.vstack((t_t2, ub[3]))
        t_t2 = t_t2.flatten()[:, None]
        x_b_test1 = lb[0] + (ub[0] - lb[0]) * lhs(1, len(t_t2))
        y_b_test1 = lb[1] + (ub[1] - lb[1]) * lhs(1, len(t_t2))
        z_b_test1 = lb[2] + (ub[2] - lb[2]) * lhs(1, len(t_t2))
        x_b_test1 = x_b_test1.flatten()[:, None]
        y_b_test1 = y_b_test1.flatten()[:, None]
        z_b_test1 = z_b_test1.flatten()[:, None]
        left = np.concatenate((0 * y_b_test1 + lb[0], y_b_test1, z_b_test1, t_t2), 1)
        right = np.concatenate((0 * y_b_test1 + ub[0], y_b_test1, z_b_test1, t_t2), 1)
        lower = np.concatenate((x_b_test1, 0 * x_b_test1 + lb[1], z_b_test1, t_t2), 1)
        upper = np.concatenate((x_b_test1, 0 * x_b_test1 + ub[1], z_b_test1, t_t2), 1)
        front = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + lb[2], t_t2), 1)
        rear = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + ub[2], t_t2), 1)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.y0 = X0[:, 1:2]
        self.z0 = X0[:, 2:3]
        self.t0 = X0[:, 3:4]

        self.x_left = left[:, 0:1]
        self.y_left = left[:, 1:2]
        self.z_left = left[:, 2:3]
        self.t_left = left[:, 3:4]

        self.x_right = right[:, 0:1]
        self.y_right = right[:, 1:2]
        self.z_right = right[:, 2:3]
        self.t_right = right[:, 3:4]

        self.x_upper = upper[:, 0:1]
        self.y_upper = upper[:, 1:2]
        self.z_upper = upper[:, 2:3]
        self.t_upper = upper[:, 3:4]

        self.x_lower = lower[:, 0:1]
        self.y_lower = lower[:, 1:2]
        self.z_lower = lower[:, 2:3]
        self.t_lower = lower[:, 3:4]

        self.x_front = front[:, 0:1]
        self.y_front = front[:, 1:2]
        self.z_front = front[:, 2:3]
        self.t_front = front[:, 3:4]

        self.x_rear = rear[:, 0:1]
        self.y_rear = rear[:, 1:2]
        self.z_rear = rear[:, 2:3]
        self.t_rear = rear[:, 3:4]

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.z_f = X_f[:, 2:3]
        self.t_f = X_f[:, 3:4]

        self.x_energy = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((X_f[:, 0:1], left[:, 0:1])), right[:, 0:1])), upper[:, 0:1])), lower[:, 0:1])), front[:, 0:1])), rear[:, 0:1]))
        self.y_energy = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((X_f[:, 1:2], left[:, 1:2])), right[:, 1:2])), upper[:, 1:2])), lower[:, 1:2])), front[:, 1:2])), rear[:, 1:2]))
        self.z_energy = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((X_f[:, 2:3], left[:, 2:3])), right[:, 2:3])), upper[:, 2:3])), lower[:, 2:3])), front[:, 2:3])), rear[:, 2:3]))
        self.t_energy = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((X_f[:, 3:4], left[:, 3:4])), right[:, 3:4])), upper[:, 3:4])), lower[:, 3:4])), front[:, 3:4])), rear[:, 3:4]))

        self.u0 = u0
        lbfgs_iter = 2000
        self.loss_lbfgs_new = np.zeros((2 * lbfgs_iter + 1))
        self.loss_lbfgs = np.zeros((lbfgs_iter + 1))
        self.count = -1

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.z0_tf = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        self.x_left_tf = tf.placeholder(tf.float32, shape=[None, self.x_left.shape[1]])
        self.y_left_tf = tf.placeholder(tf.float32, shape=[None, self.y_left.shape[1]])
        self.z_left_tf = tf.placeholder(tf.float32, shape=[None, self.z_left.shape[1]])
        self.t_left_tf = tf.placeholder(tf.float32, shape=[None, self.t_left.shape[1]])

        self.x_right_tf = tf.placeholder(tf.float32, shape=[None, self.x_right.shape[1]])
        self.y_right_tf = tf.placeholder(tf.float32, shape=[None, self.y_right.shape[1]])
        self.z_right_tf = tf.placeholder(tf.float32, shape=[None, self.z_right.shape[1]])
        self.t_right_tf = tf.placeholder(tf.float32, shape=[None, self.t_right.shape[1]])

        self.x_upper_tf = tf.placeholder(tf.float32, shape=[None, self.x_upper.shape[1]])
        self.y_upper_tf = tf.placeholder(tf.float32, shape=[None, self.y_upper.shape[1]])
        self.z_upper_tf = tf.placeholder(tf.float32, shape=[None, self.z_upper.shape[1]])
        self.t_upper_tf = tf.placeholder(tf.float32, shape=[None, self.t_upper.shape[1]])

        self.x_lower_tf = tf.placeholder(tf.float32, shape=[None, self.x_lower.shape[1]])
        self.y_lower_tf = tf.placeholder(tf.float32, shape=[None, self.y_lower.shape[1]])
        self.z_lower_tf = tf.placeholder(tf.float32, shape=[None, self.z_lower.shape[1]])
        self.t_lower_tf = tf.placeholder(tf.float32, shape=[None, self.t_lower.shape[1]])

        self.x_front_tf = tf.placeholder(tf.float32, shape=[None, self.x_front.shape[1]])
        self.y_front_tf = tf.placeholder(tf.float32, shape=[None, self.y_front.shape[1]])
        self.z_front_tf = tf.placeholder(tf.float32, shape=[None, self.z_front.shape[1]])
        self.t_front_tf = tf.placeholder(tf.float32, shape=[None, self.t_front.shape[1]])

        self.x_rear_tf = tf.placeholder(tf.float32, shape=[None, self.x_rear.shape[1]])
        self.y_rear_tf = tf.placeholder(tf.float32, shape=[None, self.y_rear.shape[1]])
        self.z_rear_tf = tf.placeholder(tf.float32, shape=[None, self.z_rear.shape[1]])
        self.t_rear_tf = tf.placeholder(tf.float32, shape=[None, self.t_rear.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None, self.z_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.x_energy_tf = tf.placeholder(tf.float32, shape=[None, self.x_energy.shape[1]])
        self.y_energy_tf = tf.placeholder(tf.float32, shape=[None, self.y_energy.shape[1]])
        self.z_energy_tf = tf.placeholder(tf.float32, shape=[None, self.z_energy.shape[1]])
        self.t_energy_tf = tf.placeholder(tf.float32, shape=[None, self.t_energy.shape[1]])

    # Initialize NNs

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

    # tf Graphs
        self.u0_pred, _, _, _ = self.net_u(self.x0_tf, self.y0_tf, self.z0_tf, self.t0_tf)
        self.u_left, self.u_x_left_pred, self.u_y_left_pred, self.u_z_left_pred = self.net_u(self.x_left_tf, self.y_left_tf, self.z_left_tf, self.t_left_tf)
        self.u_right, self.u_x_right_pred, self.u_y_right_pred, self.u_z_right_pred = self.net_u(self.x_right_tf, self.y_right_tf, self.z_right_tf, self.t_right_tf)
        self.u_upper, self.u_x_upper_pred, self.u_y_upper_pred, self.u_z_upper_pred = self.net_u(self.x_upper_tf, self.y_upper_tf, self.z_upper_tf, self.t_upper_tf)
        self.u_lower, self.u_x_lower_pred, self.u_y_lower_pred, self.u_z_lower_pred = self.net_u(self.x_lower_tf, self.y_lower_tf, self.z_lower_tf, self.t_lower_tf)
        self.u_front, self.u_x_front_pred, self.u_y_front_pred, self.u_z_front_pred = self.net_u(self.x_front_tf,
                                                                                                 self.y_front_tf,
                                                                                                 self.z_front_tf,
                                                                                                 self.t_front_tf)
        self.u_rear, self.u_x_rear_pred, self.u_y_rear_pred, self.u_z_rear_pred = self.net_u(self.x_rear_tf,
                                                                                                 self.y_rear_tf,
                                                                                                 self.z_rear_tf,
                                                                                                 self.t_rear_tf)
        self.f_u_pred, self.energy_pred, _ = self.net_f_u(self.x_f_tf, self.y_f_tf, self.z_f_tf, self.t_f_tf)
        _, _, self.energy_t_pred = self.net_f_u(self.x_energy_tf, self.y_energy_tf, self.z_energy_tf, self.t_energy_tf)

    #Neumann BCs, wi=10^3
        self.loss = 1000 * tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_left_pred) + tf.square(self.u_x_right_pred)) + \
                    tf.reduce_mean(tf.square(self.u_y_upper_pred) + tf.square(self.u_y_lower_pred)) + \
                    tf.reduce_mean(tf.square(self.u_z_front_pred) + tf.square(self.u_z_rear_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    (self.x_energy.shape[0]) * tf.reduce_mean(tf.square(tf.math.maximum(0.0, self.energy_t_pred)))


    # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': lbfgs_iter,
                                                                         'maxfun': lbfgs_iter,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

    # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def init_update(self, x0, y0, z0, u0, X_f, layers, lb, ub):
        X0 = np.concatenate((x0, y0, z0, 0 * x0 + lb[3]), 1)  # (x0, y0, t0)
        t_test = lb[3] + (ub[3] - lb[3]) * lhs(1, 50)
        t_t2 = np.vstack((lb[3], t_test[:, 0:1]))
        t_t2 = np.vstack((t_t2, ub[3]))
        t_t2 = t_t2.flatten()[:, None]
        x_b_test1 = lb[0] + (ub[0] - lb[0]) * lhs(1, len(t_t2))
        y_b_test1 = lb[1] + (ub[1] - lb[1]) * lhs(1, len(t_t2))
        z_b_test1 = lb[2] + (ub[2] - lb[2]) * lhs(1, len(t_t2))
        x_b_test1 = x_b_test1.flatten()[:, None]
        y_b_test1 = y_b_test1.flatten()[:, None]
        z_b_test1 = z_b_test1.flatten()[:, None]
        left = np.concatenate((0 * y_b_test1 + lb[0], y_b_test1, z_b_test1, t_t2), 1)
        right = np.concatenate((0 * y_b_test1 + ub[0], y_b_test1, z_b_test1, t_t2), 1)
        lower = np.concatenate((x_b_test1, 0 * x_b_test1 + lb[1], z_b_test1, t_t2), 1)
        upper = np.concatenate((x_b_test1, 0 * x_b_test1 + ub[1], z_b_test1, t_t2), 1)
        front = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + lb[2], t_t2), 1)
        rear = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + ub[2], t_t2), 1)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.y0 = X0[:, 1:2]
        self.z0 = X0[:, 2:3]
        self.t0 = X0[:, 3:4]

        self.x_left = left[:, 0:1]
        self.y_left = left[:, 1:2]
        self.z_left = left[:, 2:3]
        self.t_left = left[:, 3:4]

        self.x_right = right[:, 0:1]
        self.y_right = right[:, 1:2]
        self.z_right = right[:, 2:3]
        self.t_right = right[:, 3:4]

        self.x_upper = upper[:, 0:1]
        self.y_upper = upper[:, 1:2]
        self.z_upper = upper[:, 2:3]
        self.t_upper = upper[:, 3:4]

        self.x_lower = lower[:, 0:1]
        self.y_lower = lower[:, 1:2]
        self.z_lower = lower[:, 2:3]
        self.t_lower = lower[:, 3:4]

        self.x_front = front[:, 0:1]
        self.y_front = front[:, 1:2]
        self.z_front = front[:, 2:3]
        self.t_front = front[:, 3:4]

        self.x_rear = rear[:, 0:1]
        self.y_rear = rear[:, 1:2]
        self.z_rear = rear[:, 2:3]
        self.t_rear = rear[:, 3:4]

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.z_f = X_f[:, 2:3]
        self.t_f = X_f[:, 3:4]

        self.x_energy = np.vstack((np.vstack((np.vstack((np.vstack(
            (np.vstack((np.vstack((X_f[:, 0:1], left[:, 0:1])), right[:, 0:1])), upper[:, 0:1])), lower[:, 0:1])),
                                              front[:, 0:1])), rear[:, 0:1]))
        self.y_energy = np.vstack((np.vstack((np.vstack((np.vstack(
            (np.vstack((np.vstack((X_f[:, 1:2], left[:, 1:2])), right[:, 1:2])), upper[:, 1:2])), lower[:, 1:2])),
                                              front[:, 1:2])), rear[:, 1:2]))
        self.z_energy = np.vstack((np.vstack((np.vstack((np.vstack(
            (np.vstack((np.vstack((X_f[:, 2:3], left[:, 2:3])), right[:, 2:3])), upper[:, 2:3])), lower[:, 2:3])),
                                              front[:, 2:3])), rear[:, 2:3]))
        self.t_energy = np.vstack((np.vstack((np.vstack((np.vstack(
            (np.vstack((np.vstack((X_f[:, 3:4], left[:, 3:4])), right[:, 3:4])), upper[:, 3:4])), lower[:, 3:4])),
                                              front[:, 3:4])), rear[:, 3:4]))

        self.u0 = u0
        lbfgs_iter = 2000
        self.loss_lbfgs_new = np.zeros((2 * lbfgs_iter + 1))
        self.loss_lbfgs = np.zeros((lbfgs_iter + 1))
        self.count = -1

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.z0_tf = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        self.x_left_tf = tf.placeholder(tf.float32, shape=[None, self.x_left.shape[1]])
        self.y_left_tf = tf.placeholder(tf.float32, shape=[None, self.y_left.shape[1]])
        self.z_left_tf = tf.placeholder(tf.float32, shape=[None, self.z_left.shape[1]])
        self.t_left_tf = tf.placeholder(tf.float32, shape=[None, self.t_left.shape[1]])

        self.x_right_tf = tf.placeholder(tf.float32, shape=[None, self.x_right.shape[1]])
        self.y_right_tf = tf.placeholder(tf.float32, shape=[None, self.y_right.shape[1]])
        self.z_right_tf = tf.placeholder(tf.float32, shape=[None, self.z_right.shape[1]])
        self.t_right_tf = tf.placeholder(tf.float32, shape=[None, self.t_right.shape[1]])

        self.x_upper_tf = tf.placeholder(tf.float32, shape=[None, self.x_upper.shape[1]])
        self.y_upper_tf = tf.placeholder(tf.float32, shape=[None, self.y_upper.shape[1]])
        self.z_upper_tf = tf.placeholder(tf.float32, shape=[None, self.z_upper.shape[1]])
        self.t_upper_tf = tf.placeholder(tf.float32, shape=[None, self.t_upper.shape[1]])

        self.x_lower_tf = tf.placeholder(tf.float32, shape=[None, self.x_lower.shape[1]])
        self.y_lower_tf = tf.placeholder(tf.float32, shape=[None, self.y_lower.shape[1]])
        self.z_lower_tf = tf.placeholder(tf.float32, shape=[None, self.z_lower.shape[1]])
        self.t_lower_tf = tf.placeholder(tf.float32, shape=[None, self.t_lower.shape[1]])

        self.x_front_tf = tf.placeholder(tf.float32, shape=[None, self.x_front.shape[1]])
        self.y_front_tf = tf.placeholder(tf.float32, shape=[None, self.y_front.shape[1]])
        self.z_front_tf = tf.placeholder(tf.float32, shape=[None, self.z_front.shape[1]])
        self.t_front_tf = tf.placeholder(tf.float32, shape=[None, self.t_front.shape[1]])

        self.x_rear_tf = tf.placeholder(tf.float32, shape=[None, self.x_rear.shape[1]])
        self.y_rear_tf = tf.placeholder(tf.float32, shape=[None, self.y_rear.shape[1]])
        self.z_rear_tf = tf.placeholder(tf.float32, shape=[None, self.z_rear.shape[1]])
        self.t_rear_tf = tf.placeholder(tf.float32, shape=[None, self.t_rear.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None, self.z_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.x_energy_tf = tf.placeholder(tf.float32, shape=[None, self.x_energy.shape[1]])
        self.y_energy_tf = tf.placeholder(tf.float32, shape=[None, self.y_energy.shape[1]])
        self.z_energy_tf = tf.placeholder(tf.float32, shape=[None, self.z_energy.shape[1]])
        self.t_energy_tf = tf.placeholder(tf.float32, shape=[None, self.t_energy.shape[1]])

        # tf Graphs
        self.u0_pred, _, _, _ = self.net_u(self.x0_tf, self.y0_tf, self.z0_tf, self.t0_tf)
        self.u_left, self.u_x_left_pred, self.u_y_left_pred, self.u_z_left_pred = self.net_u(self.x_left_tf,
                                                                                             self.y_left_tf,
                                                                                             self.z_left_tf,
                                                                                             self.t_left_tf)
        self.u_right, self.u_x_right_pred, self.u_y_right_pred, self.u_z_right_pred = self.net_u(self.x_right_tf,
                                                                                                 self.y_right_tf,
                                                                                                 self.z_right_tf,
                                                                                                 self.t_right_tf)
        self.u_upper, self.u_x_upper_pred, self.u_y_upper_pred, self.u_z_upper_pred = self.net_u(self.x_upper_tf,
                                                                                                 self.y_upper_tf,
                                                                                                 self.z_upper_tf,
                                                                                                 self.t_upper_tf)
        self.u_lower, self.u_x_lower_pred, self.u_y_lower_pred, self.u_z_lower_pred = self.net_u(self.x_lower_tf,
                                                                                                 self.y_lower_tf,
                                                                                                 self.z_lower_tf,
                                                                                                 self.t_lower_tf)
        self.u_front, self.u_x_front_pred, self.u_y_front_pred, self.u_z_front_pred = self.net_u(self.x_front_tf,
                                                                                                 self.y_front_tf,
                                                                                                 self.z_front_tf,
                                                                                                 self.t_front_tf)
        self.u_rear, self.u_x_rear_pred, self.u_y_rear_pred, self.u_z_rear_pred = self.net_u(self.x_rear_tf,
                                                                                             self.y_rear_tf,
                                                                                             self.z_rear_tf,
                                                                                             self.t_rear_tf)
        self.f_u_pred, self.energy_pred, _ = self.net_f_u(self.x_f_tf, self.y_f_tf, self.z_f_tf, self.t_f_tf)
        _, _, self.energy_t_pred = self.net_f_u(self.x_energy_tf, self.y_energy_tf, self.z_energy_tf, self.t_energy_tf)

        # Neumann BCs, wi=10^3
        self.loss = 1000 * tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_left_pred) + tf.square(self.u_x_right_pred)) + \
                    tf.reduce_mean(tf.square(self.u_y_upper_pred) + tf.square(self.u_y_lower_pred)) + \
                    tf.reduce_mean(tf.square(self.u_z_front_pred) + tf.square(self.u_z_rear_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    (self.x_energy.shape[0]) * tf.reduce_mean(tf.square(tf.math.maximum(0.0, self.energy_t_pred)))

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': lbfgs_iter,
                                                                         'maxfun': lbfgs_iter,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    ## ResNet
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            # H = tf.tanh(tf.add(tf.matmul(H, W), b))
            if l == 0:
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            else:
                H1 = tf.add(tf.matmul(H, W), b)
                H2 = tf.matmul(H, tf.eye(int(W.shape[0])))
                H = tf.tanh(tf.add(H1, H2))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    ## MLP
    # def neural_net(self, X, weights, biases):
    #     num_layers = len(weights) + 1
    #
    #     H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    #     for l in range(0, num_layers - 2):
    #         W = weights[l]
    #         b = biases[l]
    #         H = tf.tanh(tf.add(tf.matmul(H, W), b))
    #     W = weights[-1]
    #     b = biases[-1]
    #     Y = tf.add(tf.matmul(H, W), b)
    #     return Y

    def net_u(self, x, y, z, t):
        X = tf.concat([x, y, z, t], 1)

        u = self.neural_net(X, self.weights, self.biases)

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]

        return u, u_x, u_y, u_z

    def net_f_u(self, x, y, z, t):
        u, u_x, u_y, u_z = self.net_u(x, y, z, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        f_u = u_t - 0.025 * (u_xx + u_yy + u_zz) + 10 * (u ** 3 - u)

        energy = (1 / self.x_energy.shape[0]) * (np.sum(12.5 * (10 ** (-3)) * ((tf.math.abs(u_x) + tf.math.abs(u_y) + tf.math.abs(u_z)) ** 2)) + (np.sum(2.5 * (1 - u ** 2) ** 2)))

        energy_t = tf.gradients(energy, t)[0]

        return f_u, energy, energy_t

    def callback(self, loss):
        self.count = self.count + 1
        print('Loss:', loss)
        self.lbfgs_loss_history(loss)

    def lbfgs_loss_history(self, loss):
        if self.count > 2000:
            self.loss_lbfgs_new[self.count] = loss
            return self.loss_lbfgs_new
        else:
            self.loss_lbfgs[self.count] = loss
        return self.loss_lbfgs

    def train(self, nIter):

        tf_dict = {self.x0_tf: self.x0, self.y0_tf: self.y0, self.z0_tf: self.z0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,
                   self.x_left_tf: self.x_left, self.y_left_tf: self.y_left, self.z_left_tf: self.z_left,
                   self.t_left_tf: self.t_left,
                   self.x_right_tf: self.x_right, self.y_right_tf: self.y_right, self.z_right_tf: self.z_right,
                   self.t_right_tf: self.t_right,
                   self.x_upper_tf: self.x_upper, self.y_upper_tf: self.y_upper, self.z_upper_tf: self.z_upper,
                   self.t_upper_tf: self.t_upper,
                   self.x_lower_tf: self.x_lower, self.y_lower_tf: self.y_lower, self.z_lower_tf: self.z_lower,
                   self.t_lower_tf: self.t_lower,
                   self.x_front_tf: self.x_front, self.y_front_tf: self.y_front, self.z_front_tf: self.z_front,
                   self.t_front_tf: self.t_front,
                   self.x_rear_tf: self.x_rear, self.y_rear_tf: self.y_rear, self.z_rear_tf: self.z_rear,
                   self.t_rear_tf: self.t_rear,
                   self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.z_f_tf: self.z_f, self.t_f_tf: self.t_f,
                   self.x_energy_tf: self.x_energy, self.y_energy_tf: self.y_energy,
                   self.z_energy_tf: self.z_energy, self.t_energy_tf: self.t_energy}

        start_time = time.time()
        k = 0
        tol = 10 ** (-2)
        a = np.zeros((int(nIter)))
        for it in range(nIter):

            start_time2 = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)

            elapsed = time.time() - start_time2
            loss_value = self.sess.run(self.loss, tf_dict)
            a[k] = loss_value
            print('It: %d, Loss: %.3e, Time: %.2f' %
                  (it, loss_value, elapsed))
            k = k + 1
            if it > 1000 and max(a[k - 1] - a[k - 2], a[k - 2] - a[k - 3], a[k - 3] - a[k - 4]) > tol:
                print("x_f Shape: ", self.x_f.shape, "\n")
                n = int(0.025 * self.x_f.shape[0])
                x_test = self.lb + (self.ub - self.lb) * lhs(4, n)
                _, f_u_pred_test, _ = model.predict(x_test)
                loss_test = np.abs(f_u_pred_test)

                n = int(0.01 * self.x_f.shape[0])

                t_test = self.lb[3] + (self.ub[3] - self.lb[3]) * lhs(1, n)
                t_t2 = t_test[:, 0:1]
                t_t2 = t_t2.flatten()[:, None]
                x_b_test1 = self.lb[0] + (self.ub[0] - self.lb[0]) * lhs(1, len(t_t2))
                y_b_test1 = self.lb[1] + (self.ub[1] - self.lb[1]) * lhs(1, len(t_t2))
                z_b_test1 = self.lb[2] + (self.ub[2] - self.lb[2]) * lhs(1, len(t_t2))
                x_b_test1 = x_b_test1.flatten()[:, None]
                y_b_test1 = y_b_test1.flatten()[:, None]
                z_b_test1 = z_b_test1.flatten()[:, None]
                left_test = np.concatenate((0 * y_b_test1 + lb[0], y_b_test1, z_b_test1, t_t2), 1)
                right_test = np.concatenate((0 * y_b_test1 + ub[0], y_b_test1, z_b_test1, t_t2), 1)
                lower_test = np.concatenate((x_b_test1, 0 * x_b_test1 + lb[1], z_b_test1, t_t2), 1)
                upper_test = np.concatenate((x_b_test1, 0 * x_b_test1 + ub[1], z_b_test1, t_t2), 1)
                front_test = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + lb[2], t_t2), 1)
                rear_test = np.concatenate((x_b_test1, y_b_test1, 0 * z_b_test1 + ub[2], t_t2), 1)


                _, f_u_left_pred_test, _ = model.predict(left_test)
                loss_left_test = np.abs(f_u_left_pred_test)
                _, f_u_right_pred_test, _ = model.predict(right_test)
                loss_right_test = np.abs(f_u_right_pred_test)
                _, f_u_upper_pred_test, _ = model.predict(upper_test)
                loss_upper_test = np.abs(f_u_upper_pred_test)
                _, f_u_lower_pred_test, _ = model.predict(lower_test)
                loss_lower_test = np.abs(f_u_lower_pred_test)
                _, f_u_front_pred_test, _ = model.predict(front_test)
                loss_front_test = np.abs(f_u_front_pred_test)
                _, f_u_rear_pred_test, _ = model.predict(rear_test)
                loss_rear_test = np.abs(f_u_rear_pred_test)


                total_err = sum(loss_test)
                threshold = total_err / 20
                tot_percent = 0
                index = 0
                err_eq_sorted = np.argsort(-loss_test, axis=0)
                while tot_percent < threshold:
                    tot_percent = tot_percent + loss_test[err_eq_sorted][index]
                    index = index + 1

                if index > 2:
                    print("total number of selected points:", index - 1)
                    x_id = np.argsort(-loss_test, axis=0)[:index - 1]
                    print("Adding new training points:", x_test[x_id], "\n")
                    squeezed_X = np.squeeze(x_test[x_id])
                    self.x_f = np.vstack((self.x_f, squeezed_X[:, 0:1]))
                    self.y_f = np.vstack((self.y_f, squeezed_X[:, 1:2]))
                    self.z_f = np.vstack((self.z_f, squeezed_X[:, 2:3]))
                    self.t_f = np.vstack((self.t_f, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_f Shape: ", self.x_f.shape, "\n")
                elif index == 2:
                    print("total number of selected points:", index - 1)
                    x_id = np.argsort(-loss_test, axis=0)[:index - 1]
                    print("Adding new training points:", x_test[x_id], "\n")
                    squeezed_X = np.squeeze(x_test[x_id])
                    self.x_f = np.vstack((self.x_f, squeezed_X[0]))
                    self.y_f = np.vstack((self.y_f, squeezed_X[1]))
                    self.z_f = np.vstack((self.z_f, squeezed_X[2]))
                    self.t_f = np.vstack((self.t_f, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_f Shape: ", self.x_f.shape, "\n")
                else:
                    print("No point is added.", "\n")

                total_err_left = sum(loss_left_test)
                threshold_left = total_err_left / 20
                tot_percent_left = 0
                index_left = 0
                err_eq_sorted_left = np.argsort(-loss_left_test, axis=0)
                while tot_percent_left < threshold_left:
                    tot_percent_left = tot_percent_left + loss_left_test[err_eq_sorted_left][index_left]
                    index_left = index_left + 1

                if index_left > 2:
                    print("total number of selected points:", index_left - 1)
                    x_id = np.argsort(-loss_left_test, axis=0)[:index_left - 1]
                    print("Adding new training points:", left_test[x_id], "\n")
                    squeezed_X = np.squeeze(left_test[x_id])
                    self.x_left = np.vstack((self.x_left, squeezed_X[:, 0:1]))
                    self.y_left = np.vstack((self.y_left, squeezed_X[:, 1:2]))
                    self.z_left = np.vstack((self.z_left, squeezed_X[:, 2:3]))
                    self.t_left = np.vstack((self.t_left, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_left Shape: ", self.x_left.shape, "\n")
                    print("Adding new training points:", right_test[x_id], "\n")
                    squeezed_X = np.squeeze(right_test[x_id])
                    self.x_right = np.vstack((self.x_right, squeezed_X[:, 0:1]))
                    self.y_right = np.vstack((self.y_right, squeezed_X[:, 1:2]))
                    self.z_right = np.vstack((self.z_right, squeezed_X[:, 2:3]))
                    self.t_right = np.vstack((self.t_right, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_right Shape: ", self.x_right.shape, "\n")
                elif index_left == 2:
                    print("total number of selected points:", index_left - 1)
                    x_id = np.argsort(-loss_left_test, axis=0)[:index_left - 1]
                    print("Adding new training points:", left_test[x_id], "\n")
                    squeezed_X = np.squeeze(left_test[x_id])
                    self.x_left = np.vstack((self.x_left, squeezed_X[0]))
                    self.y_left = np.vstack((self.y_left, squeezed_X[1]))
                    self.z_left = np.vstack((self.z_left, squeezed_X[2]))
                    self.t_left = np.vstack((self.t_left, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_left Shape: ", self.x_left.shape, "\n")
                    print("Adding new training points:", right_test[x_id], "\n")
                    squeezed_X = np.squeeze(right_test[x_id])
                    self.x_right = np.vstack((self.x_right, squeezed_X[0]))
                    self.y_right = np.vstack((self.y_right, squeezed_X[1]))
                    self.z_right = np.vstack((self.z_right, squeezed_X[2]))
                    self.t_right = np.vstack((self.t_right, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_right Shape: ", self.x_right.shape, "\n")
                else:
                    print("No point is added.", "\n")


                total_err_upper = sum(loss_upper_test)
                threshold_upper = total_err_upper / 20
                tot_percent_upper = 0
                index_upper = 0
                err_eq_sorted_upper = np.argsort(-loss_upper_test, axis=0)
                while tot_percent_upper < threshold_upper:
                    tot_percent_upper = tot_percent_upper + loss_upper_test[err_eq_sorted_upper][index_upper]
                    index_upper = index_upper + 1

                if index_upper > 2:
                    print("total number of selected points:", index_upper - 1)
                    x_id = np.argsort(-loss_upper_test, axis=0)[:index_upper - 1]
                    print("Adding new training points:", upper_test[x_id], "\n")
                    squeezed_X = np.squeeze(upper_test[x_id])
                    self.x_upper = np.vstack((self.x_upper, squeezed_X[:, 0:1]))
                    self.y_upper = np.vstack((self.y_upper, squeezed_X[:, 1:2]))
                    self.z_upper = np.vstack((self.z_upper, squeezed_X[:, 2:3]))
                    self.t_upper = np.vstack((self.t_upper, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_upper Shape: ", self.x_upper.shape, "\n")
                    print("Adding new training points:", lower_test[x_id], "\n")
                    squeezed_X = np.squeeze(lower_test[x_id])
                    self.x_lower = np.vstack((self.x_lower, squeezed_X[:, 0:1]))
                    self.y_lower = np.vstack((self.y_lower, squeezed_X[:, 1:2]))
                    self.z_lower = np.vstack((self.z_lower, squeezed_X[:, 2:3]))
                    self.t_lower = np.vstack((self.t_lower, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_lower Shape: ", self.x_lower.shape, "\n")
                elif index_upper == 2:
                    print("total number of selected points:", index_upper - 1)
                    x_id = np.argsort(-loss_upper_test, axis=0)[:index_upper - 1]
                    print("Adding new training points:", upper_test[x_id], "\n")
                    squeezed_X = np.squeeze(upper_test[x_id])
                    self.x_upper = np.vstack((self.x_upper, squeezed_X[0]))
                    self.y_upper = np.vstack((self.y_upper, squeezed_X[1]))
                    self.z_upper = np.vstack((self.z_upper, squeezed_X[2]))
                    self.t_upper = np.vstack((self.t_upper, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_upper Shape: ", self.x_upper.shape, "\n")
                    print("Adding new training points:", lower_test[x_id], "\n")
                    squeezed_X = np.squeeze(lower_test[x_id])
                    self.x_lower = np.vstack((self.x_lower, squeezed_X[0]))
                    self.y_lower = np.vstack((self.y_lower, squeezed_X[1]))
                    self.z_lower = np.vstack((self.z_lower, squeezed_X[2]))
                    self.t_lower = np.vstack((self.t_lower, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_lower Shape: ", self.x_lower.shape, "\n")
                else:
                    print("No point is added.", "\n")

                total_err_front = sum(loss_front_test)
                threshold_front = total_err_front / 20
                tot_percent_front = 0
                index_front = 0
                err_eq_sorted_front = np.argsort(-loss_front_test, axis=0)
                while tot_percent_front < threshold_front:
                    tot_percent_front = tot_percent_front + loss_front_test[err_eq_sorted_front][index_front]
                    index_front = index_front + 1

                if index_front > 2:
                    print("total number of selected points:", index_front - 1)
                    x_id = np.argsort(-loss_front_test, axis=0)[:index_front - 1]
                    print("Adding new training points:", front_test[x_id], "\n")
                    squeezed_X = np.squeeze(front_test[x_id])
                    self.x_front = np.vstack((self.x_front, squeezed_X[:, 0:1]))
                    self.y_front = np.vstack((self.y_front, squeezed_X[:, 1:2]))
                    self.z_front = np.vstack((self.z_front, squeezed_X[:, 2:3]))
                    self.t_front = np.vstack((self.t_front, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_front Shape: ", self.x_front.shape, "\n")
                    print("Adding new training points:", rear_test[x_id], "\n")
                    squeezed_X = np.squeeze(rear_test[x_id])
                    self.x_rear = np.vstack((self.x_rear, squeezed_X[:, 0:1]))
                    self.y_rear = np.vstack((self.y_rear, squeezed_X[:, 1:2]))
                    self.z_rear = np.vstack((self.z_rear, squeezed_X[:, 2:3]))
                    self.t_rear = np.vstack((self.t_rear, squeezed_X[:, 3:4]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[:, 1:2]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[:, 2:3]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 3:4]))
                    print("x_rear Shape: ", self.x_rear.shape, "\n")
                elif index_front == 2:
                    print("total number of selected points:", index_front - 1)
                    x_id = np.argsort(-loss_front_test, axis=0)[:index_front - 1]
                    print("Adding new training points:", front_test[x_id], "\n")
                    squeezed_X = np.squeeze(front_test[x_id])
                    self.x_front = np.vstack((self.x_front, squeezed_X[0]))
                    self.y_front = np.vstack((self.y_front, squeezed_X[1]))
                    self.z_front = np.vstack((self.z_front, squeezed_X[2]))
                    self.t_front = np.vstack((self.t_front, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_front Shape: ", self.x_front.shape, "\n")
                    print("Adding new training points:", rear_test[x_id], "\n")
                    squeezed_X = np.squeeze(rear_test[x_id])
                    self.x_rear = np.vstack((self.x_rear, squeezed_X[0]))
                    self.y_rear = np.vstack((self.y_rear, squeezed_X[1]))
                    self.z_rear = np.vstack((self.z_rear, squeezed_X[2]))
                    self.t_rear = np.vstack((self.t_rear, squeezed_X[3]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.y_energy = np.vstack((self.y_energy, squeezed_X[1]))
                    self.z_energy = np.vstack((self.z_energy, squeezed_X[2]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[3]))
                    print("x_rear Shape: ", self.x_rear.shape, "\n")
                else:
                    print("No point is added.", "\n")


            tf_dict = {self.x0_tf: self.x0, self.y0_tf: self.y0, self.z0_tf: self.z0, self.t0_tf: self.t0,
                       self.u0_tf: self.u0,
                       self.x_left_tf: self.x_left, self.y_left_tf: self.y_left, self.z_left_tf: self.z_left,
                       self.t_left_tf: self.t_left,
                       self.x_right_tf: self.x_right, self.y_right_tf: self.y_right, self.z_right_tf: self.z_right,
                       self.t_right_tf: self.t_right,
                       self.x_upper_tf: self.x_upper, self.y_upper_tf: self.y_upper, self.z_upper_tf: self.z_upper,
                       self.t_upper_tf: self.t_upper,
                       self.x_lower_tf: self.x_lower, self.y_lower_tf: self.y_lower, self.z_lower_tf: self.z_lower,
                       self.t_lower_tf: self.t_lower,
                       self.x_front_tf: self.x_front, self.y_front_tf: self.y_front, self.z_front_tf: self.z_front,
                       self.t_front_tf: self.t_front,
                       self.x_rear_tf: self.x_rear, self.y_rear_tf: self.y_rear, self.z_rear_tf: self.z_rear,
                       self.t_rear_tf: self.t_rear,
                       self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.z_f_tf: self.z_f, self.t_f_tf: self.t_f,
                       self.x_energy_tf: self.x_energy, self.y_energy_tf: self.y_energy,
                       self.z_energy_tf: self.z_energy, self.t_energy_tf: self.t_energy}

        elapsed_Adam = time.time() - start_time
        print('Adam Time: %.2f' % (elapsed_Adam))

        start_time_next = time.time()
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        elapsed_LBFGS = time.time() - start_time_next
        print('L-BFGS Time: %.2f' % (elapsed_LBFGS))
        b = self.loss_lbfgs
        return a, b, elapsed_Adam, elapsed_LBFGS, self.x_f, self.y_f, self.z_f, self.t_f, self.x_left, self.y_left, self.z_left, self.t_left, self.x_right, self.y_right, self.z_right, self.t_right, self.x_upper, self.y_upper, self.z_upper, self.t_upper, self.x_lower, self.y_lower, self.z_lower, self.t_lower, self.x_front, self.y_front, self.z_front, self.t_front, self.x_rear, self.y_rear, self.z_rear, self.t_rear

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.y0_tf: X_star[:, 1:2], self.z0_tf: X_star[:, 2:3], self.t0_tf: X_star[:, 3:4]}
        u_star = self.sess.run(self.u0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.y_f_tf: X_star[:, 1:2], self.z_f_tf: X_star[:, 2:3], self.t_f_tf: X_star[:, 3:4]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        energy = self.sess.run(self.energy_pred, tf_dict)

        return u_star, f_u_star, energy

if __name__ == "__main__":
    for i in range(0, 13):
        noise = 0.0

        # Domain bounds
        lb = np.array([0.0, 0.0, 0.0, 0.1 * i])
        ub = np.array([1.0, 1.0, 1.0, 0.1 * (i + 1)])

        N0 = 21*21*21
        N_f = 5000
        layers = [4, 128, 128, 128, 128, 128, 128, 1]

        t = np.linspace(0.1 * i, 0.1 * (i + 1), 11)
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        z = np.linspace(0, 1, 21)
        t = t.flatten()[:, None]
        x = x.flatten()[:, None]
        y = y.flatten()[:, None]
        z = z.flatten()[:, None]

        X_0, Y_0, Z_0 = np.meshgrid(x, y, z)
        X, Y, Z, T = np.meshgrid(x, y, z, t)
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None], T.flatten()[:, None]))

        X_f = lb + (ub - lb) * lhs(4, N_f)

        x0_star = np.hstack((X_0.flatten()[:, None], Y_0.flatten()[:, None], Z_0.flatten()[:, None]))
        x0 = x0_star[:, 0:1]
        y0 = x0_star[:, 1:2]
        z0 = x0_star[:, 2:3]

        if i == 0:
            u0 = np.tanh((0.35 - np.sqrt((x0 - 0.5) ** 2 + (y0 - 0.5) ** 2 + (z0 - 0.5) ** 2)) / 0.1)
            model = PhysicsInformedNN(x0, y0, z0, u0, X_f, layers, lb, ub)
            start_time = time.time()
            train_loss_adam, train_loss_lbfgs, elapsed_adam, elapsed_lbfgs, x_all, y_all, z_all, t_all, x_left_all, y_left_all, z_left_all, t_left_all, x_right_all, y_right_all, z_right_all, t_right_all, x_upper_all, y_upper_all, z_upper_all, t_upper_all, x_lower_all, y_lower_all, z_lower_all, t_lower_all, x_front_all, y_front_all, z_front_all, t_front_all, x_rear_all, y_rear_all, z_rear_all, t_rear_all = model.train(5000)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))
            scipy.io.savemat('x_all' + str(i) + '.mat', {"x_all" + str(i): x_all})
            scipy.io.savemat('y_all' + str(i) + '.mat', {"y_all" + str(i): y_all})
            scipy.io.savemat('z_all' + str(i) + '.mat', {"z_all" + str(i): z_all})
            scipy.io.savemat('t_all' + str(i) + '.mat', {"t_all" + str(i): t_all})
            scipy.io.savemat('x_left_all' + str(i) + '.mat', {"x_left_all" + str(i): x_left_all})
            scipy.io.savemat('y_left_all' + str(i) + '.mat', {"y_left_all" + str(i): y_left_all})
            scipy.io.savemat('z_left_all' + str(i) + '.mat', {"z_left_all" + str(i): z_left_all})
            scipy.io.savemat('t_left_all' + str(i) + '.mat', {"t_left_all" + str(i): t_left_all})
            scipy.io.savemat('x_right_all' + str(i) + '.mat', {"x_right_all" + str(i): x_right_all})
            scipy.io.savemat('y_right_all' + str(i) + '.mat', {"y_right_all" + str(i): y_right_all})
            scipy.io.savemat('z_right_all' + str(i) + '.mat', {"z_right_all" + str(i): z_right_all})
            scipy.io.savemat('t_right_all' + str(i) + '.mat', {"t_right_all" + str(i): t_right_all})
            scipy.io.savemat('x_upper_all' + str(i) + '.mat', {"x_upper_all" + str(i): x_upper_all})
            scipy.io.savemat('y_upper_all' + str(i) + '.mat', {"y_upper_all" + str(i): y_upper_all})
            scipy.io.savemat('z_upper_all' + str(i) + '.mat', {"z_upper_all" + str(i): z_upper_all})
            scipy.io.savemat('t_upper_all' + str(i) + '.mat', {"t_upper_all" + str(i): t_upper_all})
            scipy.io.savemat('x_lower_all' + str(i) + '.mat', {"x_lower_all" + str(i): x_lower_all})
            scipy.io.savemat('y_lower_all' + str(i) + '.mat', {"y_lower_all" + str(i): y_lower_all})
            scipy.io.savemat('z_lower_all' + str(i) + '.mat', {"z_lower_all" + str(i): z_lower_all})
            scipy.io.savemat('t_lower_all' + str(i) + '.mat', {"t_lower_all" + str(i): t_lower_all})
            scipy.io.savemat('x_front_all' + str(i) + '.mat', {"x_front_all" + str(i): x_front_all})
            scipy.io.savemat('y_front_all' + str(i) + '.mat', {"y_front_all" + str(i): y_front_all})
            scipy.io.savemat('z_front_all' + str(i) + '.mat', {"z_front_all" + str(i): z_front_all})
            scipy.io.savemat('t_front_all' + str(i) + '.mat', {"t_front_all" + str(i): t_front_all})
            scipy.io.savemat('x_rear_all' + str(i) + '.mat', {"x_rear_all" + str(i): x_rear_all})
            scipy.io.savemat('y_rear_all' + str(i) + '.mat', {"y_rear_all" + str(i): y_rear_all})
            scipy.io.savemat('z_rear_all' + str(i) + '.mat', {"z_rear_all" + str(i): z_rear_all})
            scipy.io.savemat('t_rear_all' + str(i) + '.mat', {"t_rear_all" + str(i): t_rear_all})
            scipy.io.savemat('train_loss_adam' + str(i) + '.mat', {"train_loss_adam" + str(i): train_loss_adam})
            scipy.io.savemat('train_loss_lbfgs' + str(i) + '.mat', {"train_loss_lbfgs" + str(i): train_loss_lbfgs})
            scipy.io.savemat('elapsed_adam' + str(i) + '.mat', {"elapsed_adam" + str(i): elapsed_adam})
            scipy.io.savemat('elapsed_lbfgs' + str(i) + '.mat', {"elapsed_lbfgs" + str(i): elapsed_lbfgs})
            scipy.io.savemat('elapsed' + str(i) + '.mat', {"elapsed" + str(i): elapsed})
            u_pred, f_u_pred, f_energy_t = model.predict(X_star)
            scipy.io.savemat('f_u_pred_t0.mat', {"f_u_pred_t0": f_u_pred})
            scipy.io.savemat('u_pred_t0.mat', {"u_pred_t0": u_pred})
            scipy.io.savemat('x_star_t0.mat', {"x_star_t0": X_star})
            scipy.io.savemat('energy_t0.mat', {"energy_t0": f_energy_t})
        elif i <= 12:
            new_t = np.ones(len(x0)) * t[0]
            x0_star = np.hstack((x0.flatten()[:, None], y0.flatten()[:, None], z0.flatten()[:, None], new_t.flatten()[:, None]))
            u0, _, _ = model.predict(x0_star)
            model.init_update(x0_star[:, 0:1], x0_star[:, 1:2], x0_star[:, 2:3], u0, X_f, layers, lb, ub)
            start_time = time.time()
            train_loss_adam, train_loss_lbfgs, elapsed_adam, elapsed_lbfgs, x_all, y_all, z_all, t_all, x_left_all, y_left_all, z_left_all, t_left_all, x_right_all, y_right_all, z_right_all, t_right_all, x_upper_all, y_upper_all, z_upper_all, t_upper_all, x_lower_all, y_lower_all, z_lower_all, t_lower_all, x_front_all, y_front_all, z_front_all, t_front_all, x_rear_all, y_rear_all, z_rear_all, t_rear_all = model.train(5000)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))
            scipy.io.savemat('x_all' + str(i) + '.mat', {"x_all" + str(i): x_all})
            scipy.io.savemat('y_all' + str(i) + '.mat', {"y_all" + str(i): y_all})
            scipy.io.savemat('z_all' + str(i) + '.mat', {"z_all" + str(i): z_all})
            scipy.io.savemat('t_all' + str(i) + '.mat', {"t_all" + str(i): t_all})
            scipy.io.savemat('x_left_all' + str(i) + '.mat', {"x_left_all" + str(i): x_left_all})
            scipy.io.savemat('y_left_all' + str(i) + '.mat', {"y_left_all" + str(i): y_left_all})
            scipy.io.savemat('z_left_all' + str(i) + '.mat', {"z_left_all" + str(i): z_left_all})
            scipy.io.savemat('t_left_all' + str(i) + '.mat', {"t_left_all" + str(i): t_left_all})
            scipy.io.savemat('x_right_all' + str(i) + '.mat', {"x_right_all" + str(i): x_right_all})
            scipy.io.savemat('y_right_all' + str(i) + '.mat', {"y_right_all" + str(i): y_right_all})
            scipy.io.savemat('z_right_all' + str(i) + '.mat', {"z_right_all" + str(i): z_right_all})
            scipy.io.savemat('t_right_all' + str(i) + '.mat', {"t_right_all" + str(i): t_right_all})
            scipy.io.savemat('x_upper_all' + str(i) + '.mat', {"x_upper_all" + str(i): x_upper_all})
            scipy.io.savemat('y_upper_all' + str(i) + '.mat', {"y_upper_all" + str(i): y_upper_all})
            scipy.io.savemat('z_upper_all' + str(i) + '.mat', {"z_upper_all" + str(i): z_upper_all})
            scipy.io.savemat('t_upper_all' + str(i) + '.mat', {"t_upper_all" + str(i): t_upper_all})
            scipy.io.savemat('x_lower_all' + str(i) + '.mat', {"x_lower_all" + str(i): x_lower_all})
            scipy.io.savemat('y_lower_all' + str(i) + '.mat', {"y_lower_all" + str(i): y_lower_all})
            scipy.io.savemat('z_lower_all' + str(i) + '.mat', {"z_lower_all" + str(i): z_lower_all})
            scipy.io.savemat('t_lower_all' + str(i) + '.mat', {"t_lower_all" + str(i): t_lower_all})
            scipy.io.savemat('x_front_all' + str(i) + '.mat', {"x_front_all" + str(i): x_front_all})
            scipy.io.savemat('y_front_all' + str(i) + '.mat', {"y_front_all" + str(i): y_front_all})
            scipy.io.savemat('z_front_all' + str(i) + '.mat', {"z_front_all" + str(i): z_front_all})
            scipy.io.savemat('t_front_all' + str(i) + '.mat', {"t_front_all" + str(i): t_front_all})
            scipy.io.savemat('x_rear_all' + str(i) + '.mat', {"x_rear_all" + str(i): x_rear_all})
            scipy.io.savemat('y_rear_all' + str(i) + '.mat', {"y_rear_all" + str(i): y_rear_all})
            scipy.io.savemat('z_rear_all' + str(i) + '.mat', {"z_rear_all" + str(i): z_rear_all})
            scipy.io.savemat('t_rear_all' + str(i) + '.mat', {"t_rear_all" + str(i): t_rear_all})
            scipy.io.savemat('train_loss_adam' + str(i) + '.mat', {"train_loss_adam" + str(i): train_loss_adam})
            scipy.io.savemat('train_loss_lbfgs' + str(i) + '.mat', {"train_loss_lbfgs" + str(i): train_loss_lbfgs})
            scipy.io.savemat('elapsed_adam' + str(i) + '.mat', {"elapsed_adam" + str(i): elapsed_adam})
            scipy.io.savemat('elapsed_lbfgs' + str(i) + '.mat', {"elapsed_lbfgs" + str(i): elapsed_lbfgs})
            scipy.io.savemat('elapsed' + str(i) + '.mat', {"elapsed" + str(i): elapsed})
            u_predname = 'u_pred' + str(i)
            f_u_predname = 'f_u_pred' + str(i)
            energyname = 'energy' + str(i)
            globals()[u_predname], globals()[f_u_predname], globals()[energyname] = model.predict(X_star)
            scipy.io.savemat('f_u_pred_t' + str(i) + '.mat', {"f_u_pred_t" + str(i): globals()[f_u_predname]})
            scipy.io.savemat('u_pred_t' + str(i) + '.mat', {"u_pred_t" + str(i): globals()[u_predname]})
            scipy.io.savemat('energy_t' + str(i) + '.mat', {"energy_t" + str(i): globals()[energyname]})
            scipy.io.savemat('x_star_t' + str(i) + '.mat', {"x_star_t" + str(i): X_star})
