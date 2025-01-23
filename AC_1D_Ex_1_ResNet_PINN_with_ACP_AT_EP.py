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
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub):

        X0 = np.concatenate((x0, 0 * x0 + lb[1]), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.x_energy = np.vstack((np.vstack((X_f[:, 0:1], X_lb[:, 0:1])), X_ub[:, 0:1]))
        self.t_energy = np.vstack((np.vstack((X_f[:, 1:2], X_lb[:, 1:2])), X_ub[:, 1:2]))

        self.u0 = u0
        lbfgs_iter = 5000
        self.loss_lbfgs_new = np.zeros((2 * lbfgs_iter + 1))
        self.loss_lbfgs = np.zeros((lbfgs_iter + 1))
        self.count = -1

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.x_energy_tf = tf.placeholder(tf.float32, shape=[None, self.x_energy.shape[1]])
        self.t_energy_tf = tf.placeholder(tf.float32, shape=[None, self.t_energy.shape[1]])

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)


        # tf Graphs
        self.u0_pred, _ = self.net_u(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.energy, _ = self.net_f_u(self.x_f_tf, self.t_f_tf)
        _, _, self.energy_t_pred = self.net_f_u(self.x_energy_tf, self.t_energy_tf)

        # Loss
        self.loss = 100 * tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred) + tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
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

    ## ResNet
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
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

    def net_u(self, x, t):
        X = tf.concat([x, t], 1)

        u = self.neural_net(X, self.weights, self.biases)

        u_x = tf.gradients(u, x)[0]

        return u, u_x

    def net_f_u(self, x, t):
        u, u_x = self.net_u(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f_u = u_t - 0.0001 * u_xx + 5 * (u ** 3 - u)
        energy = (1 / self.x_energy.shape[0]) * (np.sum(5 * (10 ** (-5)) * (np.abs(u_x) ** 2)) + (
            np.sum(1.25 * (1 - u ** 2) ** 2)))

        energy_t = tf.gradients(energy, t)[0]

        return f_u, energy, energy_t

    def callback(self, loss):
        self.count = self.count + 1
        print('Loss:', loss)
        self.lbfgs_loss_history(loss)

    def lbfgs_loss_history(self, loss):
        if self.count > 5000:
            self.loss_lbfgs_new[self.count] = loss
            return self.loss_lbfgs_new
        else:
            self.loss_lbfgs[self.count] = loss
        return self.loss_lbfgs


    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   self.x_energy_tf: self.x_energy, self.t_energy_tf: self.t_energy}


        start_time = time.time()
        k = 0
        tol = 5 * 10**(-2)
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
            if it > 1000 and max(a[k-1]-a[k-2], a[k-2]-a[k-3], a[k-3]-a[k-4]) > tol:
                print("x_f Shape: ", self.x_f.shape, "\n")
                n = int(0.15 * self.x_f.shape[0])
                x_test = self.lb + (self.ub - self.lb) * lhs(2, n)
                u_pred_test, f_u_pred_test, _ = model.predict(x_test)
                loss_test = np.abs(f_u_pred_test)


                n = int(0.025 * self.x_f.shape[0])
                t_test = self.lb[1] + (self.ub[1] - self.lb[1]) * lhs(1, n)
                t_t2 = t_test[:, 0:1]
                t_t2 = t_t2.flatten()[:, None]
                X_lb_test = np.concatenate((0 * t_t2 + lb[0], t_t2), 1)  # (lb[0], tb)
                X_ub_test = np.concatenate((0 * t_t2 + ub[0], t_t2), 1)
                u_lb_pred_test, f_u_lb_pred_test, _ = model.predict(X_lb_test)
                loss_lb_test = np.abs(f_u_lb_pred_test)

                total_err = sum(loss_test)
                threshold = total_err / 10
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
                    self.t_f = np.vstack((self.t_f, squeezed_X[:, 1:2]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 1:2]))
                    print("x_f Shape: ", self.x_f.shape, "\n")
                elif index == 2:
                    print("total number of selected points:", index - 1)
                    x_id = np.argsort(-loss_test, axis=0)[:index - 1]
                    print("Adding new training points:", x_test[x_id], "\n")
                    squeezed_X = np.squeeze(x_test[x_id])
                    self.x_f = np.vstack((self.x_f, squeezed_X[0]))
                    self.t_f = np.vstack((self.t_f, squeezed_X[1]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[1]))
                    print("x_f Shape: ", self.x_f.shape, "\n")
                else:
                    print("No point is added.", "\n")

                total_err_lb = sum(loss_lb_test)
                threshold_lb = total_err_lb / 10
                tot_percent_lb = 0
                index_lb = 0
                err_eq_sorted_lb = np.argsort(-loss_lb_test, axis=0)
                while tot_percent_lb < threshold_lb:
                    tot_percent_lb = tot_percent_lb + loss_lb_test[err_eq_sorted_lb][index_lb]
                    index_lb = index_lb + 1

                if index_lb > 2:
                    print("total number of selected points:", index_lb - 1)
                    x_id = np.argsort(-loss_lb_test, axis=0)[:index_lb - 1]
                    print("Adding new training points:", X_lb_test[x_id], "\n")
                    squeezed_X = np.squeeze(X_lb_test[x_id])
                    self.x_lb = np.vstack((self.x_lb, squeezed_X[:, 0:1]))
                    self.t_lb = np.vstack((self.t_lb, squeezed_X[:, 1:2]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 1:2]))
                    print("x_lb Shape: ", self.x_lb.shape, "\n")
                    print("Adding new training points:", X_ub_test[x_id], "\n")
                    squeezed_X = np.squeeze(X_ub_test[x_id])
                    self.x_ub = np.vstack((self.x_ub, squeezed_X[:, 0:1]))
                    self.t_ub = np.vstack((self.t_ub, squeezed_X[:, 1:2]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[:, 0:1]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[:, 1:2]))
                    print("x_ub Shape: ", self.x_ub.shape, "\n")
                elif index_lb == 2:
                    print("total number of selected points:", index_lb - 1)
                    x_id = np.argsort(-loss_lb_test, axis=0)[:index_lb - 1]
                    print("Adding new training points:", X_lb_test[x_id], "\n")
                    squeezed_X = np.squeeze(X_lb_test[x_id])
                    self.x_lb = np.vstack((self.x_lb, squeezed_X[0]))
                    self.t_lb = np.vstack((self.t_lb, squeezed_X[1]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[1]))
                    print("x_lb Shape: ", self.x_lb.shape, "\n")
                    print("Adding new training points:", X_ub_test[x_id], "\n")
                    squeezed_X = np.squeeze(X_ub_test[x_id])
                    self.x_ub = np.vstack((self.x_ub, squeezed_X[0]))
                    self.t_ub = np.vstack((self.t_ub, squeezed_X[1]))
                    self.x_energy = np.vstack((self.x_energy, squeezed_X[0]))
                    self.t_energy = np.vstack((self.t_energy, squeezed_X[1]))
                    print("x_ub Shape: ", self.x_ub.shape, "\n")
                else:
                    print("No point is added.", "\n")


            tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                           self.u0_tf: self.u0,
                           self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                           self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                           self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                           self.x_energy_tf: self.x_energy, self.t_energy_tf: self.t_energy}

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
        return a, b, elapsed_Adam, elapsed_LBFGS, self.x_f, self.t_f, self.x_lb, self.t_lb, self.x_ub, self.t_ub

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_energy = self.sess.run(self.energy, tf_dict)

        return u_star, f_u_star, f_energy

if __name__ == "__main__":
    tol_energy = 10 ** (-3)
    energy_diff = 1
    i = 0
    while energy_diff > tol_energy:
        noise = 0.0

        # Domain bounds
        lb = np.array([-1.0, 0.1*i])
        ub = np.array([1.0, 0.1*(i+1)])

        N0 = 128
        N_f = 500
        layers = [2, 100, 100, 100, 100, 1]

        t = np.linspace(0.1 *i, 0.1*(i+1), 21)
        x = np.linspace(-1, 1, 512)
        t = t.flatten()[:, None]
        x = x.flatten()[:, None]

        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        tb = t

        X_f = lb + (ub - lb) * lhs(2, N_f)

        x0 = np.linspace(-1.0, 1.0, N0)
        x0 = x0.flatten()[:, None]

        if i==0:
            u0 = (x0 ** 2) * np.cos(np.pi * x0)
            model = PhysicsInformedNN(x0, u0, tb, X_f, layers, lb, ub)
            start_time = time.time()
            train_loss_adam, train_loss_lbfgs, elapsed_adam, elapsed_lbfgs, all_x, all_t, all_x_lb, all_t_lb, all_x_ub, all_t_ub = model.train(5000)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))
            u_pred, f_u_pred, f_energy_t = model.predict(X_star)
            scipy.io.savemat('all_x_train_t0.mat', {"all_x_train_t0": all_x})
            scipy.io.savemat('all_t_train_t0.mat', {"all_t_train_t0": all_t})
            scipy.io.savemat('all_x_lb_train_t0.mat', {"all_x_lb_train_t0": all_x_lb})
            scipy.io.savemat('all_t_lb_train_t0.mat', {"all_t_lb_train_t0": all_t_lb})
            scipy.io.savemat('all_x_ub_train_t0.mat', {"all_x_ub_train_t0": all_x_ub})
            scipy.io.savemat('all_t_ub_train_t0.mat', {"all_t_ub_train_t0": all_t_ub})
            scipy.io.savemat('f_u_pred_t0.mat', {"f_u_pred_t0": f_u_pred})
            scipy.io.savemat('u_pred_t0.mat', {"u_pred_t0": u_pred})
            scipy.io.savemat('x_star_t0.mat', {"x_star_t0": X_star})
            scipy.io.savemat('energy_t0.mat', {"energy_t0": f_energy_t})
            scipy.io.savemat('train_loss_adam' + str(i) + '.mat', {"train_loss_adam" + str(i): train_loss_adam})
            scipy.io.savemat('train_loss_lbfgs' + str(i) + '.mat', {"train_loss_lbfgs" + str(i): train_loss_lbfgs})
            scipy.io.savemat('elapsed_adam' + str(i) + '.mat', {"elapsed_adam" + str(i): elapsed_adam})
            scipy.io.savemat('elapsed_lbfgs' + str(i) + '.mat', {"elapsed_lbfgs" + str(i): elapsed_lbfgs})
            name = 'model'

            energy_new = np.zeros((len(t), len(x)))
            for j in range(0, len(t)):
                for k in range(0, len(x)):
                    if t[j] == X_star[j*512 + k, 1:2]:
                        energy_new[j, k] = f_energy_t[j*512 + k]

            energy_sum = np.sum(energy_new, 1)
            energy_diff = (energy_sum[0]-energy_sum[-1])

        elif i >= 1:
            new_t = np.ones(len(x0)) * t[0]
            x0_star = np.hstack((x0.flatten()[:, None], new_t.flatten()[:, None]))
            u0, _, _ = np.array(globals()[name].predict(x0_star))
            name = 'model' + str(i)
            globals()[name] = PhysicsInformedNN(x0, u0, tb, X_f, layers, lb, ub)
            start_time = time.time()
            train_loss_adam, train_loss_lbfgs, elapsed_adam, elapsed_lbfgs, all_x, all_t, all_x_lb, all_t_lb, all_x_ub, all_t_ub = globals()[name].train(5000)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))
            u_predname = 'u_pred' + str(i)
            f_u_predname = 'f_u_pred' + str(i)
            energy_predname = 'energy' + str(i)
            globals()[u_predname], globals()[f_u_predname], globals()[energy_predname] = globals()[name].predict(X_star)
            scipy.io.savemat('all_x_train_t' + str(i) + '.mat', {"all_x_train_t" + str(i): all_x})
            scipy.io.savemat('all_t_train_t' + str(i) + '.mat', {"all_t_train_t" + str(i): all_t})
            scipy.io.savemat('all_x_lb_train_t' + str(i) + '.mat', {"all_x_lb_train_t" + str(i): all_x_lb})
            scipy.io.savemat('all_t_lb_train_t' + str(i) + '.mat', {"all_t_lb_train_t" + str(i): all_t_lb})
            scipy.io.savemat('all_x_ub_train_t' + str(i) + '.mat', {"all_x_ub_train_t" + str(i): all_x_ub})
            scipy.io.savemat('all_t_ub_train_t' + str(i) + '.mat', {"all_t_ub_train_t" + str(i): all_t_ub})
            scipy.io.savemat('f_u_pred_t' + str(i) + '.mat', {"f_u_pred_t" + str(i): globals()[f_u_predname]})
            scipy.io.savemat('u_pred_t' + str(i) + '.mat', {"u_pred_t" + str(i): globals()[u_predname]})
            scipy.io.savemat('x_star_t' + str(i) + '.mat', {"x_star_t" + str(i): X_star})
            scipy.io.savemat('energy_t' + str(i) + '.mat', {"energy_t" + str(i): globals()[energy_predname]})
            scipy.io.savemat('train_loss_adam' + str(i) + '.mat', {"train_loss_adam" + str(i): train_loss_adam})
            scipy.io.savemat('train_loss_lbfgs' + str(i) + '.mat', {"train_loss_lbfgs" + str(i): train_loss_lbfgs})
            scipy.io.savemat('elapsed_adam' + str(i) + '.mat', {"elapsed_adam" + str(i): elapsed_adam})
            scipy.io.savemat('elapsed_lbfgs' + str(i) + '.mat', {"elapsed_lbfgs" + str(i): elapsed_lbfgs})

            energy_new = np.zeros((len(t), len(x)))
            for j in range(0, len(t)):
                for k in range(0, len(x)):
                    if t[j] == X_star[j * 512 + k, 1:2]:
                        energy_new[j, k] = globals()[energy_predname][j * 512 + k]

            energy_sum = np.sum(energy_new, 1)
            energy_diff = (energy_sum[0] - energy_sum[-1])


        i = i + 1



