import numpy as np
import math
import matplotlib.pyplot as plt

# plot target fuction
p = np.linspace(-2, 2, 500)
t = np.exp(-1 * abs(p)) * np.sin(math.pi * p)

plt.plot(p, t)
plt.title("Function plot")
plt.show()


# -- functions --
def purelin(n):
    return n

def purelin_deriv(n):
    return 1

def logsig(n):
    return 1 / (1 + np.exp(-1 * n))

def logsig_deriv(n):
    return (1 - logsig(n)) * logsig(n)

# -- initialize weight and bais
S = eval(input("enter the number of neurons:"))
R = 1
w1 = 1 * np.random.rand(S,R) - 0.5
b1 = 1 * np.random.rand(S,R) - 0.5
w2 = 1 * np.random.rand(R,S) - 0.5
b2 = 1 * np.random.rand(R,R) - 0.5

# -- bp function --
def bpNetwork(p, t, w1, w2, b1, b2, gra):
    alpha = 0.1
    epoch = 2000
    E = list()
    MSE = list()
    epochs = list()
    sum_s2 = 0
    sum_s1 = 0
    sum_s2_dot_a1 = 0
    sum_s1_dot_a0 = 0

    if gra == 'SGD' or gra == 'sgd':
        flag = 1
    elif gra == 'BGD' or gra == 'bgd':
        flag = 2

    # train network
    for j in range(epoch):
        for i in range(len(p)):
            # forward
            n1 = np.dot(w1, p[i]) + b1
            a1 = logsig(n1)
            n2 = np.dot(w2, a1) + b2
            a2 = purelin(n2)
            e = t[i] - a2
            E.append(e)
            # bp
            s2 = -2 * purelin_deriv(n2) * e
            F1 = np.diag(logsig_deriv(n1).flat)
            s1 = np.dot(np.dot(F1, w2.T), s2)

            if flag == 1:
                # update weight and bais
                w2 = w2 - alpha * s2 * a1.T
                b2 = b2 - alpha * s2
                w1 = w1 - alpha * s1 * p[i].T
                b1 = b1 - alpha * s1
            elif flag == 2:
                # set value for update
                s2_dot_a1 = s2 * a1.T
                sum_s2_dot_a1 += s2_dot_a1
                s1_dot_a0 = s1 * p[i].T
                sum_s1_dot_a0 += s1_dot_a0
                sum_s2 += s2
                sum_s1 += s1
        if flag == 2:
            # update weight and bais
            avg_s2_dot_a1 = sum_s2_dot_a1 / len(p)
            avg_s1_dot_a0 = sum_s1_dot_a0 / len(p)
            avg_s2 = sum_s2 / len(p)
            avg_s1 = sum_s1 / len(p)
            w2 = w2 - alpha * avg_s2_dot_a1
            b2 = b2 - alpha * avg_s2
            w1 = w1 - alpha * avg_s1_dot_a0
            b1 = b1 - alpha * avg_s1
        # get mse
        mse = np.square(E).mean()
        MSE.append(float(mse))
        epochs.append(j)
    
    # predict
    pred = list()
    for i in range(len(p)):
        n1_new = np.dot(w1, p[i]) + b1
        a1_new = logsig(n1_new)
        n2_new = np.dot(w2, a1_new) + b2
        a2_new = purelin(n2_new)
        pred.append(float(a2_new))
    
    # plot prediction and mse
    if flag == 1:
        plt.plot(p, pred)
        plt.title("SGD Prediction")
        plt.show()

        plt.plot(epochs, MSE)
        plt.title("SGD MSE")
        plt.show() 
    elif flag == 2:
        plt.plot(p, pred)
        plt.title("BGD Prediction")
        plt.show()

        plt.plot(epochs, MSE)
        plt.title("BGD MSE")
        plt.show()

# -- run function --
# sgd
bpNetwork(p, t, w1, w2, b1, b2, 'sgd')
# bgd
bpNetwork(p, t, w1, w2, b1, b2, 'bgd')



