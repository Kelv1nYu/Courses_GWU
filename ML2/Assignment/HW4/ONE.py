import numpy as np
import matplotlib.pyplot as plt

#--- Transfer functions ---

def hardlim(n):
	if n < 0:
		return 0
	else:
		return 1

def hardlims(n):
    if n < 0:
        return -1
    else:
        return 1

def purelin(n):
    return n

def satlin(n):
    if n < 0:
        return 0
    elif n > 1:
        return 1
    else:
        return n

def satlins(n):
    if n < -1:
        return -1
    elif n > 1:
        return 1
    else:
        return n

def poslin(n):
    if n < 0:
        return 0
    else:
        return n

# --- E.1 ---
x_1 = np.arange(-2, 2, 0.1)
y_1 = list()
for i in x_1:
    y_1.append(purelin(poslin(-1 * i + 0.5) + poslin(i + 1) - 1))

plt.plot(x_1, y_1, 'b')
plt.title('E.1')
plt.show()

# --- E.2 ---
plt.figure()
plt.subplot(211)
x_2 = np.arange(-2, 2, 0.1)
y1 = list()
for i in x_2:
    y1.append(hardlims(i + 1))
plt.plot(x_1, y1)

plt.subplot(245)
y2 = list()
for i in x_2:
    y2.append(hardlim(-1 * i + 1))
plt.plot(x_2, y2)

plt.subplot(246)
y3 = list()
for i in x_2:
    y3.append(purelin(2 * i + 3))
plt.plot(x_2, y3)

plt.subplot(247)
y4 = list()
for i in x_2:
    y4.append(satlins(2 * i + 3))
plt.plot(x_2, y4)

plt.subplot(248)
y5 = list()
for i in x_2:
    y5.append(poslin(-2 * i - 1))
plt.plot(x_2, y5)
plt.show()


# --- E.3 ---
x_3 = np.arange(-3, 3, 0.1)
y_3 = list()
for i in x_3:
    y_3.append(purelin(satlin(2 * i + 2) - satlin((i - 1))))
plt.plot(x_3, y_3, 'b')
plt.title('E.3')
plt.show()

# --- E.6 ---

# function for perception
def network(p, t):

    # initial weight and bias, you can randomly initialize them or choose values on your own
    # w = np.random.randn(2)
    w = np.array([0.5, 0.5])
    b = 0.5
    # b = np.random.randn(1)
    while True:
        # create a list to store e value
        eList = list()
        for i in range(len(p)):
            # get new weight and bias
            n = np.dot(w, np.array(p[i]).T) + b
            # hardlim()
            if n >= 0:
                a = 1
            else:
                a = 0
            e = t[i] - a
            eList.append(e)
            w = w + np.multiply(e, p[i])
            b = b + e
        # check if all e values are 0
        if not any(eList):
            break

    return w, b

# initial input and target
p = [[1, 4], [1, 5], [2, 4], [2, 5], [3, 1], [3, 2], [4, 1], [4, 2]]
t = [0, 0, 0, 0, 1, 1, 1, 1]

# get weight and bias
weight, bias = network(p, t)
print("weight: ", weight)
print("bias: ", bias)

# test weight and bias
testList = list()
for index in range(len(p)):
    n = np.dot(weight, np.array(p[index]).T) + bias
    if n >= 0:
        a = 1
    else:
        a = 0
    e = t[index] - a
    testList.append(e)

print(testList)

# draw Decision Boundary

if weight[1] == 0:
    y_6 = np.linspace(-5,5,10)
    x_6 = (- weight[1] * y_6 - bias) / weight[0]

else:
        x_6 = np.linspace(-5,5,10)
        y_6 = (- weight[0] * x_6 - bias) / weight[1]

# red is 0, blue is 1
plt.scatter(1, 4, color='red')
plt.scatter(1, 5, color='red')
plt.scatter(2, 4, color='red')
plt.scatter(2, 5, color='red')
plt.scatter(3, 1, color='blue')
plt.scatter(3, 2, color='blue')
plt.scatter(4, 1, color='blue')
plt.scatter(4, 2, color='blue')

# plot weight vector
plt.quiver(0, 0, weight[0], weight[1], angles='xy', scale_units='xy', scale=1)
plt.plot(x_6, y_6, '-r', label='g(x)')
plt.title('Final decision boundary')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

