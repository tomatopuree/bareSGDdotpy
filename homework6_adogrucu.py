'''
GRADER, NOTICE

USES PYTHON 3

usage:
python3 homework6_adogrucu.py


IMPORTANT IMPLEMENTATION NOTES!!!!

 * running this file completes every requirement for the homework
 * that includes the test set results
 * the way i implemented the hyperparameter selection makes it so that
 there is no need to retrain the network for test set results
 * this is because i save the weights and the biasses of every trial
 in findBestHyperparameters, so i reuse them to calculate the y_hat
 for the test set

'''

import numpy as np


# [0.1 0.8 0.1] - > [0 1 0]
def prob_to_onehot(yhat):

    return np.eye(10)[np.argmax(yhat, axis = 1)]

# give it y and yhat, get one number (CE loss) back
def cross_entropy_loss(reg, y, yhat, weights1, weights2 ,w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu):

    rowwisesum = np.sum(y * np.log(yhat) , axis = 1)

    lefterior = (-1/yhat.shape[0]) * np.sum(rowwisesum , axis = 0)

    # L1 REG : sum abs val
    l1term = np.abs(weights1).mean() + np.abs(weights2).mean()

    # L2 REG : frobenius (sum squares)
    l2term = np.square(weights1).mean() + np.square(weights2).mean()

    if(reg):
        return lefterior + l1term + l2term
    else:
        return lefterior

# give it train labels and onehotized predictions, get PC
def percent_correct(y, onehotized_yhat):

    a = np.argmax(y, axis = 1)
    b = np.argmax(onehotized_yhat, axis = 1)

    return np.sum(a == b)/y.shape[0]

def relu_prime(z):
    res = z.copy()
    res[res <= 0] = 0
    res[res > 0] = 1
    return res

def sign(matrix):
    res = matrix.copy()
    res[res < 0] = -1
    res[res == 0] = 0
    res[res > 0] = 1
    return res

# tfw you're not main but you take hyperparameters
def gradient_calculator(train_images, train_labels, weights1, weights2, bias1, bias2, w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu):

    # now (784 x batchsize)
    ti_batch = train_images.T

    y_hat, z1, z1_ac = yhatcalculator(train_images, weights1, weights2, bias1, bias2)
    # print("y_hat: ", y_hat.shape)
    # print("z1: ", z1.shape)
    # print("z1ac: ", z1_ac.shape)

    y = train_labels
    w2_gradient = np.dot((y_hat-y).T, z1_ac).T
    w2_gradient += (w2_l2regu * w2_gradient) + (w2_l1regu * sign(w2_gradient))
    b2_gradient = y_hat-y
    # print("w2grad: ", w2_gradient.shape)
    # print("b2grad: ", b2_gradient.shape)

    g = np.dot((y_hat-y), weights2.T) * relu_prime(z1)

    # print("g: ", g.shape)
    w1_gradient = np.dot(g.T, train_images)
    w1_gradient += (w1_l2regu * w1_gradient) + (w1_l1regu * sign(w1_gradient))
    b1_gradient = g.T

    # print("w1grad: ", w1_gradient.shape)
    # print("b2grad: ", b1_gradient.shape)

    return w2_gradient, b2_gradient.T, w1_gradient.T, b1_gradient

# the net
def yhatcalculator(train_images, weights1, weights2, bias1, bias2):

    X = train_images

    z1 = np.dot(weights1.T, X.T)
    z1_b = (np.add(z1.T, bias1)).T
    z1_ac = np.maximum(z1_b, 0, z1_b)

    z2 = np.dot(weights2.T, z1_ac)
    z2_b = (np.add(z2.T, bias2)).T
    z2_ac = (np.exp(z2_b)/ np.sum(np.exp(z2_b), axis = 0))

    return z2_ac.T, z1.T, z1_ac.T



## where the magic happens
def sgd(hidden_layer_count, val_im, val_la, train_images, train_labels, test_images, test_labels, lr, epochs, mb_size, w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu):

    import math
    weights1 = (1/math.sqrt(hidden_layer_count)) * np.random.randn(784,hidden_layer_count)
    weights2 = (1/math.sqrt(10)) * np.random.randn(hidden_layer_count,10)

    bias1 = 0.01 * np.ones((hidden_layer_count))
    bias2 = 0.01 * np.ones((10))

    learning_rate = lr # epsilon
    epc = 0

    while(epc < epochs):

        # minibatch
        for i in range(0,int(train_images.shape[0]/mb_size)):

            # cut shit up
            partial_ti = train_images[(i*mb_size):((i+1)*mb_size),:]
            partial_tl = train_labels[(i*mb_size):((i+1)*mb_size),:]

            # get dem gradients
            w2_gradient, b2_gradient, w1_gradient, b1_gradient = gradient_calculator(partial_ti, partial_tl, weights1, weights2, bias1, bias2, w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu)

            # hillclimb weights&biasses
            # print("w1grad" , w1_gradient[0:10,0:10])
            weights1 = weights1 - learning_rate * w1_gradient

            b1_gradient = np.mean(b1_gradient , axis=1)
            bias1 = bias1 - learning_rate * b1_gradient

            weights2 = weights2 - learning_rate * w2_gradient

            b2_gradient = np.mean(b2_gradient , axis=1)
            bias2 = bias2 - learning_rate * b2_gradient

            # calculate results w new weights and biasses
            yhat, z1, z1_ac = yhatcalculator(train_images, weights1, weights2, bias1, bias2)

            # print CE_loss for train
            print("Current CE loss is (epoch " + str(epc) + ") : ")
            print(" %.3f" % cross_entropy_loss(True, train_labels, yhat, weights1, weights2 ,w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu))

            # print PC for train
            onehotized_yhat = prob_to_onehot(yhat)
            print("Current PC is: ")
            print(" %.3f" % percent_correct(train_labels, onehotized_yhat))

        #increment epoch count
        epc += 1

        # anneal learning rate
        if((epc % 1) == 0):
            learning_rate = learning_rate * 0.99

        # print this at every epoch, or the last iteration.
        print("VAL SET PC is (epoch " + str(epc-1) + ") : ")
        y_hat_val, z1_unused, z1_ac_unused = yhatcalculator(val_im, weights1, weights2, bias1, bias2)

        onehotized_yhat = prob_to_onehot(y_hat_val)
        pc = percent_correct(val_la, onehotized_yhat)
        print(pc)

        print("VAL SET CE loss is: ")
        ce = cross_entropy_loss(True, val_la, y_hat_val , weights1, weights2 ,w1_l1regu, w2_l1regu, w1_l2regu, w2_l2regu)
        print(ce)

        import time
        time.sleep(0.5)

    return pc, ce, weights1, bias1, weights2, bias2

# here be that function you guys wanted
def findBestHyperparameters(valid_images, valid_labels, ti, tl, test_images, test_labels):
    resultsandthings = []

    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 1000, 0.5, 0.5, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(40, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 1000, 0.1, 0.1, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(50, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 1000, 1, 1, 1, 1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.0015, 20, 1000, 0.1, 0.1, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(40, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 1000, 0.3, 0.3, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(40, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 1000, 0.1, 0.1, 0.3, 0.3)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 25, 1000, 0.1, 0.1, 0.3, 0.3)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.002, 20, 1000, 0.1, 0.1, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 15, 5000, 0.1, 0.1, 0.1, 0.1)
    resultsandthings.append(resultandthing)
    resultandthing = sgd(30, valid_images, valid_labels, ti, tl, test_images, test_labels, 0.001, 20, 5000, 0.7, 0.7, 0.2, 0.2)
    resultsandthings.append(resultandthing)

    resultspace = [(30, 0.001, 20, 1000, 0.5, 0.5, 0.1, 0.1),
                    (40, 0.001, 20, 1000, 0.1, 0.1, 0.1, 0.1),
                    (50, 0.001, 20, 1000, 1, 1, 1, 1),
                    (30, 0.0015, 20, 1000, 0.1, 0.1, 0.1, 0.1),
                    (40, 0.001, 20, 1000, 0.3, 0.3, 0.1, 0.1),
                    (40, 0.001, 20, 1000, 0.1, 0.1, 0.3, 0.3),
                    (30, 0.001, 25, 1000, 0.1, 0.1, 0.3, 0.3),
                    (30, 0.002, 20, 1000, 0.1, 0.1, 0.1, 0.1),
                    (30, 0.001, 15, 5000, 0.1, 0.1, 0.1, 0.1),
                    (30, 0.001, 20, 5000, 0.7, 0.7, 0.2, 0.2)]


    x = 0
    index = 0
    for i in range(len(resultsandthings)):
        if(resultsandthings[i][0] > x):
            x = resultsandthings[i][0]
            index = i

    return resultspace[index], resultsandthings, index


if __name__ == "__main__":

    import numpy as np

    train_images = np.load("mnist_train_images.npy")
    train_labels = np.load("mnist_train_labels.npy")

    test_images = np.load("mnist_test_images.npy")
    test_labels = np.load("mnist_test_labels.npy")

    valid_images = np.load("mnist_validation_images.npy")
    valid_labels = np.load("mnist_validation_labels.npy")

    # randomize things
    idx = np.random.permutation(train_images.shape[0])
    ti,tl = train_images[idx], train_labels[idx]

    besthyps, resultsandthings, index = findBestHyperparameters(valid_images, valid_labels, ti, tl, test_images, test_labels)

    print("")
    print("")
    print("")
    print("The hyperparameters for the best learner are: ", besthyps)
    print("")
    print("")
    print("")

    ## test set stuff

    weights1 = resultsandthings[index][2]
    bias1 = resultsandthings[index][3]
    weights2 = resultsandthings[index][4]
    bias2 = resultsandthings[index][5]

    print("TEST SET PC is: ")
    y_hat_test, z1_unused, z1_ac_unused = yhatcalculator(test_images, weights1, weights2, bias1, bias2)

    onehotized_yhat = prob_to_onehot(y_hat_test)
    pc = percent_correct(val_la, onehotized_yhat)
    print(pc)

    print("TEST SET CE loss is: ")
    ce = cross_entropy_loss(True, test_labels, y_hat_test, weights1, weights2 ,besthyps[4], besthyps[5], besthyps[6], besthyps[7])
    print(ce)
