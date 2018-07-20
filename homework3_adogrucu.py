'''
GRADER, NOTICE

USES PYTHON 3

usage:
python3 homework3_adogrucu.py

What does this script do you ask?

It conducts stochasthic gradient descent on MNIST using a neural network with only 
one layer. It uses batches (stochasthic, remember?) and learning rate annealation.

The forward pass is in yhatcalculator(). 
The Test set results are printed at every epoch for conveniency.

Takes around 5 minutes to converge at test set PC > %91.5
'''

import numpy as np



## HOMEWERK 3



# [0.1 0.8 0.1] - > [0 1 0]
def prob_to_onehot(yhat):

    return np.eye(10)[np.argmax(yhat, axis = 1)]

# give it y and yhat, get one number (CE loss) back
def cross_entropy_loss(y, yhat):

    rowwisesum = np.sum(y * np.log(yhat) , axis = 1)

    return (-1/yhat.shape[0]) * np.sum(rowwisesum , axis = 0)

# give it train labels and onehotized predictions, get PC
def percent_correct(y, onehotized_yhat):

    a = np.argmax(y, axis = 1)
    b = np.argmax(onehotized_yhat, axis = 1)

    return np.sum(a == b)/y.shape[0]

# X:(784 x batchsize) weights:(784x10)
def gradient_calculator(train_images, train_labels, weights):

    print(train_images.shape)
    print(train_labels.shape)

    # now (784 x batchsize)
    ti_batch = train_images.T

    yhat = yhatcalculator(train_images, weights)
    print(yhat.shape)

    y = train_labels

    return (1/ti_batch.shape[1]) * np.dot(ti_batch, (yhat - y))

# regressor: X (55kx784), weights(784,10)
def yhatcalculator(train_images, weights):

    X_b = train_images.T

    #pre-activation
    z = np.dot(weights.T, X_b)

    #post activation (sum columnwise cause transposed z)
    z_ac = (np.exp(z)/ np.sum(np.exp(z), axis = 0))

    #(55k x 10)
    return z_ac.T


## where the magic happens


def sgd(train_images, train_labels, test_images, test_labels, lr, epochs, mb_size):

    w_0 = 0.01 * np.random.randn(784,10)

    learning_rate = lr # epsilon
    epc = 0
    weights = w_0

    while(epc < epochs):

        # 550 times
        for i in range(0,int(train_images.shape[0]/mb_size)):

            partial_ti = train_images[(i*mb_size):((i+1)*mb_size),:]

            partial_tl = train_labels[(i*mb_size):((i+1)*mb_size),:]

            #update gradient with weight
            gradient = gradient_calculator(partial_ti, partial_tl, weights)

            #move opposite to gradient, multd by step size
            weights = weights - learning_rate * gradient

            #calculate results with new weight
            yhat = yhatcalculator(train_images, weights)

            # print CE_loss for train
            print("Current CE loss is (epoch " + str(epc) + ") : ")
            print(cross_entropy_loss(train_labels, yhat))

            # print PC for train
            onehotized_yhat = prob_to_onehot(yhat)
            print("Current PC is: ")
            print(percent_correct(train_labels, onehotized_yhat))

        #increment epoch count
        epc += 1

        # anneal learning rate
        if((epc % 1) == 0):
            learning_rate = learning_rate * 0.99

        # print this at every epoch, or the last iteration.
        print("TEST SET PC is (epoch " + str(epc) + ") : ")
        y_hat_test = yhatcalculator(test_images, weights)
        onehotized_yhat = prob_to_onehot(y_hat_test)
        print(percent_correct(test_labels, onehotized_yhat))

        print("TEST SET CE loss is: ")
        print(cross_entropy_loss(test_labels, y_hat_test))

        import time
        time.sleep(5)




if __name__ == "__main__":

    import numpy as np

    train_images = np.load("mnist_train_images.npy")
    train_labels = np.load("mnist_train_labels.npy")

    test_images = np.load("mnist_test_images.npy")
    test_labels = np.load("mnist_test_labels.npy")

    # randomize things
    idx = np.random.permutation(train_images.shape[0])
    ti,tl = train_images[idx], train_labels[idx]

    sgd(ti, tl, test_images, test_labels, 0.14, 60, 1000)
