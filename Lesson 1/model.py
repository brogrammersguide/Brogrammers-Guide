import numpy as np
from sklearn.datasets import make_classification


def make_weights(input_size, output_size):

    weights = np.random.randn(input_size, output_size)
    return weights * 1/np.sqrt(input_size)

def make_biases(output_size):


    biases = np.ones(output_size)
    return biases

def relu(x, derivative = False):


    if not derivative:

        x[x < 0] = 0

        return x

    x[x > 0] = 1
    x[x <= 0] = 0

    return x

def softmax(x):

    x = x - np.max(x, axis = 1, keepdims = True)
    x = np.exp(x)
    return x/np.sum(x, axis = 1, keepdims = True)





class MLP:

    def __init__(self, sizes):


        self.weights = self.get_weights(sizes)
        self.biases = self.get_biases(sizes)
        self.sizes = sizes

    def get_weights(self, sizes):


        weights = []
        for ii, size in enumerate(sizes[:-1]):

            weights.append(make_weights(size, sizes[ii+1]))

        return weights


    def get_biases(self, sizes):

        return [make_biases(size) for size in sizes[1:]]

    def forward_propagation(self, X):


        a = X
        preactivations = []
        activations = [X]

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            #compute preactivation
            z = a.dot(weight) + bias
            #apply activation function
            a = relu(z)
            #store activations and preactivations for backpropagation
            activations.append(a)
            preactivations.append(z)
        #compute final scores
        scores = a.dot(self.weights[-1]) + self.biases[-1]
        #apply softmax function
        probabilities = softmax(scores)
        #append preactivations update instance attributes return softmax
        preactivations.append(scores)
        self.preactivations = preactivations
        self.activations = activations
        return probabilities

    def backpropagation(self, probabilities,
                        y, one_hot = False, regularization_penalty = 0):

        #Create empty list to contain gradients
        weight_gradients = []
        bias_gradients = []
        preactivation_gradients = []
        preactivations = self.preactivations
        activations = self.activations
        #compute error for layers
        weights = list(reversed(self.weights))
        for ii, preactivation in enumerate(reversed(preactivations)):
            #check format of y and compute error in last layer
            if not ii:
                if one_hot:
                    upper_error = probabilities - y
                else:
                    upper_error = probabilities
                    upper_error[range(len(y)),y] -= 1
                preactivation_gradients.insert(0, upper_error)
            else:
                #store index of weight connecting to upper layer
                weight_upper = ii - 1
                #chain local gradient of activation to upper error
                activation_gradient = np.dot(upper_error,
                                             weights[weight_upper].T)

                #chain local gradient of preactivation to gradient of layer activation
                upper_error = np.multiply(activation_gradient,
                                          relu(preactivation,
                                               derivative = True))

                #add error to beginning of gradient list
                preactivation_gradients.insert(0, upper_error)

        #compute weight and bias gradients
        for ii, activation in enumerate(activations):
            #chain local gradient of weight to upper error
            weight_gradient = np.dot(activation.T, preactivation_gradients[ii])
            #append gradients to list
            weight_gradients.append(weight_gradient)
            #compute bias gradients and append to list
            bias_gradients.append(np.sum(preactivation_gradients[ii], axis = 0))

        #add regularization term to gradients
        weight_gradients = [weight_gradients[i] +
                           regularization_penalty * self.weights[i]
                           for i in range(len(self.weights))]

        return weight_gradients, bias_gradients

    def regularization_loss(self, regularization_penalty):

        if regularization_penalty:

            reg_loss = sum([(regularization_penalty * (1/2 * weight**2)).sum()
            for weight in self.weights])

        else:

            reg_loss = 0
        return reg_loss


    def loss(self, scores, y, one_hot = False, regularization_penalty = 0):


        if one_hot:
            predicted = scores[range(len(scores)), np.argmax(y, axis = 1)]
        else:
            predicted = scores[range(len(scores)), y]

        loss = -np.log(predicted).mean()
        loss += self.regularization_loss(regularization_penalty)

        weight_gradients, bias_gradients = self.backpropagation(scores,
                                    y,
                                    one_hot = one_hot,
                                    regularization_penalty = regularization_penalty)

        return loss, weight_gradients, bias_gradients

    def accuracy(self, X, y):


        return np.mean(self.predict(X) == y)



    def update_paramaters(self, weight_gradients, bias_gradients, learning_rate):

        for layer in range(len(self.weights)):

            self.weights[layer] = self.weights[layer] - (learning_rate *
                        weight_gradients[layer])
            self.biases[layer] = self.biases[layer] - (learning_rate *
                       bias_gradients[layer])

    def fit(self, X, y, batch_size = 128, regularization_penalty = 0,
            learning_rate = 1e-3, one_hot = False, epochs = 10):

        for epoch in range(epochs):

            for iteration in range(X.shape[0]//batch_size + 1):
                batch = np.random.choice(range(X.shape[0]), batch_size)
                xbatch = X[batch]
                ybatch = y[batch]

                scores = self.forward_propagation(xbatch)
                loss, weight_gradients, bias_gradients = self.loss(scores,
                                ybatch,
                                one_hot = one_hot,
                                regularization_penalty = regularization_penalty)

                self.update_paramaters(weight_gradients,
                                       bias_gradients, learning_rate)

            print("At epoch ", epoch, "the loss is ", loss)

    def predict(self, X):

        return np.argmax(self.forward_propagation(X), axis = 1)


if __name__ == "__main__":


    X, y = make_classification()

    clf = MLP((20, 60, 100, 2))

    clf.fit(X, y, epochs = 100)

    print(clf.accuracy(X, y))