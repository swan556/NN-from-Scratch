import numpy as np
import json
import nnfs
from nnfs.datasets import spiral_data
class layer:
    def __init__(self, n_i, n_n):
        self.w = 0.01 * np.random.randn(n_i, n_n)
        self.b = np.zeros((1, n_n))
    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w) + self.b
    def backward(self, dL_dz):
        self.dL_dw = np.dot(self.x.T, dL_dz)
        self.dL_db = np.sum(dL_dz, axis = 0, keepdims=True)
        self.dL_dx = np.dot(dL_dz, self.w.T)

class relu:
    def forward(self, z):
        self.z = z
        self.a = np.maximum(0, z)
    def backward(self, dL_da):
        self.dL_dz = dL_da.copy()
        self.dL_dz[self.a <= 0] = 0

class softmax:
    def forward(self, z):
        self.z = z
        z -= np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(z)
        self.y_hat = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
    def backward(self, dL_da):
        self.dL_dz = np.empty_like(dL_da)
        for index, (single_a, single_dL_dz) in enumerate(zip(self.y_hat, dL_da)): 
            single_a = single_a.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_a) - np.dot(single_a, single_a.T)
            self.dL_dz[index] = np.dot(jacobian_matrix, single_dL_dz)

class loss:
    def calculate(self, y_hat, y):
        sample_losses = self.forward(y_hat, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class xentropy(loss):
    def forward(self, y_hat, y_true):
        samples = len(y_hat)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dL_dy, y_true):
        samples = len(dL_dy)
        labels = len(dL_dy[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dL_da = -y_true / dL_dy
        self.dL_da /= samples

class softmax_xentropy():
    def __init__(self):
        self.activation = softmax()
        self.loss = xentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.y_hat
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
            
class optimiser_SGD:
    def __init__(self, alpha = 0.5, decay = 0.001):
        self.alpha = alpha
        self.current_alpha = alpha
        self.decay = decay
        self.iterations = 0
    def pre_update(self):
        if self.decay:
            self.current_alpha = self.alpha/(1+ self.decay*self.iterations)
    def update_parameters(self, layer):
        layer.w -= self.alpha * layer.dL_dw
        layer.b -= self.alpha * layer.dL_db        
    def post_update(self):
        self.iterations += 1
class get_data:
    def initialize(self, filename):
        def label_encoder(y):
            y_encoded = []
            for gesture in y:
                if(gesture == "open"):
                    y_encoded.append(0)
                elif(gesture == "close"):
                    y_encoded.append(1)
                elif(gesture == "fcuk_off"):
                    y_encoded.append(2)
                elif(gesture == "thumbs_up"):
                    y_encoded.append(3)
            return y_encoded

        with open(filename, "r") as f:
            data = json.load(f)
        for obesrvation in data:    
            palmpoint = obesrvation["landmarks"][0]
            x0 = palmpoint[0]
            y0 = palmpoint[1]
            z0 = palmpoint[2]

            anchor = [x0 , y0, z0]

            for coordinates in obesrvation["landmarks"]:
                for i in range (3):
                    coordinates[i] = coordinates[i] - anchor[i]

        X = [np.array(obs["landmarks"]).flatten() for obs in data]
        y = np.array([obs["gesture"] for obs in data])
        y_encoded = np.array(label_encoder(y))
        X = np.array(X)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        permutation = np.random.permutation(len(X))
        shuffled_X = X[permutation]
        shuffled_y = np.array(y_encoded[permutation])
        return shuffled_X,shuffled_y

        # return spiral_data(samples = 100, classes=3)