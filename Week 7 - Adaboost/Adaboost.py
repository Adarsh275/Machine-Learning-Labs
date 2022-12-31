import numpy as np
from sklearn.tree import DecisionTreeClassifier
from numpy.random.mtrand import sample

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        Sample_Weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):
            skt = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            skt.fit(X, y, Sample_Weights)
            y_pred = skt.predict(X)

            self.stumps.append(skt)
            
            error = self.stump_error(y, y_pred, Sample_Weights=Sample_Weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            Sample_Weights = self.update_weights(
                y, y_pred, Sample_Weights, alpha)

        return self

    def stump_error(self, y, y_pred, Sample_Weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            Sample_Weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """

        labels = (y != y_pred).astype(int)
        error = np.dot(Sample_Weights, labels)/np.sum(Sample_Weights)
        return error

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        alpha = 0.5 * (np.log((1 - error + eps) / (error + eps)))  # check this
        return alpha

    def update_weights(self, y, y_pred, Sample_Weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            Sample_Weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """

        error = self.stump_error(y, y_pred, Sample_Weights)
        # N = 2 * (error/(1-error)) ** 0.5
        n_samples = len(Sample_Weights)
        for i in range(n_samples):
            # scaling_factor = Sample_Weights[i]/N
            if y[i] == y_pred[i]:
                # print('here')
                Sample_Weights[i] = 0.5 * Sample_Weights[i] / \
                    (1 - error)  # scaling_factor * np.exp(-alpha)
            else:
                # print('should not be here')
                Sample_Weights[i] = 0.5 * Sample_Weights[i] / \
                    (error)  # scaling_factor * np.exp(alpha)
        Sample_Weights = np.asarray(Sample_Weights) / np.sum(Sample_Weights)
        return Sample_Weights

    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        for i in range(self.n_stumps):
            predictions = [self.stumps[i].predict(X)]

        return predictions

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy