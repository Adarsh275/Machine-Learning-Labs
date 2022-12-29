from ast import For
from math import dist
from operator import le
import re
import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        answer = []
        num_items= len(x)
        mm= len(self.data)
        space= len(x[0])

        for i in range(num_items):
            row = []
            for j in range(mm):
                total_sum = 0
                for k in range(space):
                    total_sum = total_sum + abs(x[i][k] - self.data[j][k])**self.p
                dist = total_sum**(1/self.p)
                row.append(dist)
            answer.append(row)

        return answer            

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neighbour_distance, index_of_Neighbours)
            neighbour_distance -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            index_of_Neighbours -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neighbour_distance and index_of_Neighbours must be SORTED in increasing order of distance
        """
        neighbour_distance = []
        index_of_Neighbours = []
        Space_between = self.find_distance(x)

        num_items = len(x)
        m = len(self.data)

        for i in range(num_items):
            row = [(Space_between[i][j], j) for j in range(m)]
            row.sort()
            n_row = [row[j][0] for j in range(m)]
            i_row = [row[j][1] for j in range(m)]
            neighbour_distance.append(n_row[:self.k_neigh])
            index_of_Neighbours.append(i_row[:self.k_neigh])

        return (neighbour_distance, index_of_Neighbours)

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted_val target value for each input)(int)
        """
        frequencies = []
        k = self.k_neighbours(x)
        for j in range(len(x)):
            frequency = {}
            for i in range(self.k_neigh):
                d = k[0][j][i]
                index = k[1][j][i]
                t = self.target[index]
                if t not in frequency.keys():
                    if self.weighted:
                        frequency[t] = 1/d
                    else:
                        frequency[t] = 1
                else:
                    if self.weighted:
                        frequency[t] = frequency[t]+1/d
                    else:
                        frequency[t] = frequency[t]+ 1

            frequencies.append(frequency)

        predicted_val = []
        for frequency in frequencies:
            prediction = self.target[0]
            max_f = frequency[prediction]
            for t in frequency.keys():
                if frequency[t] > max_f:
                    prediction = t 
            predicted_val.append(prediction)

        return predicted_val

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: precision metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            precision : (float.)
        """
        precision = 0 
        p = self.predict(x)
        n = len(y)
        for i in range(n):
            if p[i] == y[i]:
                precision  = precision + 1
        precision = precision * 100/n
        return precision