import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            Hidden_States_Sequence: Most likely state sequence 
        """
        K = len(self.A)
        T = len(seq)
        T1 = np.empty((K, T))
        T2 = np.empty((K, T))

        T1[:, 0] = self.pi * self.B[:, self.emissions_dict[seq[0]]]
        T2[:, 0] = 0

        for t in range(1, T):
            for k in range(K):
                T1[k, t] = np.max(T1[:, t - 1] * self.A[:, k]) * self.B[k, self.emissions_dict[seq[t]]]
                T2[k, t] = np.argmax(T1[:, t - 1] * self.A[:, k])

        nu = np.max(T1[:, T - 1])
        Hidden_States_Sequence = np.empty(T, dtype=int)
        Hidden_States_Sequence[T - 1] = np.argmax(T1[:, T - 1])
        for t in range(T - 2, -1, -1):
            Hidden_States_Sequence[t] = T2[Hidden_States_Sequence[t + 1], t + 1]
        self.reverse_state_dict = dict(
            zip(self.states_dict.values(), self.states_dict.keys()))
        Hidden_States_Sequence = list(
            map(lambda x: self.reverse_state_dict[x], Hidden_States_Sequence))
        
        # print(Hidden_States_Sequence)
        return Hidden_States_Sequence