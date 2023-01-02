import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model
    
    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta


    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])


    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self


    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        gk=[]

        for i in range(len(data)):
            for j in range(len(self.centroids)):
                gk.append(np.linalg.norm(self.centroids[j]-data[i]))
        
        Cluster=[]

        d=np.reshape(gk,(len(data),len(self.centroids)))

        for i in range(len(d)):
            Cluster.append(np.argmin(d[i])) 
        return Cluster


    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        radius=len(self.centroids)
        center=len(self.centroids[0])
        New_Center=np.zeros(shape=(radius,center))

        j=0  #data ittrarion 
        for i in cluster_assgn:
            New_Center[i]=np.add(New_Center[i],data[j])
            j = j+1
        cluster_assgn=np.array(cluster_assgn)

        for k in range(len(New_Center)):
            count=(cluster_assgn==k).sum()
            New_Center[k]=New_Center[k]*(1/count)  

        self.centroids=New_Center


    def evaluate(self, data):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        Dist=[]
        m = len(data)
        n = len(self.centroids)
        for i in range(m):
            for j in range(n):
                Dist.append(np.square(self.centroids[j]-data[i]))
        Dist = np.sum(Dist, axis=1)
        se = 0
        for i in Dist:
            se = se + i
        return se