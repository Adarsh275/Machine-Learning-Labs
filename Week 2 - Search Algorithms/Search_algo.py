import heapq

# hepler functions are defined here

def DFS(Visited_Nodes,path,cost,goals,start):
    if start not in Visited_Nodes:
        Visited_Nodes.append(start)
        path.append(start)
        if start in goals:
            return -1
        else:
            for i in range(1,len(cost[start])):
                if(cost[start][i]!=-1 and cost[start]!=0):
                    check=DFS(Visited_Nodes,path,cost,goals,i)
                    if(check==-1):
                        return -1
            path.pop()


# A* Traversal Algorithm is given bellow.

def A_star_Traversal(cost, Heuristic, start_point, goals):
    
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        Heuristic: Heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """

    Visited_Nodes = [] # Visited node list array 
    path = [start_point]
    Queue = [[0+Heuristic[start_point], path]]
    while len(Queue) > 0: 
        temp = Queue.pop(0)
        current_path = temp[1]
        current_cost = temp[0]
        node = current_path[len(current_path)-1]
        current_cost = current_cost - Heuristic[node]
        if node in goals:
            return current_path
        Visited_Nodes.append(node)
        children=[]
        for i in range(len(cost)):
            if cost[node][i] not in [0, -1]:
                children.append(i)
        for i in children:
            new_current_path = current_path + [i]
            new_path_cost = current_cost + cost[node][i] + Heuristic[i]
            if i not in Visited_Nodes and new_current_path not in [i[1] for i in Queue]:
                Queue.append((new_path_cost, new_current_path))
                Queue = sorted(Queue, key=lambda x: (x[0], x[1]))
            elif new_current_path in [i[1] for i in Queue]:
                for index in range(len(Queue)):
                    if Queue[index][1] == path:
                       i=index
                       break
                Queue[i][0] = min(Queue[i][0], new_path_cost)
                Queue = sorted(Queue, key=lambda x: (x[0], x[1]))
    return list()
global path


# DFS traversal algorithm in given bellow

def DFS_Traversal(cost, start_point, goals):
   
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """

    path=[]
    Visited_Nodes=[]
    start = start_point
    DFS(Visited_Nodes,path,cost,goals,start)
    return path