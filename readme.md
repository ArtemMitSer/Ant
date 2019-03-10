import numpy as np

#Инициализация класса муравьев 
class Ant:
    def __init__(self, start_vertex):
        self.start_vertex = start_vertex
        self.vertexs = [self.start_vertex]
        self.L = 0
        
#Описание метода run. Правила движения муравьев по графу. 
#Движение муравьев по графу, рассчет вероятностей перехода на следующую вершину    
    def run(self, matrix, pheromons, alpha, beta):
        current_vertex = self.start_vertex  
        for _ in range(matrix.shape[0] - 1):
            list_p = []
            list_vertex = []
            for vertex in range(matrix.shape[0]):
                if vertex in self.vertexs:
                    continue
                t = pheromons[current_vertex][vertex]
                d = matrix[current_vertex][vertex]
                p = (t ** alpha) * ((1 / d) ** beta)
                list_p.append(p)
                list_vertex.append(vertex)
            list_p = np.array(list_p)
            list_p = list_p / sum(list_p)
            list_p = np.cumsum(list_p)
            random_number = np.random.random()
            for num, border in zip(list_vertex, list_p):
                if random_number < border:
                    next_vertex = num
                    break
            self.L += matrix[current_vertex][next_vertex]

            current_vertex = next_vertex
            self.vertexs.append(current_vertex)
        self.L += matrix[self.vertexs[-1]][self.vertexs[0]]
#Инициализация класса таблицы         
class Graph:
    def __init__(self, matrix, num_ants, alpha, beta, count_epoch, p):
        self.alpha = alpha
        self.beta = beta
        self.matrix = matrix
        self.pheromons = np.abs(np.random.normal(0, 0.001, matrix.shape))
        self.num_vertix = matrix.shape[0]
        self.num_ants = num_ants
        self.count_epoch = count_epoch
        self.p = p
        self.best_L = 999999

    def solve(self):
        for _ in range(self.count_epoch):
            self.ants = []
            for i in range(self.num_ants):
                random_start_vertix = np.random.randint(0, self.num_vertix)
                self.ants.append(Ant(start_vertex=random_start_vertix))

            for ant in self.ants:
                ant.run(matrix=self.matrix, pheromons=self.pheromons, alpha=self.alpha, beta=self.beta)
#Обновление форомонов 
            set_eager = set()
            for ant in self.ants:
                vertexs = ant.vertexs
                for i in range(1, self.num_vertix):
                    current_vertex, next_vertex = vertexs[i - 1], vertexs[i]
                    first_eager = (current_vertex, next_vertex)
                    second_eager = (next_vertex, current_vertex)
                    if first_eager not in set_eager and second_eager not in set_eager:

                        set_eager.add(first_eager)
                        set_eager.add(second_eager)
                        self.pheromons[next_vertex][current_vertex] *= (1 - self.p)
                        self.pheromons[current_vertex][next_vertex] *= (1 - self.p)

                    self.pheromons[next_vertex][current_vertex] += (1 / ant.L)
                    self.pheromons[current_vertex][next_vertex] += (1 / ant.L)
                if self.best_L > ant.L:
                    self.best_L = ant.L
        return self.best_L

#Считывание входных данных
a = input()
arr = list(map(int, a.split()))
matrix = arr
data = []
data.append(matrix)

for i in range((len(matrix))-1):
    a = input()
    arr = list(map(int, a.split()))
    data.append(arr)
data = np.array(data)

graph = Graph(matrix=data, num_ants=20, alpha=1, beta=1, count_epoch=1500, p=0.5)
print(graph.solve())
