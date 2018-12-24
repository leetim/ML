from queue import *
import numpy as np

mapl = lambda f, a: np.array(list(map(f, a)))
data = open("data.txt").read().split()
N, data = data[0], data[1:]
N = int(N)
jobs = mapl(int, data[:N])
jobs_time = mapl(float, data[N:2*N])
M, Mat = data[2*N], data[2*N+1:]
M = int(M)
Mat = np.array([mapl(float, Mat[i*4:(i+1)*4])for i in range(M)])
print(N, M)
print(jobs)
print(jobs_time)
print(Mat)

################################################################################
inf = 10**13
# in_data = list(open("input.txt").read().split())
# out = open("output.txt", "w")
# k = int(in_data[0])
# words = list(enumerate(in_data[1:], start = 1))
# # keys = "".join(words)
# keys = list({i for i in "".join(in_data[1:])})
# keys = list(enumerate(keys, start = len(words)+1))
################################################################################
n = N + M + 2
start = 0
finish = n - 1
phi = [0 for i in range(n)]
phi[finish] = 0

A = list()
for j, jv in enumerate(jobs):
    for p_index in range(M):
        val = jobs_time[j]*Mat[p_index][jv - 1]
        A.append([1 + j, 1 + N + p_index, 1.0/val, len(A)+1, 1])
for i in range(N):
    A.append([start, 1 + i, 0, len(A)+1, 1])
for i in range(M):
    A.append([1 + N + i, finish, 0, len(A)+1, 1000])

################################################################################
class Edge:
    def __init__(self, src, to, cost, flow, max_capacity, ind):
        self.src = src
        self.to = to
        self.cost = cost
        self.new_cost = cost
        self.flow = flow
        self.ind = ind
        self.max_capacity = max_capacity
    def get_cap(self):
        return self.max_capacity - self.flow
    def get_cost(self):
        return self.cost + phi[self.to] - phi[self.src]
    def print_edge(self):
        return ("(%d, %d, %d, %d)"%(self.src, self.to, self.ind, self.flow))
    def __lt__(self, other):
        return self.get_cost() < other.get_cost()

################################################################################
E = [list() for i in range(n)]
E_back = [list() for i in range(n)]
for t in A:
    E[t[0]].append(Edge(t[0], t[1], t[2], 0, t[4], t[3]))
    E[t[1]].append(Edge(t[1], t[0], -t[2], 0, 0, -t[3]))
for i in E:
    for j in i:
        E_back[j.to].append(j)
def get_back(e):
    for eb in E[e.to]:
        if eb.ind == -e.ind and eb.src == e.to:
            return eb
    return None

################################################################################
def dijkstr():
    visited = set()
    verts = [[inf, i, None] for i in range(n)]
    verts[finish][0] = 0
    q = PriorityQueue()
    q.put(verts[finish].copy())
    while not q.empty():
        val, t, e = q.get()
        if t in visited:
            continue
        visited.add(t)
        for eb in E_back[t]:
            if eb.get_cap() > 0:
                if verts[eb.src][0] > verts[eb.to][0] + eb.get_cost():
                    verts[eb.src] = [verts[eb.to][0] + eb.get_cost(), eb.src, eb]
                    q.put(verts[eb.src].copy())
    if verts[start][0] == inf:
        return 0
    for i in range(n):
        phi[i] += verts[i][0]
    cur = start
    way = list()
    while cur != finish:
        val, t, e = verts[cur]
        e.flow += 1
        be = get_back(e)
        be.flow -= 1
        cur = e.to
    return 1

################################################################################
temp = 1
while temp > 0:
    temp = dijkstr()
flow_val = sum([j.flow for j in filter(lambda x: x.flow > 0, E[start])])

################################################################################
way = list()
visited = set()
def get_way(v = start):
    if v == start:
        way.clear()
        visited.clear()
    if v == finish:
        return True
    if v in visited:
        return False
    visited.add(v)
    for e in E[v]:
        if e.flow > 0:
            if get_way(e.to):
                way.append(e)
                return True

################################################################################
ways = list()
for i in range(flow_val):
    get_way()
    ways.append(way[::-1])
    for i in way:
        i.flow -= 1
# res = [i[1] for i in words]
print(ways)
pairs = [(e.src, e.to - N) for _, e, _ in ways]
sorted(pairs)
print( np.array([i for _, i in pairs]))# for L in ways:
#     e = L[1]
#     ch = keys[e.to-1-len(words)][1]
#     word = words[e.src-1][1]
#     pos = word.find(ch)
#     res[e.src-1] = "%s&%s"%(word[:pos], word[pos:])
# for i in res:
#     out.write("%s\n"%i)
