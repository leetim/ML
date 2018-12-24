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

def cost_function(a):
    res = np.zeros(M)
    for j, p in enumerate(a):
        res[p] += jobs_time[j]*Mat[p][jobs[j]-1]
    return 10**5/np.max(res)

def cost_function_arr(a):
    # v = sorted([[cost_function(i), i] for i in a])
    # c = np.array([i for i, _ in v])
    # p = np.array([i for _, i in v])
    return np.array([cost_function(i) for i in a])

def mutation(a , cof = 0.05):
    r = np.random.rand(N)
    res = np.zeros(N, dtype=int)
    for i, _ in enumerate(a):
        if r[i] > cof:
            res[i] = (a[i]+np.random.randint(M))%M
        else:
            res[i] = a[i]
    return res

def easy_improve(a):
    cost = cost_function(a)
    b = a.copy()
    for i in range(N):
        for j in range(i+1, N):
            b[i], b[j] = b[j], b[i]
            if cost < cost_function(b):
                return b
            else:
                b[i], b[j] = b[j], b[i]
    print("No cool improve!")
    return a


def selection(a, b):
    r = np.random.rand(N)
    res = np.zeros(N, dtype=int)
    for i, _ in enumerate(a):
        if r[i] > 0.60:
            res[i] = a[i]
        else:
            res[i] = b[i]
    # print(res)
    # print("ind")
    return res

def get_random(arr):
    n = len(arr)
    r = np.random.rand()*arr[n-1]
    a, b = 0, n
    while b - a != 1:
        t = (a + b) // 2
        if arr[t] >= r:
            b = t
        else:
            a = t
    return a

def get_best(vals, pop):
    max_val = 0
    for i,v in enumerate(vals):
        if v > vals[max_val]:
            max_val = i
    print("Best number: {}".format(max_val))
    return pop[max_val]

def get_bedest(vals, pop):
    max_val = 0
    for i,v in enumerate(vals):
        if v < vals[max_val]:
            max_val = i
    # print(vals[max_val])
    return pop[max_val]

def genetic(start):
    pop = start
    n = len(start)
    for step in range(500):
        print(step)
        costs = cost_function_arr(pop)
        # print(get_best(costs, pop))
        print("Max: {}".format(np.max(costs)))
        print("Average: {}".format(np.average(costs)))
        print("Median: {}".format(np.median(costs)))
        print("Min: {}".format(np.min(costs)))
        vals = (costs*10/max(costs))**2
        # print(get_bedest(costs, pop))
        prob = np.zeros(n + 1)
        for i in range(n):
            prob[i+1] = prob[i] + vals[i]
        # print(prob)
        new_pop = [np.zeros((N, ), dtype=int) for i in range(n)]
        new_pop[0] = easy_improve(get_best(costs, pop))
        print(new_pop[0])
        for i in range(1, n//40):
            new_pop[i] = mutation(new_pop[0], 0.005)
        for i in range(n//40, n):
            i1, i2 = get_random(prob), get_random(prob)
            if costs[i1] > costs[i2]:
                new_pop[i] = mutation(selection(pop[i1], pop[i2]))
            else:
                new_pop[i] = mutation(selection(pop[i2], pop[i1]))
        pop = np.array(new_pop)
    return get_best(cost_function_arr(pop), pop)
        
pop_count = 5000
start_pop = np.random.randint(M, size=(pop_count, N))
print(start_pop)
print(genetic(start_pop))