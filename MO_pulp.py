from numpy import array, arange
import json
import copy
import matplotlib.pyplot as plt
import pulp


def gauss(A):
    n = len(A)

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n + 1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = array([0 for i in range(n)], dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x


def get_initials():
    file = open('in.json', encoding='utf-8')
    data = file.read()
    json_data = json.loads(data)
    c = array([array(json_data['критерии'][criteria]) for criteria in json_data['критерии']])
    b = array([*json_data['ограничения']['<='], *-array(json_data['ограничения']['>='], dtype=float)], dtype=int)
    methods = array(json_data['methods'])
    c_names = array([criteria for criteria in json_data['критерии']])
    new_c = []
    for i, criteria in enumerate(c):
        tmp = []
        for j in criteria:
            tmp.append(' '.join(str(i) for i in j))
        tmp = ' '.join(tmp).split()
        new_c.append(array(tmp, dtype=int))
    c = array(new_c)
    n = len(json_data['ограничения']['<='])
    m = len(json_data['ограничения']['>='])
    a = []
    for i in range(0, n * m, m):
        if i // m == 0:
            z = [1 for _ in range(m)]
            z.extend([0 for _ in range(n * m - m)])
        elif i // m == n - 1:
            z = [0 for _ in range(n * m - m)]
            z.extend([1 for _ in range(m)])
        else:
            z = [1 for _ in range(m)]
            z = [*[0 for _ in range(i)], *z, *[0 for _ in range(n * m - m - i)]]
        a.append(z)
    for i in range(0, n * m, n):
        z = [0 for _ in range(n * m)]
        for j in range(i // n, n * m, m):
            z[j] = 1
        a.append(z)
    a = array([array(item) for item in a])
    for i, item in enumerate(a):
        if b[i] < 0:
            a[i] = -item
    return a, b, c, c_names, methods, n, m


def partial_optimization(a, b, c, c_names, methods, n, m):
    assignments = [(i,) for i in range(n * m)]
    optimization_matrix = []
    for i, coefficient in enumerate(c):
        model = pulp.LpProblem(c_names[i], pulp.LpMinimize)
        x = pulp.LpVariable.dicts('x', assignments, lowBound=0, cat=pulp.LpInteger)
        # Objective Function
        if methods[i] == "max":
            coefficient = -coefficient
        model += (pulp.lpSum([coefficient[k] * x[(k,)] for k in range(n * m)]))
        # <=
        for k in range(len(b)):
            model += (pulp.lpSum([a[k][j] * x[(j,)] for j in range(n * m)])) <= b[k]

        model.solve()
        real_x = array([x[var].varValue for var in x], dtype=int)
        arr = []
        for j, val in enumerate(c):
            # max
            arr.append(val.dot(real_x) if methods[j] == "max" else -val.dot(real_x))
        optimization_matrix.append(array(arr))
    return array(optimization_matrix)


def global_optimization(b, coefficients, methods, weight):
    n_m = len(coefficients[0])
    assignments = [(i,) for i in range(n_m)]
    c = []
    for i, coefficient in enumerate(coefficients):
        if methods[i] == "min":
            c.append(-coefficient)
        else:
            c.append(coefficient)
    c_glob = array(array([weight[i] * c[i] for i in range(len(weight))]).sum(axis=0))
    model = pulp.LpProblem('Q_gl', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', assignments, lowBound=0, cat=pulp.LpInteger)
    # Objective Function
    model += (pulp.lpSum([c_glob[k] * x[(k,)] for k in range(n_m)]))
    # <=
    for k in range(len(b)):
        model += (pulp.lpSum([a[k][j] * x[(j,)] for j in range(n * m)])) <= b[k]
    model.solve()
    real_x = array([x[var].varValue for var in x], dtype=int)
    return c_glob, real_x


def normalization(optimization_matrix):
    norm_matrix = array([array(row, dtype=float) for row in optimization_matrix])
    for i in range(len(optimization_matrix)):
        min_i = min(optimization_matrix[i])
        max_i = max(optimization_matrix[i])
        for j in range(len(optimization_matrix[i])):
            norm_matrix[i][j] = (norm_matrix[i][j] - min_i) / (max_i - min_i)
    return norm_matrix


def weight_def(norm_matrix):
    alpha = [array([norm_matrix[i][j] for j in range(len(norm_matrix[i])) if j != i]).mean() for i in
             range(len(norm_matrix))]
    a_eq = []
    for i in range(len(alpha) - 1):
        al = [1 - alpha[i + 1], -(1 - alpha[i])]
        z = [*[0 for _ in range(i)], *al, *[0 for _ in range(len(alpha) - i - 2)], 0]
        a_eq.append(z)
    a_eq.append([*[1 for _ in range(len(alpha))], 0])
    a_eq = array(a_eq)
    b_eq = array([*[0 for _ in range(len(alpha) - 1)], 1])
    for i in range(len(a_eq)):
        a_eq[i][-1] = b_eq[i]
    return gauss(a_eq)


def plot(res, c_names, verbose):
    ind = arange(len(res))
    width = 0.35
    fig = plt.figure(dpi=200)
    plt.ylabel('руб.')
    plt.xticks(ind, c_names)
    plt.bar(ind, res, width)
    fig.savefig('{}.png'.format(verbose))
    # plt.show()
    plt.close()


def print_matrix(A):
    if A.dtype == int:
        return '\n'.join(['\t'.join(['{}'.format(item) for item in row]) for row in A])
    if A.dtype == float:
        return '\n'.join(['\t'.join(['{:<10.6}'.format(item) for item in row]) for row in A])


def print_vec(vec):
    if vec.dtype == float:
        return ' '.join(['{:.3}'.format(item) for item in vec])
    if vec.dtype == int:
        return ' '.join(['{}'.format(item) for item in vec])


def centered(txt):
    return format(txt, '=^100')


def iteration(a, b, c, c_names, methods, n, m, i):
    optimization_matrix = partial_optimization(a, b, c, c_names, methods, n, m)
    print(centered(f'Итерация {i}'))
    print(centered('Матрица оптимизации'))
    print(print_matrix(optimization_matrix.transpose()))
    norm_matrix = normalization(optimization_matrix.transpose())
    print(centered('Нормированная матрица'))
    print(print_matrix(norm_matrix))
    weight = weight_def(norm_matrix)
    print('λ =', print_vec(weight))
    c_glob, x = global_optimization(b, c, methods, weight)
    print(centered('Коэффициенты глобального критерия'))
    print(print_vec(c_glob))
    res_1 = array([item.dot(x) for item in c])
    print(centered('Результат оптимизации глобального критерия'))
    print('xg = ', print_vec(x))
    print(', '.join([f'{c_names[i]} = {res_1[i]}' for i in range(n)]))
    plot(res_1, c_names, i)


a, b, c, c_names, methods, n, m = get_initials()
iteration(a, b, c, c_names, methods, n, m, 1)
# q3 >=450
tmp_a = [-c[2]]
tmp_b = [-450]
a = array([*a, *tmp_a])
b = array([*b, *tmp_b])
iteration(a, b, c, c_names, methods, n, m, 2)
