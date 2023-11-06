import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

class Hard_margin:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = None
        self.multipliers = None
        self.support_vectors = None
        self.support_vectors_y = None
        self.w = None
        self.b = None
        self.C = 100

    def compute_w(self, multipliers, X, y):
        return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

    def compute_b(self, w, X, y):
        return np.sum([y[i] - np.dot(w, X[i]) for i in range(len(X))]) / len(X)

    def fit(self, X, y):
        X0 = np.array([X[i] for i in range(len(X)) if y[0][i] == 1])
        X1 = np.array([X[i] for i in range(len(X)) if y[0][i] == -1])
        N = len(X1)

        if len(X0) != len(X1):
          print('khong chay dau')
        
        V = np.concatenate((X0.T, -X1.T), axis = 1)
        #V = new_x
        K = matrix(V.T.dot(V))

        p = matrix(-np.ones((int(2*N), 1)))
        # build A, b, G, h
        G = matrix(np.vstack((-np.eye(int(2*N)), np.eye(int(2*N)))))

        h = matrix(np.vstack((np.zeros((int(2*N), 1)), self.C*np.ones((int(2*N), 1)))))
        A = matrix(y.reshape((-1, int(2*N))))
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)

        
        l = np.array(sol['x'])
        new_X = np.concatenate((X0.T, X1.T), axis = 1)
        S = np.where(l > 1e-5)[0]  
        S2 = np.where(l < .999*self.C)[0] 
        M = [val for val in S if val in S2] 
        XT = new_X.T 
        VS = V[:, S]
        lS = l[S]
        yM = y[:, M]
        XM = XT[M]
        w_dual = VS.dot(lS).reshape(-1, 1)
        b_dual = np.mean(yM.T - w_dual.T.dot(XM.T))
        
        print(w_dual.T, b_dual) 
        return w_dual, b_dual



# p = [[184.62322072223486, 220.38080448716858, 1], [220.16729547132954, 162.5243463941023, 1], [239.53943336756996, 199.4205020060454, 1], [242.6939562743389, 169.02194224275686, 1], [241.0544238209796, 181.91407858143296, 1], [184.4446666369114, 205.0110770160632, 1], [226.20029382924412, 221.60889553044075, 1], [235.61751121964852, 191.67323290299336, 1], [196.9518447110314, 209.23657021678665, 1], [167.74155329956687, 227.75091977287357, 1], [204, 190, 1], [369.46965141005404, 366.65823793534753, -1], [348.1962081320871, 373.601895102735, -1], [379.0406827762358, 398.2707257442146, -1], [341.90389464115947, 372.57542965681444, -1], [350.8751884630671, 333.4996347405597, -1], [377.2190607025304, 355.89651273811995, -1], [317.02151194525857, 399.9282560365913, -1], [307.9659655005524, 331.9294626372366, -1], [366.219176576544, 383.95759134081135, -1], [350.3257170908266, 394.76107602182606, -1], [344, 366, -1]]

# def process(X_):
#     X = [i[:-1] for i in X_]
#     y = [i[-1] for i in X_]
#     y = np.array([y]).T.astype('float')
#     X = np.array(X).astype('float')
#     return X, y

# def visualize(X):
#     plt.scatter(X[:,0], X[:,1])
    
# def draw_line(w, b):
#     X = []
#     Y = []
#     for x in range(1094, 15, -1):
#         y = (w[0][0]*x + b)/(-w[1][0])
#         if y <= 16 or y >= 564:
#             continue
#         X.append(x)
#         Y.append(y)
#     plt.plot(X, Y)

# X, y = process(p)
# visualize(X)

# svm = Hard_margin()
# w, b = svm.fit(X, y)

# print(w, b)
# draw_line(w, b)

# plt.show()