import numpy as np
import matplotlib.pyplot as plt


class SVM_HingeLoss:
    def __init__(self):
        self.C = 0.5
        self.lr = 0.0001
        self.epochs = 10000

    def hingeloss(self, x, y, w, b):
        u = y * ((np.dot(w, x.T)) + b)
        loss =  0.5 * np.dot(w, w.T) + self.C * np.maximum(0, 1-u)
        return loss[0][0]

    def fit(self, X, Y, w, b):
        '''
        X: Nx2    Y: 1xN
        w: 2x1    b: scalar
        '''
        w_list = []
        b_list = []
        w = w.T
        for i in range(self.epochs):
            u = Y * (np.dot(w, X.T) + b)
            H = np.where(u < 1)[1]

            Z = Y.T * X
            ZH = Z[H, :]
            gradw = w + self.C * np.sum(-ZH, axis= 0)

            YH = Y[:,H]
            gradb = self.C * np.sum(-YH, axis= 1)

            w = w - self.lr * gradw
            b = b - self.lr * gradb

            if i % int(self.epochs/100) == 0:
                w_list.append(w.T)
                b_list.append(b[0])

            loss = self.hingeloss(X, Y, w, b)
            if loss < 1e-3:
                break
        return w_list, b_list, loss

class SVM_HingeLoss_MiniBatch:
    def __init__(self):
        self.C = 0.5
        self.lr = 0.0001
        self.epochs = 10000

    def hingeloss(self, x, y, w, b):
        u = y * ((np.dot(w, x.T)) + b)
        loss =  0.5 * np.dot(w, w.T) + self.C * np.maximum(0, 1-u)
        return loss[0][0]

    def fit(self, X_, Y_, w, b, batch_size= 10):
        '''
        X: Nx2    Y: 1xN
        w: 2x1    b: scalar
        '''
        w_list = []
        b_list = []
        w = w.T
        for i in range(self.epochs):
            for batch in range(0, len(X_), batch_size):
                X = X_[batch:batch + batch_size, :]
                Y = Y_[:, batch:batch + batch_size]
                u = Y * (np.dot(w, X.T) + b)
                H = np.where(u < 1)[1]

                Z = Y.T * X
                ZH = Z[H, :]
                gradw = w + self.C * np.sum(-ZH, axis= 0)

                YH = Y[:,H]
                gradb = self.C * np.sum(-YH, axis= 1)

                w = w - self.lr * gradw
                b = b - self.lr * gradb

            if i % int(self.epochs/100) == 0:
                w_list.append(w.T)
                b_list.append(b[0])

            loss = self.hingeloss(X, Y, w, b)
            if loss < 1e-5:
                break
        return w_list, b_list, loss
    

# p = [[255.1842428762551, 493.85373114635837, 1], [234.92390215178983, 497.55488487287374, 1], [259.2816780522734, 459.67731232831807, 1], [261.3751127062345, 458.92424845551886, 1], [259.7719013303774, 487.3016793863483, 1], [293.70153372144887, 449.38714438872967, 1], [304.5201231319043, 476.58326158217386, 1], [275.7879310852492, 507.10532402620333, 1], [280.7318021791563, 491.20303570873335, 1], [286.970175394945, 450.91872493328486, 1], [272, 470, 1], [267.2855803493185, 377.21436990114375, 1], [252.5468493093637, 433.75162560772117, 1], [249.83339643745236, 390.53850452380954, 1], [303.32617317872786, 415.79303691574904, 1], [261.14336950484113, 375.55577390066355, 1], [293.7334795856815, 384.8442155725713, 1], [304.42988323304394, 427.7745458942412, 1], [278.5542108554132, 450.1479697483741, 1], [250.58125302433726, 394.89356419298093, 1], [286.05171535589477, 415.23649676233185, 1], [281, 414, 1], [449.21986468222815, 529.5384560997381, -1], [429.64274289544323, 501.20369677596136, -1], [386.70678806022175, 504.7838378972825, -1], [401.81021237848176, 522.571301860843, -1], [377.291401949074, 538.0083946080483, -1], [451.5914681157879, 474.1010611017623, -1], [419.7225358047715, 505.8754765407793, -1], [415.0632845620578, 463.8296150765463, -1], [420.2723312626953, 526.9918814775034, -1], [446.0546478063318, 536.761796087821, -1], [415, 500, -1], [367.46679813100434, 403.53854724539815, -1], [398.38193172138637, 442.8703069492127, -1], [423.066318472356, 415.45308499387903, -1], [388.8969910639517, 389.85442716057855, -1], [407.4629649654038, 399.2401550858695, -1], [430.1921517864124, 379.24517097635, -1], [402.28055951238053, 403.47625756549223, -1], [397.3592118717166, 419.63981711524144, -1], [435.8282249972467, 451.22350502258814, -1], [405.39995176055487, 439.0878121249393, -1], [405, 419, -1]]

def process(X_):
    X = [i[:-1] for i in X_]
    y = [i[-1] for i in X_]
    y = np.array([y]).astype('float')
    X = np.array(X).astype('float')
    return X, y

# def visualize(X):
#     plt.scatter(X[:,0], X[:,1])
    
# def draw_line(w, b):
#     X = []
#     Y = []
#     for x in range(1094, 15, -1):
#         y = (w[0][0]*x + b)/(-w[1][0])
#         X.append(x)
#         Y.append(y)
#     plt.plot(X, Y)

# X, y = process(p)
# visualize(X)

# model = SVM_HingeLoss()
# w0 = np.random.uniform(-1,1,(2, 1))
# b0 = np.random.uniform(-1,1) * 100
# w, b, losses = model.fit(X, y, w0, b0)
# draw_line(w[-1], b[-1])
# plt.show()