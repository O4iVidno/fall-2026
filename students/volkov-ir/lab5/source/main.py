import numpy as np
from data_preprocessing import prepare_data

X_train, X_test, y_train, y_test, feature_names = prepare_data()


def calculate_z(X, w):
    return X @ w

def calculate_margin(X, w, y):
    mult_vector = np.dot(X, w)
    return np.multiply(mult_vector, y)

def calculate_sigmoid(per):
    return 1 / (1 + np.exp(-per))

def calculate_loss(margin):
    # если m >= 0, то np.log(1 + np.exp(-margin))
    # если m < 0,  то -margin np.log(1 + np.exp(margin))
    return np.maximum(0, -margin) + np.log(1 + np.exp(-np.abs(margin)))

def calculate_Q(loss):
    return np.sum(loss)

def calculate_Qmean(loss):
    return np.sum(loss)/len(loss)

def calculate_gradient(X, y, sigmoid_m):
    a = (1 - sigmoid_m) * y
    gradient = -X.T @ a
    return gradient

def calculate_hessian(X, sigmoid_m):
    d = (1 - sigmoid_m) * sigmoid_m
    h = X.T * d @ X
    return h

def init_w_zero(X):
    return np.zeros(X.shape[1])


def method_newton_raph(X, y, w, epsilon):
    count_q = 0
    limit = 100
    w_list = []
    Q_list = []
    Q_old = np.inf

    while True:

        margin = calculate_margin(X, w, y)
        loss = calculate_loss(margin)
        Q = calculate_Q(loss)
        count_q += 1

        Q_list.append(Q)
        w_list.append(w)

        if abs(Q - Q_old) <= epsilon:
            break
        Q_old = Q

        if count_q >= limit:
            break

        sigmoid_m = calculate_sigmoid(margin)
        gra = calculate_gradient(X, y, sigmoid_m)
        hes = calculate_hessian(X, sigmoid_m)
        delta_w = np.linalg.inv(hes) @ gra
        w = w - delta_w

    return w, Q, w_list, Q_list, count_q

def init_w(X, y):
    w  = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return w

# # def method_irls(X, y, w, epsilon):
# #
# #     return w, Q, w_list, Q_list, count_q
#
# # X = np.array([[1, 2.5], [1, -1], [1, 0.5]])
# # y = np.array([1, 1, -1])
# X = np.array([[1, 0.8, -1.5], [1, -2.0, 0.6], [1, 3.0, 1.0], [1, 0.5, 2.0], [1, -1.0, -0.5], [1, 1.5, 0.0], [1, -0.3, 1.2]])
# y = np.array([-1, 1, 1, 1, 1, -1, -1])
# # w = init_w_zero(X)
# epsilon = 1e-9
#
# # w, Q, w_list, Q_list, count = method_newton_raph(X, y, w, epsilon)
# # print(w)
# # print(Q)
# # print(Q_list)
# # print(w_list)
# # print(count)
#
# # z = calculate_z(X, w)
# # margin = calculate_margin(X, w, y)
# # sigmoid_z = calculate_sigmoid(z)
# # sigmoid_m = calculate_sigmoid(margin)
# # loss = calculate_loss(margin)
# # q = calculate_Q(loss)
# # qmean = calculate_Qmean(loss)
# # gra = calculate_gradient(X, y, sigmoid_m)
# # hes = calculate_hessian(X, sigmoid_m)
# #
# # print(z)
# # print(margin)
# # print(sigmoid_z)
# # print(sigmoid_m)
# # print(loss)
# # print(q)
# # print(qmean)
# # print(gra)
# # print(hes)
# w = init_w(X, y)
# print(w)
# sigma_m = calculate_sigmoid(calculate_margin(X, w, y))
# gamma = np.sqrt((1-sigma_m)*sigma_m)
# print(gamma)