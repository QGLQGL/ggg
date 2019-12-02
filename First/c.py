import numpy as np
import matplotlib.pyplot as plt

# a = np.array([[2,-1],[3.5,-1]])
# a_inv = np.linalg.inv(a)
# a_bis = a_inv.dot(a)
# a_bis = np.rint(a_bis)
# a_eigenvalues = np.linalg.eigvals(a)
# # print(a_eigenvalues,"....")
# # print(a_bis)
#
# diag = np.diag(a_eigenvalues)
#
#
#
#
# v = list(np.linalg.eig(a))[1]
# v_inv = np.linalg.inv(v)
#
# c = v.dot(diag).dot(v_inv)
# print(c)

t = np.linspace(0,2*np.pi,100)
x = np.cos(t)
y = np.sin(t)

a = np.array([x,y])
b = np.array([[1,-1],[-1,4]])
c = np.matmul(b,a)

plt.figure()
plt.plot(x,y)
plt.plot(c[0],c[1])
plt.xlim(-6,6)
plt.ylim(-6,6)
plt.show()





#2
def plotvector(v=None,v1=None):
    x = np.linspace(0,v[0],10)
    y = np.linspace(0,v[1],10)

    x1 = np.linspace(0,v1[0])
    y1 = np.linspace(0, v1[1])

    plt.figure()
    plt.plot(x,y)
    plt.plot(x1,y1)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()
#
#
#
# #3
# v = [1,1]
# a = np.array([[0,-1],[1,0]])
# b = np.matmul(a,v)
# plotvector(v,b)



#
# def a(n):
#     t = np.linspace(0, 2 * np.pi, n)
#     x = np.cos(t)
#     y = np.sin(t)
#
#
#     a = np.array([x, y])
#     print(a)
#     # b = np.array([[1, -1], [-1, 4]])
#     # c = np.matmul(b, a)
#     plt.figure()
#     # plt.plot(x1,x2)
#     # plt.xlim(-2, 2)
#     # plt.ylim(-2, 2)
#     # plt.show()
#     for i in range(n):
#         x1 = np.linspace(0, a[0][i], n)
#         x2 = np.linspace(0, a[1][i], n)
#         # print(a[0][i], a[1][i])
#         plt.plot(x1, x2)
#         plt.xlim(-2, 2)
#         plt.ylim(-2, 2)
#     plt.show()
#
# a(200)
