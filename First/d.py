import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

# #1
# A = np.array([[7,2],[3,4],[5,3]])
# U,D,V = np.linalg.svd(A)
#
# print(U,"1")
# print(np.linalg.eig(A.dot(A.T))[1],"1.1")
# print(V,"2")
# print(np.linalg.eig(A.T.dot(A))[1],"2.1")
# print(D,"3")



# #2
# img = cv.imread("c.jpg",0)
# print(img)
# img = cv.resize(img,(721,833))
# # print(img)
# cv.imshow("a",img)
#
#
# # print(np.min(img))
# A = np.array(img)
# #print(A)
# U,D,V = np.linalg.svd(img)
# #print(U)
# # print(U,"1")
# # print(np.linalg.eig(A.dot(A.T))[1],"1.1")
# # print(V,"2")
# # print(np.linalg.eig(A.T.dot(A))[1],"2.1")
# # print(D.size,"3")
# for i in [151]:
#     print(U[:,:i],"U")
#     print(D[:i],"D")
#     print(V[:i,:],"V")
#     reconstimg = np.matrix(U[:,:i])*np.diag(D[:i])*np.matrix(V[:i,:])
#     print(reconstimg)
#     cv.imshow("b",reconstimg)
#
# if cv.waitKey(0) == 27:
#     cv.destroyAllWindows()



np.random.seed(123)
x = 5*np.random.rand(100)
y = 0.5*x+5+np.random.randn(100)

x=x.reshape(100,1)
y=y.reshape(100,1)

A = np.hstack((x,np.ones(np.shape(x))))

A_plus = np.linalg.pinv(A)
coefs = A_plus.dot(y)

x_line = np.linspace(0, 5, 1000)
y_line = coefs[0]*x_line + coefs[1]

plt.plot(x, y, "+")
plt.plot(x_line,y_line)
plt.show()