import numpy as np

# #1`
# a = np.zeros(10)
# print(a)
# a[4] = 1
# print(a)


# #2
# b = np.random.rand(5,3)
# c = np.random.rand(3,2)
# print(b)
# print("------------------------")
# print(c)
# print("------------------------")
# d = np.matmul(b,c)
# print(d)



#3 升序
# a = np.random.rand(int(input("输入序列的长度:")))
# # print(a)
# a = a*10
# a = np.fix(a).astype(int)
# print(a)
# temp = 0
# for i in range(0,a.size-1):
#     for j in range(0,a.size-1-i):
#         if(a[j]>a[j+1]):
#             # temp = a[j]
#             # a[j] = a[j+1]
#             # a[j+1] = temp
#             a[j],a[j+1] = a[j+1],a[j]
#
# print(a)



# #4
# a = [1,2,0,0,4,0]
# a = np.array(a)
# print(np.where(a==0))




# #5
# a = np.random.rand(10,10)*10
# a = np.fix(a).astype(int)
# #每行最大值
# print(np.max(a,0))
# #每列最大值
# print(np.max(a,1))
# #整个矩阵中的最大值
# print(np.max(a))




#
a = [[3,3.2],[3.5,3.6]]
b = [[118.4],[135.2]]
a = np.array(a)
a = np.linalg.inv(a)
b = np.array(b)
print(np.matmul(a,b))