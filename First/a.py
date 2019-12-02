'''test'''
import math

"""while计算1+。。。+100"""
def add():
    a = 100
    sum = 0
    while (a):
        sum = sum + a
        a = a - 1
    return sum

"""判别闰年"""
def leapyear(year):
    if year%4 == 0 and (year%100==0 and year%400==0):
        print(year,"is leap year.")
    else:
        print(year,"is not leap year.")

"""输出金字塔"""
def showGraph():
    a = 9
    b = 1
    c = 4
    while(a):
        if(b <= 5):
            print("*"*b)
            b=b+1
        if b>5 and c > 0:
            print("*" * c)
            c = c - 1
        a = a-1

'''比较两个数大小并升序打印'''
def sort():
    a = input("please input numble1:")
    b = input("please input number2:")
    if(a>=b):
        print(b,a)
    else:
        print(a,b)

'''给定数字集合{1，2，3，4} 计算满足下述条件的三位数的个数
条件1：组成的三位数互异
条件2：每个三位数中无重复数字'''
def a():
    count = 0
    list = [1,2,3,4]
    for i in list:
        for j in list:
            for k in list:
                if i!=j and j!=k and k!=i:
                    print(i*100+j*10+k)
                    count = count+1
    print("满足条件的数共有",count,"个。")


def findnum():
    x=1
    while(x):
        y = int(math.sqrt(x+100))
        z = int(math.sqrt(x+268))
        if y*y-100==x and z*z-268==x:
            print(x)
            break
        x=x+1
        # print(x)

def findnum_17_max():
    x = 1
    y = 0
    while(x):
        if x%17==0 and x>y:
            y = x
        x = x+1
        if(x>200):
            print(y)
            break

def factorization():
    a = int(input("please input number:"))
    # print(id(a))
    b = a
    j=""
    for i in range(2,a):
        # print(id(a))
        while(i!=a):
            if(a%i==0):
                a = a/i
                # print(id(a),"aaaa")
                j = j+str(i)+"*"
            else:
                # j = j + str(i) + "*"
                break
        if (a % i == 0):
            a = a / i
            # print(id(a), "aaaa")
            j = j + str(i) + "*"
    print(b, "=", j[:-1])

def ball(high,num):
    for i in range(0,num):
        S = (3-0.5**(num-2))*high
    print("第{}次弹起时小球所经过的路程为{}m".format(num,S))
    H = 0.5**num*high
    print("小球第{}次反弹的高度为{}m".format(num, H))


def b(n):
    for j in range(2, n):
        if (n % j == 0):
            return 0
        else:
            return 1
def c():
    for n in range(100,201):
        if(b(n)):
            print(n)
def d():
    dict = {'a':1,'b':2,'c':3}
    for i in dict:
        print("key is {}, value is{}".format(i, dict[i]))

class test():
    def __init__(self,n):
        self.a = n
    def __add__(self):
        return self.a+1

if __name__ == "__main__":
    # print(add())  #计算1+...+100

    # leapyear(int(input("请输入年份：")))   #判别闰年

    # showGraph()        #金字塔

    # sort()               #比较两个数大小并升序打印

    # a()

    # findnum()            #x+100=y*y and x+268 =z*z

    # findnum_17_max()

    # factorization()

    # ball(100, 11)

    # c()

    d()

    # with open("D:\\aaa.txt", "w") as f:
    #     f.write("123")


    a=test(1)
    print(a.add())