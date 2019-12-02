import ast
import random
'''通讯录'''
class addresslist():
    def __init__(self):
        self.name=input("please input name:")
        self.list=ast.literal_eval(input("please input [telephone,email,workaddress]:"))
        self.dict={self.name:self.list}
    def add(self):
        self.dict.update({self.name:self.list})
    def delete1(self,name):
        del self.dict[name]



class circle_sohere():
    def __init__(self,c_radius,s_radius):
        self.c_r=c_radius
        self.s_r=s_radius
    def c_perimeter(self):
        return 2*3.14*self.c_r
    def c_area(self):
        return 3.14*self.c_r**2
    def s_area(self):
        return 4*3.14*self.s_r**2
    def s_volume(self):
        return (4/3)*3.14*self.s_r**3

def celsius_Fahrenheit(T):
    return 9/5*T+32


if __name__ == "__main__":
    # d={}
    # while(1):
    #     a = addresslist()
    #     a.add()
    #     b = int(input("添加：0；删除：1；退出：2"))
    #     for i in d:
    #         print(i,d[i])
    #     if b==0:
    #         continue
    #     if b==1:
    #         name = input("输入要删除的人的姓名：")
    #         a.delete1(name)
    #     if b==2:
    #         break



    # a=circle_sohere(1,2)fo
    # print(a.c_perimeter(),a.c_area())
    # print(a.s_area(),a.s_volume())

    # print(celsius_Fahrenheit(1))


    # with open("data.txt","a+") as f:
    #     for i in range(0,100000):
    #         f.writelines(str(random.randint(1, 100)) + "\n")

    # with open(r"input.txt","r") as f:
    #     lines = f.readlines()
    #     c=1
    #     for i in lines:
    #         print("第{}行{}".format(c,lines))
    #         c=c+1
    try:
        a = int(input("num1"))
        b = int(input("num2"))
        print(a+b)
    except:
        print("请输入阿拉伯数字")
