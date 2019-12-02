# %%
import math
def ItemSimilarity(train):
    c = dict()      # 同时被购买的次数
    N = dict()      # 购买用户数
    for u,items in train.items():
        for i in items.keys():
            if i not N.keys():
                N[i] = 0
            N[i] += 1
            for j in items.keys():
                