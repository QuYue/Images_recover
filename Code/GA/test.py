# -*- coding: utf-8 -*-
"""
@Time    : 2019/10/12 17:59
@Author  : QuYue
@File    : test.py.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import numpy as np
from geatpy import crtpc
# help(crtpc)

#%% 实整数值种群染色体矩阵的创建
# 定义种群规模
Nind = 4 # 种群规模
Encoding = 'RI' # 实整数编码

# 创建“区域描述器”，表明有4个决策变量，范围分别是[-3.1, 4.2], [-2. 2], [0, 1], [3, 3]
# FieldDR 第三行[0, 0, 1, 1]表明前两个决策变量是连续型的，后两个变量是离散型的
FieldDR = np.array([[-3.1, -2, 0, 3],
                    [4.2, 2, 1, 5],
                    [0, 0, 1, 1]])
# 调用crtri函数创建实数值种群
Chrom = crtpc(Encoding, Nind, FieldDR)
print(Chrom)

#%% 二进制种群染色体矩阵的创建
# 定义种群规模（个体数目）
Nind = 4
Encoding = 'BG' # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
# 创建“译码矩阵”
FieldD = np.array([[3, 2], # 各决策变量编码后所占二进制位数，此时染色体长度为3+2=5
                   [0, 0], # 各决策变量的范围下界
                   [7, 3], # 各决策变量的范围上界
                   [0, 0], # 各决策变量采用什么编码方式(0为二进制编码，1为格雷编码)
                   [0, 0], # 各决策变量是否采用对数刻度(0为采用算术刻度)
                   [1, 1], # 各决策变量的范围是否包含下界(对bs2int实际无效，详见help(bs2int))
                   [1, 1], # 各决策变量的范围是否包含上界(对bs2int实际无效)
                   [0, 0]])# 表示两个决策变量都是连续型变量（0为连续1为离散）
# 调用crtri函数创建实数值种群
Chrom=crtpc(Encoding, Nind, FieldD)
print(Chrom)

#%%
from geatpy import bs2ri
# help(bs2ri)
Phen = bs2ri(Chrom, FieldD)
print('表现型矩阵 = \n', Phen)