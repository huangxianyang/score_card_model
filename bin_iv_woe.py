# -*- coding: utf-8 -*-
"""
连续变量卡方分箱 get_chimerge(),bin_trans()
特征信息价值 Get_IV()
特征权重转换 Woe_Trans()
"""

import numpy as np
import pandas as pd

##################################################################################
#分箱及转换
def get_chimerge(df,var,y,mmax=5):
    """
    基于卡方分箱
    ------------------
    parameter:
            df : dataframe, 必须包含需要分箱的变量,以及目标变量
            var : str 分箱变量
            y : str 目标变量
            mmax : 最大分箱数
    -------------------------------------------
    return:
            result_data : dataframe 分箱结果表
    """
    #进行数据格式化录入,统计需分箱变量每个值负样本数
    total_num = df.groupby([var])[y].count()
    total_num = pd.DataFrame({'total_num': total_num})
    positive_class = df.groupby([var])[y].sum()
    positive_class = pd.DataFrame({'positive_class': positive_class})
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True, how='inner')
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup) #数组转换, 提高计算效率

    #处理连续没有正样本或负样本的区间，并进行区间的合并
    i=0
    while (i <= np_regroup.shape[0] - 2):
            if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
                np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
                np_regroup[i, 0] = np_regroup[i + 1, 0]
                np_regroup = np.delete(np_regroup, i + 1, 0)
                i -= 1
            i += 1

    #对相邻两个区间进行卡方值计算
    chi_table = np.array([]) #相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
        * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
              np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi) 

    #把卡方值最小的两个区间进行合并
    while (1):
            if (len(chi_table) <= (mmax - 1) and min(chi_table) >= confidenceVal):
                break
            chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  #找出卡方值最小的位置索引
            np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            if (chi_min_index == np_regroup.shape[0] - 1):  #最小值试最后两个区间的时候
                #计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2]* np_regroup[chi_min_index, 1]) ** 2 \
                                               * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2])\
                                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                #删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)

            else:
                #计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) \
                                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                #计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                           * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) \
                                        * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                #删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    #输出结果
    result_data = pd.DataFrame()
    #分箱上下限
    lower_list = []
    upper_list = []
    for i in np.arange(np_regroup.shape[0]):
            if i == 0:
                lower_list.append(float("-inf"))
                upper_list.append(np_regroup[0,0])
            elif i == np_regroup.shape[0] - 1:
                lower_list.append(np_regroup[i-1,0])
                upper_list.append(float("inf"))
            else:
                lower_list.append(np_regroup[i-1,0])
                upper_list.append(np_regroup[i,0])

    result_data['lower'] = lower_list  # 结果表区间
    result_data['upper'] = upper_list #结果表区间
    result_data["bin"] = range(1,len(lower_list)+1)
    result_data['y_0'] = np_regroup[:, 2]  # 负样本数目
    result_data['y_1'] = np_regroup[:, 1]  # 正样本数目
    return result_data

#分箱转换

#############################################################################
#woe转换

def Woe_Trans(df,good=0,bad=1,y='y'):
    """
    输出woe转换映射,df必须是分箱后的数据
    -------------------------
    parameter:
            df: dataframe, must include columns y
            good: int, good label ,target label 
            bad: int,bad label,target label
            y: str,target columns name
    return:
            woe_maps : dict key is feature name,value is feature classifer woe
    """
    
    feature_name = list(df.drop(y,axis=1).columns)
    woe_maps = dict()
    for i in feature_name:
        col_good_ratio = df.loc[df[y]==good,i].value_counts()/len(df[df[y]==good]) #各分类好客户比例
        col_bad_ratio = df.loc[df[y]==bad,i].value_counts()/len(df[df[y]==bad]) #各分类坏客户比例
        col_good_bad_ratio = col_good_ratio/col_bad_ratio #各分类好坏比
        woe = -np.log(col_good_bad_ratio) #各分类特征权重
        woe_maps[i] = woe.to_dict()
    return woe_maps

####################################################################################
#计算变量IV
def Get_IV(df,good=0,bad=1,y='y'):
    """
    输出各特征信息价值,注意df必须是分箱后的数据
    -------------------------
    parameter:
            df: dataframe, must include columns y
            good: int, good label ,target label 
            bad: int,bad label,target label
            y: str,target columns name
    return:
            iv : dict key is feature name,value is feature information values
    """
    
    feature_name = list(df.drop(y,axis=1).columns)
    IV = dict()
    for i in feature_name:
        col_good_ratio = df.loc[df[y]==good,i].value_counts()/len(df[df[y]==good]) #各分类好客户比例
        col_bad_ratio = df.loc[df[y]==bad,i].value_counts()/len(df[df[y]==bad]) #各分类坏客户比例
        col_good_bad_ratio = col_good_ratio/col_bad_ratio #各分类好坏比
        woe = np.log(col_good_bad_ratio) #各分类特征权重
        iv = (col_good_ratio-col_bad_ratio)*woe #各分类信息价值
        col_iv = sum(iv) #特征信息价值
        IV[i] = col_iv
    return IV 
