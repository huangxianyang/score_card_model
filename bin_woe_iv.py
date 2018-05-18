# -*- coding: utf-8 -*-
"""
连续变量最优分箱
分类变量最优降基
特征信息价值 Get_IV()
特征权重转换
"""

def Get_IV(df,good=0,bad=1,y='y'):
    """
    输出各特征信息价值
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