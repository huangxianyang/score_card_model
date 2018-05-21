# -*- coding: utf-8 -*-
"""
连续变量最优分箱
分类变量最优降基
特征信息价值 Get_IV()
特征权重转换 Woe_Trans()
"""
def moon_bin(X,Y,n=5):
    """
   等频分箱
   ----------------------
   parameter:
       X : pandas Series
       y : pandas Series
       n : int cut number
    ----------------------
    return:
      d4 : dataframe, result qcut
   """   
    
    import scipy.stats.stats as stats
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"x":X,"y":Y, "Bucket" : pd.qcut(X,n)})
        d2 = d1.groupby("Bucket",as_index=True)
        r,p= stats.spearmanr(d2.mean().x,d2.mean().y)
        n -= 1
    d3 = pd.DataFrame()
    d3["min_"+X.name] = d2.min().x
    d3["max_"+X.name] = d2.max().x
    d3[Y.name] = d2.sum().y
    d3["total"] = d2.count().y
    d3[Y.name + "_rate"] = d2.mean().y
    d4 = (d3.sort_index(by = 'min_' + X.name)).reset_index(drop = True)
    return d4

def Woe_Trans(df,good=0,bad=1,y='y'):
    """
    输出woe转换映射
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
