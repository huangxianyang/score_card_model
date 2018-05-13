#socre_card.py
"""通用评分卡"""
#################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import matplotlib.mlab as mlab

sns.set_style("darkgrid",{"font.sans-serif":["simhei","Arial"]})

#################################################################################
"""
维度统计
"""
def draw_pie(s):
    """
    字符型变量饼图
    -------------------------------------
    Params
    s: pandas Series
    lalels:labels of each unique value in s
    dropna:bool obj
    -------------------------------------
    Return
    show the plt object
    """
    counts = s.value_counts(dropna=True)
    labels = counts.index
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
    ax.axis('equal')
    ax.set_title(r'pie of {}'.format(s.name))
    plt.show()


def draw_bar(s, x_ticks=None, pct=False, horizontal=False):
    """
    字符型变量条形图
    -------------------------------------------
    Params
    s: pandas Series
    x_ticks: list, ticks in X axis
    pct: bool, True means trans data to odds
    dropna: bool obj,True means drop nan
    horizontal: bool, True means draw horizontal plot
    -------------------------------------------
    Return
    show the plt object
    """
    counts = s.value_counts(dropna=True)
    if pct == True:
        counts = counts / s.shape[0]
    ind = np.arange(counts.shape[0])
    plt.figure(figsize=(8, 6))
    if x_ticks is None:
        x_ticks = counts.index

    if horizontal == False:
        p = plt.bar(ind, counts)
        plt.ylabel('frequecy')
        plt.xticks(ind, tuple(counts.index))
    else:
        p = plt.barh(ind, counts)
        plt.xlabel('frequecy')
        plt.yticks(ind, tuple(counts.index))
    plt.title('Bar plot for %s' % s.name)
    plt.show()


def drawHistogram(s, num_bins):
    """
    连续变量分布图
    ---------------------------------------------
    Params
    s: pandas series
    num_bins: number of bins
    save: bool, is save?
    filename png name
    ---------------------------------------------
    Return
    show the plt object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    mu = s.mean()
    sigma = s.std()

    n, bins, patches = ax.hist(s, num_bins, normed=1, rwidth=0.95, facecolor="blue")

    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--')
    ax.set_xlabel(s.name)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (s.name, mu, sigma))
    plt.show()

#########################################################################################
"""
内置函数
"""


def _check_target_binary(y):
    """检查目标变量是否为二分类
    ------------------------------
    Param
    y:二分类变量,pd.Series格式
    ------------------------------
    Return
    if y is not binary, print valueerror
    """
    from sklearn.utils.multiclass import type_of_target
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')
    else:
        pass


def _isNullZero(x):
    """
    检查变量是否为null或者0
    -----------------------------
    Params
    x: data
    -----------------------------
    Return
    bool obj
    """
    cond1 = np.isnan(x)
    cond2 = x == 0
    return cond1 or cond2


def _Gvalue(binDS, method):
    """
    计算当前分割的度量
    ----------------------------------------
    Params
    binDS: pandas dataframe
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    -----------------------------------------
    Return
    M_value: float or np.nan
    """
    R = binDS['bin'].max()
    N = binDS['total'].sum()

    N_mat = np.empty((R, 3))
    # calculate sum of 0,1
    N_s = [binDS[0].sum(), binDS[1].sum()]
    # calculate each bin's sum of 0,1,total
    # store values in R*3 ndarray
    for i in range(int(R)):
        subDS = binDS[binDS['bin'] == (i + 1)]
        N_mat[i][0] = subDS[0].sum()
        N_mat[i][1] = subDS[1].sum()
        N_mat[i][2] = subDS['total'].sum()

    # Gini
    if method == 1:
        G_list = [0] * R
        for i in range(int(R)):

            for j in range(2):
                G_list[i] = G_list[i] + N_mat[i][j] * N_mat[i][j]
            G_list[i] = 1 - G_list[i] / (N_mat[i][2] * N_mat[i][2])
        G = 0
        for j in range(2):
            G = G + N_s[j] * N_s[j]

        G = 1 - G / (N * N)
        Gr = 0
        for i in range(int(R)):
            Gr = Gr + N_mat[i][2] * (G_list[i] / N)
        M_value = 1 - Gr / G

    # Entropy
    elif method == 2:
        for i in range(int(R)):
            for j in range(2):
                if np.isnan(N_mat[i][j]) or N_mat[i][j] == 0:
                    M_value = 0

        E_list = [0] * R
        for i in range(int(R)):
            for j in range(2):
                E_list[i] = E_list[i] - ((N_mat[i][j] / float(N_mat[i][2])) * np.log(N_mat[i][j] / N_mat[i][2]))

            E_list[i] = E_list[i] / np.log(2)  # plus
        E = 0
        for j in range(2):
            a = (N_s[j] / N)
            E = E - a * (np.log(a))

        E = E / np.log(2)
        Er = 0
        for i in range(2):
            Er = Er + N_mat[i][2] * E_list[i] / N
        M_value = 1 - (Er / E)
        return M_value

    # Pearson X2
    elif method == 3:
        N = N_s[0] + N_s[1]
        X2 = 0
        M = np.empty((R, 2))
        for i in range(int(R)):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + (N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j]) / (M[i][j])

        M_value = X2

    # Info value
    else:
        if any([_isNullZero(N_mat[i][0]), _isNullZero(N_mat[i][1]), _isNullZero(N_s[0]), _isNullZero(N_s[1])]):
            M_value = np.NaN
        else:
            IV = 0
            for i in range(int(R)):
                IV = IV + (N_mat[i][0] / N_s[0] - N_mat[i][1] / N_s[1]) * np.log(
                    (N_mat[i][0] * N_s[1]) / (N_mat[i][1] * N_s[0]))
            M_value = IV

    return M_value


def _calCMerit(temp, ix, method):
    """
    计算当前临时表的评价函数
    ---------------------------------------------
    Params
    temp: pandas dataframe, 最优分箱临时表
    ix: single int obj,index of temp, from length of temp
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    ---------------------------------------------
    Return
    M_value: float or np.nan
    """
    # split data by ix
    temp_L = temp[temp['i'] <= ix]
    temp_U = temp[temp['i'] > ix]
    # calculate sum of 0, 1, total for each splited data
    n_11 = float(sum(temp_L[0]))
    n_12 = float(sum(temp_L[1]))
    n_21 = float(sum(temp_U[0]))
    n_22 = float(sum(temp_U[1]))
    n_1s = float(sum(temp_L['total']))
    n_2s = float(sum(temp_U['total']))
    # calculate sum of 0, 1 for whole data
    n_s1 = float(sum(temp[0]))
    n_s2 = float(sum(temp[1]))
    N_mat = np.array([[n_11, n_12, n_1s],
                      [n_21, n_22, n_2s]])
    N_s = [n_s1, n_s2]
    # Gini
    if method == 1:
        N = n_1s + n_2s
        G1 = 1 - ((n_11 * n_11 + n_12 * n_12) / float(n_1s * n_1s))
        G2 = 1 - ((n_21 * n_21 + n_22 * n_22) / float(n_2s * n_2s))
        G = 1 - ((n_s1 * n_s1 + n_s2 * n_s2) / float(N * N))
        M_value = 1 - ((n_1s * G1 + n_2s * G2) / float(N * G))
    # Entropy
    elif method == 2:
        N = n_1s + n_2s
        E1 = -((n_11 / n_1s) * (np.log((n_11 / n_1s))) + (n_12 / n_1s) * (np.log((n_12 / n_1s)))) / (np.log(2))
        E2 = -((n_21 / n_2s) * (np.log((n_21 / n_2s))) + (n_22 / n_2s) * (np.log((n_22 / n_2s)))) / (np.log(2))
        E = -(((n_s1 / N) * (np.log((n_s1 / N))) + ((n_s2 / N) * np.log((n_s2 / N)))) / (np.log(2)))
        M_value = 1 - (n_1s * E1 + n_2s * E2) / (N * E)
    # Pearson chisq
    elif method == 3:
        N = n_1s + n_2s
        X2 = 0
        M = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + ((N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j])) / M[i][j]

        M_value = X2
    # Info Value
    else:
        try:
            IV = ((n_11 / n_s1) - (n_12 / n_s2)) * np.log((n_11 * n_s2) / (n_12 * n_s1)) + (
                        (n_21 / n_s1) - (n_22 / n_s2)) * np.log((n_21 * n_s2) / (n_22 * n_s1))
            M_value = IV
        except ZeroDivisionError:
            M_value = np.nan
    return M_value


def _bestSplit(binDS, method, BinNo):
    """
    find the best split for one bin dataset
    middle procession functions for _candSplit
    --------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    BinNo: int obj, bin number of binDS
    --------------------------------------
    Return
    newbinDS: pandas dataframe
    """
    binDS = binDS.sort_values(by=['bin', 'pdv1'])
    mb = len(binDS[binDS['bin'] == BinNo])

    bestValue = 0
    bestI = 1
    for i in range(1, mb):
        # split data by i
        # metric: Gini,Entropy,pearson chisq,Info value
        value = _calCMerit(binDS, i, method)
        # if value>bestValue，then make value=bestValue，and bestI = i
        if bestValue < value:
            bestValue = value
            bestI = i
    # create new var split
    binDS['split'] = np.where(binDS['i'] <= bestI, 1, 0)
    binDS = binDS.drop('i', axis=1)
    newbinDS = binDS.sort_values(by=['split', 'pdv1'])
    # rebuild var i
    newbinDS_0 = newbinDS[newbinDS['split'] == 0]
    newbinDS_1 = newbinDS[newbinDS['split'] == 1]
    newbinDS_0['i'] = range(1, len(newbinDS_0) + 1)
    newbinDS_1['i'] = range(1, len(newbinDS_1) + 1)
    newbinDS = pd.concat([newbinDS_0, newbinDS_1], axis=0)
    return newbinDS  # .sort_values(by=['split','pdv1'])


def _candSplit(binDS, method):
    """
    Generate all candidate splits from current Bins
    and select the best new bins
    middle procession functions for binContVar & reduceCats
    ---------------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    --------------------------------------------
    Return
    newBins: pandas dataframe, split results
    """
    # sorted data by bin&pdv1
    binDS = binDS.sort_values(by=['bin', 'pdv1'])
    # get the maximum of bin
    Bmax = max(binDS['bin'])
    # screen data and cal nrows by diffrence bin
    # and save the results in dict
    temp_binC = dict()
    m = dict()
    for i in range(1, Bmax + 1):
        temp_binC[i] = binDS[binDS['bin'] == i]
        m[i] = len(temp_binC[i])
    """
    CC
    """
    # create null dataframe to save info
    temp_trysplit = dict()
    temp_main = dict()
    bin_i_value = []
    for i in range(1, Bmax + 1):
        if m[i] > 1:  # if nrows of bin > 1
            # split data by best i
            temp_trysplit[i] = _bestSplit(temp_binC[i], method, i)
            temp_trysplit[i]['bin'] = np.where(temp_trysplit[i]['split'] == 1, Bmax + 1, temp_trysplit[i]['bin'])
            # delete bin == i
            temp_main[i] = binDS[binDS['bin'] != i]
            # vertical combine temp_main[i] & temp_trysplit[i]
            temp_main[i] = pd.concat([temp_main[i], temp_trysplit[i]], axis=0)
            # calculate metric of temp_main[i]
            value = _Gvalue(temp_main[i], method)
            newdata = [i, value]
            bin_i_value.append(newdata)
    # find maxinum of value bintoSplit
    bin_i_value.sort(key=lambda x: x[1], reverse=True)
    # binNum = temp_all_Vals['BinToSplit']
    binNum = bin_i_value[0][0]
    newBins = temp_main[binNum].drop('split', axis=1)
    return newBins.sort_values(by=['bin', 'pdv1'])

def _applyBinMap(x, bin_map):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    # bin_res = np.array([0] * x.shape[-1], dtype=int)
    # x2 = x2.copy()
    res = x.copy()
    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]
        # 寻找出 >=lower and < upper的位置
        loc = np.where((x >= lower) & (x < upper))[0]
        ind = x.iloc[loc].index

        res.loc[ind] = i

    res.name = res.name + "_BIN"

    return res

def _EqualWidthBinMap(x, Acc, adjust):
    """
    Data bining function,
    middle procession functions for binContVar
    method: equal width
    Mind: Generate bining width and interval by Acc
    --------------------------------------------
    Params
    x: pandas Series, data need to bining
    Acc: float less than 1, partition ratio for equal width bining
    adjust: float or np.inf, bining adjust for limitation
    --------------------------------------------
    Return
    bin_map: pandas dataframe, Equal width bin map
    """
    varMax = x.max()
    varMin = x.min()
    # generate range by Acc
    Mbins = int(1. / Acc)
    minMaxSize = (varMax - varMin) / Mbins
    # get upper_limit and loewe_limit
    ind = range(1, Mbins + 1)
    Upper = pd.Series(index=ind, name='upper')
    Lower = pd.Series(index=ind, name='lower')
    for i in ind:
        Upper[i] = varMin + i * minMaxSize
        Lower[i] = varMin + (i - 1) * minMaxSize

    # adjust the min_bin's lower and max_bin's upper
    Upper[Mbins] = Upper[Mbins] + adjust
    Lower[1] = Lower[1] - adjust
    bin_map = pd.concat([Lower, Upper], axis=1)
    bin_map.index.name = 'bin'
    return bin_map


def _combineBins(temp_cont, target):
    """
    merge all bins that either 0 or 1 or total =0
    middle procession functions for binContVar
    ---------------------------------
    Params
    temp_cont: pandas dataframe, middle results of binContVar
    target: target label
    --------------------------------
    Return
    temp_cont: pandas dataframe
    """
    for i in temp_cont.index:
        rowdata = temp_cont.ix[i, :]

        if i == temp_cont.index.max():
            ix = temp_cont[temp_cont.index < i].index.max()

        else:
            ix = temp_cont[temp_cont.index > i].index.min()
        if any(rowdata[:3] == 0):  # 如果0,1,total有一项为0，则运行
            #
            temp_cont.ix[ix, target] = temp_cont.ix[ix, target] + rowdata[target]
            temp_cont.ix[ix, 0] = temp_cont.ix[ix, 0] + rowdata[0]
            temp_cont.ix[ix, 'total'] = temp_cont.ix[ix, 'total'] + rowdata['total']
            #
            if i < temp_cont.index.max():
                temp_cont.ix[ix, 'lower'] = rowdata['lower']
            else:
                temp_cont.ix[ix, 'upper'] = rowdata['upper']
            temp_cont = temp_cont.drop(i, axis=0)

    return temp_cont.sort_values(by='pdv1')


def _getNewBins(sub, i):
    """
    get new lower, upper, bin, total for sub
    middle procession functions for binContVar
    -----------------------------------------
    Params
    sub: pandas dataframe, subdataframe of temp_map
    i: int, bin number of sub
    ----------------------------------------
    Return
    df: pandas dataframe, one row
    """
    l = len(sub)
    total = sub['total'].sum()
    first = sub.iloc[0, :]
    last = sub.iloc[l - 1, :]

    lower = first['lower']
    upper = last['upper']
    df = pd.DataFrame()
    df = df.append([i, lower, upper, total], ignore_index=True).T
    df.columns = ['bin', 'lower', 'upper', 'total']
    return df


def _groupCal(x, y, badlabel=1):
    """
    group calulate for x by y
    middle proporcessing function for reduceCats
    -------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    badlabel: target label
    ------------------------------------
    Return
    temp_cont: group calulate table
    m: nrows of temp_cont
    """

    temp_cont = pd.crosstab(index=x, columns=y, margins=False)
    temp_cont['total'] = temp_cont.sum(axis=1)
    temp_cont['pdv1'] = temp_cont[badlabel] / temp_cont['total']

    temp_cont['i'] = range(1, temp_cont.shape[0] + 1)
    temp_cont['bin'] = 1
    m = temp_cont.shape[0]
    return temp_cont, m

def _count_binary(a, event=1):
    """变量频数统计
    ------------------------------
    Params
    a: pd.Series格式
    event: 事件1 拖欠状态
    ------------------------------
    Return
    event_count: 事件event=1的总数
    non_event_count: 事件event !=1的总数
    """
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count

################################################################################################
"""
连续变量最优分箱
离散型变量最优降基
"""
def binContVar(x, y, method, mmax=5, Acc=0.01, target=1, adjust=0.0001):
    """
    连续变量最优分箱 for contiouns var x by (y & method)
    method is represent by number,
        1:Gini, 2:Entropy, 3:pearson chisq, 4:Info value
    ---------------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: int, bining number
    Acc: float less than 1, partition ratio for equal width bining
    badlabel: target label
    adjust: float or np.inf, bining adjust for limitation
    ---------------------------------------------
    Return
    temp_Map: pandas dataframe, Optimal bining map
    """
    # if y is not 0-1 binary variable, then raise a error
    _check_target_binary(y)
    # data bining by Acc, method: width equal
    bin_map = _EqualWidthBinMap(x, Acc, adjust=adjust)
    # mapping x to bin number and combine with x&y
    bin_res = _applyBinMap(x, bin_map)
    temp_df = pd.concat([x, y, bin_res], axis=1)
    # calculate freq of 0, 1 in y group by bin_res
    t1 = pd.crosstab(index=temp_df[bin_res.name], columns=y)
    # calculate freq of bin, and combine with t1
    t2 = temp_df.groupby(bin_res.name).count().iloc[:, 0]
    t2 = pd.DataFrame(t2)
    t2.columns = ['total']
    t = pd.concat([t1, t2], axis=1)
    # merge t & bin_map by t,
    # if all(0,1,total) == 1, so corresponding row will not appear in temp_cont
    temp_cont = pd.merge(t, bin_map,
                         left_index=True, right_index=True,
                         how='left')
    temp_cont['pdv1'] = temp_cont.index
    # if any(0,1,total)==0, then combine it with per bin or next bin
    temp_cont = _combineBins(temp_cont, target)
    # calculate other temp vars
    temp_cont['bin'] = 1
    temp_cont['i'] = range(1, len(temp_cont) + 1)
    temp_cont['var'] = temp_cont.index
    nbins = 1
    # exe candSplit mmax times
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': 'oldbin'})
    temp_Map1 = temp_cont.drop([0, target, 'pdv1', 'i'], axis=1)
    temp_Map1 = temp_Map1.sort_values(by=['bin', 'oldbin'])
    # get new lower, upper, bin, total for sub
    data = pd.DataFrame()
    s = set()
    """
    if temp_Map1.shape[0] < mmax:
        print("请选择合适的分箱数")
    """
    for i in temp_Map1['bin']:
        if i in s:
            pass
        else:
            sub_Map = temp_Map1[temp_Map1['bin'] == i]
            rowdata = _getNewBins(sub_Map, i)
            data = data.append(rowdata, ignore_index=True)
            s.add(i)

    # resort data
    data = data.sort_values(by='lower')
    data = data.drop('bin', axis=1)
    data['bin'] = range(1, mmax + 1)

    data.index = data['bin']
    return data


def reduceCats(x, y, method=1, mmax=5, badlabel=1):
    """
    离散变量最优降基 for x by y & method
    method is represent by number,
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ----------------------------------------------
    Params:
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: number to reduce
    badlabel: target label
    ---------------------------------------------
    Return
    temp_cont: pandas dataframe, reduct category map
    """
    _check_target_binary(y)
    temp_cont, m = _groupCal(x, y, badlabel=badlabel)
    nbins = 1
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': x.name})
    temp_cont = temp_cont.drop([0, 1, 'i', 'pdv1'], axis=1)
    return temp_cont.sort_values(by='bin')

###########################################################################################
"""
根据最优分箱函数进行数据转换 
最优降基函数进行数据转换
"""

def applyBinMap(x, bin_map):
    """
    根据最优分箱连续变量数据转换
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    # bin_res = np.array([0] * x.shape[-1], dtype=int)
    # x2 = x2.copy()
    x_new = x.copy()
    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]
        # 寻找出 >=lower and < upper的位置
        loc = np.where((x >= lower) & (x < upper))[0]
        ind = x.iloc[loc].index

        x_new.loc[ind] = i

    x_new.name = x_new.name + "_BIN"

    return x_new


def applyReduceCats(x, bin_map):
    """
    根据最优降基分类变量数据转换
    convert x to newbin by bin_map
    ------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, mapTable contain new bins
    ------------------------------
    Return
    new_x: pandas Series, convert results
    """
    d = dict()
    for i in bin_map.index:
        subData = bin_map[bin_map.index == i]
        value = subData.ix[i, 'bin']
        d[i] = value

    new_x = x.map(d)
    new_x.name = x.name + '_BIN'
    return new_x

#################################################################################
def woe_single_x(x, y):
    """
    计算特征的WOE和信息价值
    -----------------------------------------------
    Param
    x: 单个特征值,如data['SEX']
    y: 目标变量,data['y']
    -----------------------------------------------
    return
    woe_dict: 包括该特征对应的woe值
    iv: 是改变量的信息价值
    """
    event = 1
    EPS = 1e-7
    # 检查是否是二分类
    _check_target_binary(y)
    # 好坏客户量
    event_total, non_event_total = _count_binary(y, event=event)
    # 特征分类
    x_labels = np.unique(x)

    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = _count_binary(y1, event=event)
        # 计算P_1 和 P_0
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total

        if rate_event == 0:
            rate_event = EPS

        elif rate_non_event == 0:
            rate_non_event = EPS

        else:
            pass

        woe1 = np.math.log(rate_event / rate_non_event)  # 好坏比
        woe_dict[x1] = woe1

        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv

def _single_woe_trans(x, y):
    """
    单个特征woe转换
    ---------------------------------------
    Param
    x: 单个特征, pandas series
    y: 目标变量, pandas series
    ---------------------------------------
    Return
    x_woe_trans: woe trans by x
    woe_map: map for woe trans
    info_value: infor value of x
    """
    woe_map, info_value = woe_single_x(x, y)
    x_woe_trans = x.map(woe_map)
    x_woe_trans.name = x.name + "_WOE"
    return x_woe_trans, woe_map, info_value

def woe_trans(varnames, y, df):
    """
    批量特征woe转换
    ---------------------------------------
    Param
    varnames: list
    y:  pandas series, target variable
    df: pandas dataframe, dataframe必须包括所有varnames变量
    ---------------------------------------
    Return
    df: pandas dataframe, trans results
    woe_maps: dict, key is varname, value is woe
    iv_values: dict, key is varname, value is info value
    """
    iv_values = {}
    woe_maps = {}
    for var in varnames:
        x = df[var]
        x_woe_trans, woe_map, info_value = _single_woe_trans(x, y)
        df = pd.concat([df, x_woe_trans], axis=1)
        woe_maps[var] = woe_map
        iv_values[var] = info_value

    return df, woe_maps, iv_values

#特征权重
def get_feature(X,y,n_estimators):
    """
    X拟合特征,y数据标签,n_estimators随机森林树的数量
    """
    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier(n_estimators=n_estimators,random_state=123)#构建分类随机森林分类器
    rf.fit(X,y) #对自变量和因变量进行拟合

    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (12,6)
    sns.set_style("darkgrid",{"font.sans-serif":["simhei","Arial"]})
    ## feature importances 可视化##
    importances = rf.feature_importances_
    feat_names = X.columns
    indices = np.argsort(importances)[::-1]
    fig = plt.figure(figsize=(20,6))
    plt.title("feature importance")
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
    plt.xlim([-1, len(indices)])

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target

def plot_roc_curve(pred_y, y):
    """
    plot roc curve
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    ----------------------------------
    plt object
    """
    fpr, tpr, _ = roc_curve(y, pred_y)
    c_stats = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label="ROC curve")
    s = "AUC = %.4f" % c_stats
    plt.text(0.8, 0.2, s, bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')  # ROC 曲线
    plt.legend(loc='best')
    plt.show()


"""
KS曲线以及KS统计表
"""


def ks_stats(pred_y, y, k=20):
    """
    plot K-S curve and output ks table
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number
    ----------------------------------
    ks_results: pandas dataframe
    ks_ax: plt object, k-s curcve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(pred_y, name='pred_y', index=y.index)], axis=1)
    datasets.columns = ["y", "pred_y"]
    datasets = datasets.sort_values(by="pred_y", axis=0, ascending=True)
    # 计算正负案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    N = Nrows - P
    n = float(Nrows) / k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    ks_df = pd.DataFrame()
    rlt = {
        "tile": str(0),
        "Ptot": 0,
        "Ntot": 0}
    ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i * n
        up = (i + 1) * n
        tile = datasets.ix[lo:(up - 1), :]
        Ptot = sum(tile['y'])
        Ntot = n - Ptot
        rlt = {
            "tile": str(i + 1),
            "Ptot": Ptot,
            "Ntot": Ntot}
        ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    # 计算各子集中的正负例比例,以及累积比例
    ks_df['PerP'] = ks_df['Ptot'] / P
    ks_df['PerN'] = ks_df['Ntot'] / N
    ks_df['PerP_cum'] = ks_df['PerP'].cumsum()
    ks_df['PerN_cum'] = ks_df['PerN'].cumsum()
    # 计算ks曲线以及ks值
    ks_df['ks'] = ks_df['PerN_cum'] - ks_df['PerP_cum']
    ks_value = ks_df['ks'].max()
    s = "KS value is %.4f" % ks_value
    # 整理得出ks统计表
    ks_results = ks_df.ix[1:, :]
    ks_results = ks_results[['tile', 'Ntot', 'Ptot', 'PerN', 'PerP', 'PerN_cum', 'PerP_cum', 'ks']]
    ks_results.columns = ['子集', '负例数', '正例数', '负例比例', '正例比例', '累积负例比例', '累积正例比例', 'ks']
    # 获取ks值所在的数据点
    ks_point = ks_results.ix[:, ['子集', 'ks']]
    ks_point = ks_point.ix[ks_point['ks'] == ks_point['ks'].max(), :]
    # 绘制KS曲线
    ks_ax = _ks_plot(ks_df=ks_df, ks_label='ks', good_label='PerN_cum', bad_label='PerP_cum',
                     k=k, point=ks_point, s=s)
    return ks_results, ks_ax


def _ks_plot(ks_df, ks_label, good_label, bad_label, k, point, s):
    """
    middle function for ks_stats, plot k-s curve
    """
    plt.plot(ks_df['tile'], ks_df[ks_label], "r-.", label="ks_curve", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[good_label], "g-.", label="good", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[bad_label], "m-.", label="bad", lw=1.2)
    # plt.plot(point[0], point[1], 'o', markerfacecolor="red",
    # markeredgecolor='k', markersize=6)
    plt.legend(loc=0)
    plt.plot([0, k], [0, 1], linestyle='--', lw=0.8, color='k', label='Luck')
    plt.xlabel("decilis")  # 等份子集
    plt.title(s)  # KS曲线图
    plt.show()





def plot_confusion_matrix(y,
                          pred,
                          labels,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    混淆矩阵
    ------------------------------------------
    Params
    y：real data labels
    pred: predict results
    labels: labels
    normalize: bool, True means trans results to percent
    cmap: color index
    ------------------------------------------
    Return
    plt object
    """
    cm = confusion_matrix(y, pred, labels=labels)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在指定的轴上展示图像

    plt.colorbar()  # 增加色柱
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)  # 设置坐标轴标签
    plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("标准化混淆矩阵")
    else:
        # print('混淆矩阵')
        pass
    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title("confusion matrix")
    plt.show()


"""
提升图和洛伦茨曲线
"""


def lift_lorenz(pred_y, y, k=10):
    """
    plot lift_lorenz curve
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number
    ----------------------------------
    lift_ax: lift chart
    lorenz_ax: lorenz curve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(pred_y, name='pred_y', index=y.index)], axis=1)
    datasets.columns = ["y", "pred_y"]
    datasets = datasets.sort_values(by="pred_y", axis=0, ascending=False)
    # 计算正案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    n = float(Nrows) / k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    lift_df = pd.DataFrame()
    rlt = {
        "tile": str(0),
        "Ptot": 0,
    }
    lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i * n
        up = (i + 1) * n
        tile = datasets.ix[lo:(up - 1), :]
        Ptot = sum(tile['y'])
        rlt = {
            "tile": str(i + 1),
            "Ptot": Ptot,
        }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    # 计算正例比例&累积正例比例
    lift_df['PerP'] = lift_df['Ptot'] / P
    lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
    # 计算随机正例数、正例率以及累积随机正例率
    lift_df['randP'] = float(P) / k
    lift_df['PerRandP'] = lift_df['randP'] / P
    lift_df.ix[0, :] = 0
    lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
    lift_ax = lift_Chart(lift_df, k)
    lorenz_ax = lorenz_cruve(lift_df)
    return lift_ax, lorenz_ax


def lift_Chart(df, k):
    """
    middle function for lift_lorenz, plot lift Chart
    """
    # 绘图变量
    PerP = df['PerP'][1:]
    PerRandP = df['PerRandP'][1:]
    # 绘图参数
    fig, ax = plt.subplots(figsize=(12,6))
    index = np.arange(k + 1)[1:]
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, PerP, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Per_p')  # 正例比例
    rects2 = plt.bar(index + bar_width, PerRandP, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='random_P')  # 随机比例
    plt.xlabel('Group')
    plt.ylabel('Percent')
    plt.title('lift_Chart')
    plt.xticks(index + bar_width / 2, tuple(index))
    plt.legend()
    plt.tight_layout()
    plt.show()


def lorenz_cruve(df):
    """
    middle function for lift_lorenz, plot lorenz cruve
    """
    # 准备绘图所需变量
    PerP_cum = df['PerP_cum']
    PerRandP_cum = df['PerRandP_cum']
    decilies = df['tile']
    # 绘制洛伦茨曲线
    plt.figure(figsize=(13,6))
    plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')  # lorenz曲线
    plt.plot(decilies, PerRandP_cum, 'k-.', label='random')  # 随机
    plt.legend()
    plt.xlabel("decilis")  # 等份子集
    plt.title("lorenz_cruve", fontsize=10)  # 洛伦茨曲线
    plt.show()

############################################################################################
"""
评分卡生成函数
"""
def creditCards(paramsEst,woe_maps,bin_maps, red_maps, basepoints=600,odds=60,PDO=20):
    """
    评分卡生成函数
    output credit card for each var in model
    --------------------------------------------
    ParamsEst: pandas Series, params estimate results in logistic model,
               index is param names,value is  estimate results
    bin_maps: dict, key is varname of paramEst , value is pandas dataframe contains map table
    red_maps: dict, key is varname of paramEst, value is pandas dataframe contains redCats table
    woe_maps: dict, key is varname of paramEst, value is also a dict contain binNumber--woe
    basepoints: expect base points
    odds: reciprocal of Odds
    PDO: double coefficients
    -------------------------------------------
    Return
    creditCard: pandas dataframe
    """
    # 计算A&B
    alpha, beta = _score_cal(basepoints, odds, PDO)
    # 计算基础分
    points_0 = round(alpha - beta * paramsEst['const'])
    # 根据各段woe，计算相应得分
    points = pd.DataFrame()
    for k in woe_maps.keys():
        d = pd.DataFrame(woe_maps[k], index=[k]).T

        d['points'] = round(-beta * d.ix[:, k] * paramsEst[k])
        if k in bin_maps.keys():
            bin_map = bin_maps[k]
            bin_map = bin_map.drop(['total', 'bin'], axis=1)
            bin_map['range'] = bin_map.apply(lambda x: str(x[0]) + '--' + str(x[1]), axis=1)
            bin_map = bin_map.drop(['lower', 'upper'], axis=1)
            d = pd.merge(d, bin_map, left_index=True, right_index=True)

        elif k in red_maps.keys():
            red_map = red_maps[k]
            s = tableTranslate(red_map)
            s = pd.DataFrame(s.T, columns=['range'])
            d = pd.merge(d, s, left_index=True, right_index=True)

        else:
            d['range'] = d.index

        n = len(d)
        ind_0 = []
        i = 0
        while i < n:
            ind_0.append(k)
            i += 1

        d.index = [ind_0, list(d.index)]
        d = d.drop(k, axis=1)
        points = pd.concat([points, d], axis=0)

    # 输出评分卡
    points_0 = pd.DataFrame([[points_0, '-']],
                            index=[['basePoints'], ['-']],
                            columns=['points', 'range'])
    credit_card = pd.concat([points_0, points], axis=0)
    credit_card.index.names = ["varname", "binCode"]
    return credit_card


def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin'] == i].index
        value = list(value.map(lambda x: str(x) + ';'))
        value = "".join(value)
        res[i] = value
    return res


def _score_cal(basepoints, odds, PDO):
    """
    cal alpha&beta for score formula,
    score = alpha + beta * log(odds)
    ---------------------------------------
    Params
    basepoints: expect base points
    odds: cal by logit model
    PDO: points of double odds
    ---------------------------------------
    Return
    alpha, beta
    """
    beta = PDO / np.log(2)
    alpha = basepoints - beta * np.log(odds)
    return alpha, beta

##############################################################################################
#statsmodels评分卡
def draw_roc(y_pred, y_test, ks=True):
    """
    ROC及KS曲线
    ------------------
    参数: y_pred: series or narray
         y_test: Series or narray
    """
    from sklearn.metrics import accuracy_score
    tprlist = []
    fprlist = []
    auc = 0
    ks_list, m1, m2, ks_value = [], [], [], 0
    for i in range(1, 1001):
        thres = 1 - i / 1000
        yp = []
        for item in y_pred:
            if item > thres:
                yp.append(1)
            else:
                yp.append(0)
        Nobs = len(y_test)
        h1 = sum(yp)
        t1 = sum(y_test)
        fn = int((sum(abs(y_test - yp)) + t1 - h1) / 2)
        tp = t1 - fn
        fp = h1 - tp
        tn = Nobs - h1 - fn
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        tprlist.append(tpr)
        fprlist.append(fpr)
        ks_list.append(tpr - fpr)
    for i in range(999):
        auc = auc + (fprlist[i + 1] - fprlist[i]) * tprlist[i]
    print("auc=", auc)
    plt.plot(fprlist, tprlist)
    plt.show()
    if ks:
        for i in range(10):
            m1.append(tprlist[i * 100])
            m2.append(fprlist[i * 100])
        ks_value = max(ks_list)
        print('ks value=', ks_value)
        x1 = range(10)
        x_axis = []
        for i in x1:
            x_axis.append(i / 10)
        plt.plot(x_axis, m1)
        plt.plot(x_axis, m2)
        plt.show()
        y_pred01 = []
        for item in y_pred:
            if item > 0.5:
                y_pred01.append(1)
            else:
                y_pred01.append(0)
        print("accuracy score=", accuracy_score(y_pred01, y_test))
 

def logitreg(df, k=0, ks=True):
    """
    logitmodel
    ============
    参数:
    df:dataframe
    k=0,int,
    ks=True
    """
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split 
    
#     x = df
#     x1, x0 = x[x['target'] == 1], x[x['target'] == 0]
#     y1, y0 = x1['target'], x0['target']
#     x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=0)
#     x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, random_state=0)
#     x_train, x_test, y_train, y_test = pd.concat([x0_train, x1_train]), pd.concat([x0_test, x1_test]), pd.concat(
#         [y0_train, y1_train]), pd.concat([y0_test, y1_test])
#     x_train, x_test = sm.add_constant(x_train.iloc[:, 1:]), sm.add_constant(x_test.iloc[:, 1:])
    x_train,x_test,y_train,y_test = train_test_split(df.drop("target",axis=1),df["target"],test_size=0.3,random_state=0)
    x_train, x_test = sm.add_constant(x_train.iloc[:, 1:]), sm.add_constant(x_test.iloc[:, 1:])
    var = list(x_train)[1:]
    st = set()
    st.add("const")

    while True:
        pvs = []
        for item in var:
            if item not in st:
                l = list(st) + [item]
                xx = x_train[l]
                logit_mod = sm.OLS(y_train, xx) #我的Python版本不支持logit
                logitres = logit_mod.fit()
                pvs.append([item, logitres.pvalues[item]])
        v = sorted(pvs, key=lambda x: x[1])[0]  
        if v[1] < 0.05:
            st.add(v[0])
            continue  #while语法有误,此处加continue
        else:
            break

        ltest = list(st)
        xtest = x_train[ltest]
        test_mod = sm.OLS(y_train, xtest)
        testres = test_mod.fit()
        for item in set(st): #迭代过程不应该更改集合大小
            if testres.pvalues[item] > 0.05:
                st.remove(item)
                print("We have removed item:", item)

    print("the list to use for logistic regression:", st)

    luse = list(st)
    vars_to_del = []
    for item in var:
        if item not in luse:
            vars_to_del.append(item)

    for item in vars_to_del:
        var.remove(item) #此处将pop改为remove

    xuse = x_train[luse]
    logit_mod = sm.OLS(y_train, xuse)
    logit_res = logit_mod.fit()
    print(logit_res.summary())
    print("the roc and ks of train set is:")
    y_pred = np.array(logit_res.predict(x_test[luse]))
    draw_roc(y_pred, y_test, ks)
    print("the roc and ks of test set is:")
    y_ptrain = np.array(logit_res.predict(x_train[luse]))
    draw_roc(y_ptrain, y_train, ks)
    return logit_res, luse

#########################################################################
#特征工程
def get_corr(data_new,figsize):
    """
    特征相关系数
    ------------------------
    parameter:
    data_new: dataFrame,columns must be number
    figsize: tupe,two number
    return:
            heatmap
    """
    #相关系数分析
    colormap = plt.cm.viridis
    plt.figure(figsize=figsize)
    plt.title('皮尔森相关性系数', y=1.05, size=8)
    mask = np.zeros_like(data_new.corr(),dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(data_new.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True,mask=mask)
    
