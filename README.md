评分卡模型
维度统计函数: 字符变量饼图函数 draw_pie, 字符变量条形图函数 draw_bar, 连续变量条形图函数 drawHistogram
数据集划分: divide_data
分箱及降基函数: 连续变量最优分箱函数 binContVar, 离散变量最优降基函数 educeCats
数据转换: 根据最优分箱连续变量数据转换函数 applyBinMap, 根据最优降基分类变量数据转换函数 applyReduceCats
WOE转换:  单个特征woe及iv值计算函数 woe_single_x, 单个特征woe转换函数 _single_woe_trans, 批量特征woe转换 woe_trans
特征权重函数:基于随机森林模型特征函数get_feature,信息价值函数 woe_trans 
模型评估: 混淆矩阵 plot_confusion_matrix , roc曲线 plot_roc_curve,ks曲线ks_stats, 提升图lift_lorenz
评分卡生成: creditCards, statsmodels评分卡函数logitreg,输出ks及roc曲线,
特征相关系数: get_corr
网络搜索: search_param
