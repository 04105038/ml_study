# -*- coding: utf-8 -*-

import sys

if __name__ == '__main__':
    # 弱分类器的数目
    # import pudb;pudb.set_trace()
    n_estimator = 10
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression  # 线性回归
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.preprocessing import OneHotEncoder

    # 随机生成分类数据。
    X, y = make_classification(n_samples=80000,n_features=20,n_classes=2)

    # 切分为测试集和训练集，比例0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
    X_train_gbdt, X_train_lr, y_train_gbdt, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    # 调用GBDT分类模型。
    gbdt = GradientBoostingClassifier(n_estimators=n_estimator)
    # 调用one-hot编码。
    one_hot = OneHotEncoder()
    # 调用LR分类模型。
    lr = LogisticRegression()


    '''使用X_train训练GBDT模型，后面用此模型构造特征'''
    gbdt.fit(X_train_gbdt, y_train_gbdt)

    X_leaf_index = gbdt.apply(X_train_gbdt)[:, :, 0]  # apply返回每个样本在每科树中所属的叶子节点索引。行数为样本数，列数为树数目。值为在每个数的叶子索引
    X_lr_leaf_index = gbdt.apply(X_train_lr)[:, :, 0] # apply返回每个样本在每科树中所属的叶子节点索引。行数为样本数，列数为树数目。值为在每个数的叶子索引
    print '每个样本在每个树中所属的叶子索引\n',X_leaf_index
    # fit one-hot编码器
    one_hot.fit(X_leaf_index)  # 训练one-hot编码，就是识别每列有多少可取值
    X_lr_one_hot = one_hot.transform(X_lr_leaf_index)  # 将训练数据，通过gbdt树，形成的叶子节点（每个叶子代表了原始特征的一种组合）索引，编码成one0-hot特征。
    # 编码后的每个特征代表原来的一批特征的组合。

    ''' 
    使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
    '''

    # 使用lr训练gbdt的特征组合
    print '使用逻辑回归训练GBDT组合特征的结果'
    lr.fit(X_lr_one_hot, y_train_lr)
    # 用训练好的LR模型多X_test做预测
    y_pred_grd_lm = lr.predict_proba(one_hot.transform(gbdt.apply(X_test)[:, :, 0]))[:, 1]  # 获取测试集正样本的概率
    # 根据预测结果输出
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_grd_lm)  # 获取真正率和假正率以及门限
    roc_auc = auc(fpr, tpr)
    print 'auc值为\n',roc_auc
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='area = %0.2f' %  roc_auc)
    plt.show()



    # 使用lr直接训练原始数据
    print '使用逻辑回归训练原始数据集的结果'
    lr.fit(X_train_lr, y_train_lr)
    # 用训练好的LR模型多X_test做预测
    y_pred_grd_lm = lr.predict_proba(X_test)[:, 1]  # 获取测试集正样本的概率
    # 根据预测结果输出
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_grd_lm)  # 获取真正率和假正率以及门限
    roc_auc = auc(fpr, tpr)
    print 'auc值为\n',roc_auc
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='area = %0.2f' %  roc_auc)
    plt.show()