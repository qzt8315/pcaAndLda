from numpy import *
import numpy as np

# 通过PCA降维算法
'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum>=arraySum*percentage:
            return num

'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''
def pca(dataMat,percentage=0.9):
    meanVals=mean(dataMat, axis=0)  # 对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved = dataMat - meanVals
    covMat=cov(meanRemoved, rowvar=0)  # cov()计算方差
    eigVals, eigVects=linalg.eig(mat(covMat))  # 利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k=eigValPct(eigVals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  # 对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1]  # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects=eigVects[:, eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat
    return real(lowDDataMat),real(reconMat)

# 计算LDA
# 特征均值,计算每类的均值，返回一个向量
def class_mean(data,label,clusters):
    mean_vectors = []
    data = np.asarray(data)
    for cl in range(1,clusters+1):
        mean_vectors.append(np.mean(data[label == cl, :], axis=0).tolist())
    # print(mean_vectors)
    return np.array(mean_vectors)

# 计算类内散度
def within_class_SW(data,label,clusters):
    m = data.shape[1]
    S_W = np.zeros((m,m))
    mean_vectors = class_mean(data, label, clusters)
    for cl, mv in zip(range(1,clusters+1),mean_vectors):
        class_sc_mat = np.zeros((m,m))
        mv = mv.reshape(len(mv), 1)
        # 对每个样本数据进行矩阵乘法
        for row  in data[label == cl]:
            row = row.T
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W +=class_sc_mat
    # print(S_W)
    return S_W

# 计算类间散度
def between_class_SB(data,label,clusters):
    m = data.shape[1]
    all_mean =np.mean(data,axis = 0).T
    S_B = np.zeros((m,m))
    mean_vectors = class_mean(data,label,clusters)
    for cl ,mean_vec in enumerate(mean_vectors):
        n = data[label == cl+1, :].shape[0]
        mean_vec = mean_vec.reshape(data.shape[1],1)  # make column vector
        S_B += n * (mean_vec - all_mean).dot((mean_vec - all_mean).T)
    # print(S_B)
    return S_B

# 进行lda处理
def lda(data, label, k):
    """
    :param data: 输入数据
    :param label:
    :param k:
    :return:
    """
    # data, label = read_iris()
    clusters = len(set(label.tolist()[0]))
    S_W = within_class_SW(data,np.array(label.tolist()[0]),clusters)
    S_B = between_class_SB(data,np.array(label.tolist()[0]),clusters)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    # print(S_W)
    # print(S_B)
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i]
        # print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        # print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(data.shape[1],1) for i in range(k)])
    # print('Matrix W:\n', W.real)
    # print('Matrix data:\n', data)
    # print(W.shape)
    # print('shape:\n', np.real(data.dot(W)))
    return np.real(data * W)
