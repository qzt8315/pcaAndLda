from numpy import *

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # 获取训练样本的数量

    # 计算欧式距离
    # 使用矩阵的方法进行计算
    diff = tile(newInput, (numSamples, 1)) - dataSet  # 计算差值
    squaredDiff = diff ** 2  # 平方
    squaredDist = sum(squaredDiff, axis=1)  # 相加
    distance = squaredDist ** 0.5 # 开方

    # 对计算出的距离进行排序
    # argsort()进行倒序排序
    sortedDistIndices = argsort(distance)

    classCount = {} # 定义一个字典
    for i in range(k):
        # 选择k个最小距离的对象
        voteLabel = labels[sortedDistIndices[i]]

        # 计算每个标签的计数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 返回最大数量标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex