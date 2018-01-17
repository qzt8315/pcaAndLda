import numpy as np
import scipy.io as sio
from pcaandlda import pca,lda
import knn
import time

file = 'ORL_Dataset/ORL_32_32.mat'

def KNN(testData, testLabel, trainData, trainLabel):
    # 使用二折交叉验证
    missCount = 0
    print('开始时间:', time.strftime('%Y-%m-%d %H:%M:%S'))
    start = time.time()
    for i in range(testData.shape[0]):
        out = knn.kNNClassify(testData[i],trainData,trainLabel,1)
        if out != testLabel[i]:
            missCount += 1
    t = testData
    testData = trainData
    trainData = t
    t = testLabel
    testLabel = trainLabel
    trainLabel = t
    for i in range(testData.shape[0]):
        out = knn.kNNClassify(testData[i],trainData,trainLabel,1)
        if out != testLabel[i]:
            missCount += 1
    # 运行结果显示
    print('结束时间:', time.strftime('%Y-%m-%d %H:%M:%S'))
    end = time.time()
    print('test dataset total:', len(testLabel) + len(trainLabel),
          ',miss count:' ,missCount, ',correct ratio:',
          1 - missCount/(len(testLabel) + len(trainLabel)), ',miss ratio:',
          missCount/(len(testLabel) + len(trainLabel)), ',spend:', end - start)



if __name__ == "__main__":
    data = sio.loadmat(file)
    img_data = np.matrix(np.float64(data['alls'].T))
    label = np.matrix(data['gnd'])
    print('使用pca处理数据...')
    a,b = pca(img_data)

    # 划分训练集和测试集
    trainData = np.array(a[0:-1:2, :].tolist())
    trainLabel = np.array(label[0, 0:-1:2].tolist()[0])
    testData = np.array(a[1:-1:2, :].tolist())
    testLabel = np.array(label[0, 1:-1:2].tolist()[0])
    # print(trainData.shape)
    # print(trainLabel)
    # print(testData)
    # print(testLabel)
    # print(trainData)
    # print(testData)
    KNN(testData, testLabel, trainData, trainLabel)

    print('\n使用lda处理数据...')
    a = lda(img_data, label, 64)
    # print('自己实现的lda算法为:\n',a)

    # 划分训练集和测试集
    trainData = np.array( a[0:-1:2, :].tolist() )
    trainLabel = np.array( label[0, 0:-1:2].tolist()[0] )
    testData = np.array( a[1:-1:2, :].tolist() )
    testLabel = np.array(label[0, 1:-1:2].tolist()[0])
    # print(trainData.shape)
    # print(trainLabel)
    # print(testData)
    # print(testLabel)
    # print(trainData)
    # print(testData)

    KNN(testData, testLabel, trainData, trainLabel)