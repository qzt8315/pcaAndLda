import numpy as np
import scipy.io as sio
import math

file = 'ORL_Dataset/ORL_32_32.mat'
k = 2 # k折交叉验证

if __name__ == "__main__":
    data = sio.loadmat(file)
    img_data = np.matrix(np.float64(data['alls'].T))
    label = np.matrix(data['gnd'])
    errCount = 0
    for i in range(40):
        test_data = img_data[i * 10+5:i * 10 + 10]
        for k in range(5):
            t_data = test_data[k]
            min = None
            minnorm = None
            for j in range(40):
                t = img_data[j * 10:j * 10 + 5]
                test = img_data[j * 10+5:j * 10 + 10]
                origin = (test[0] + test[1] + test[2] + test[3] + test[4])/5
                w = (t*t.T + 0.000000000001* np.matrix(np.eye(t.shape[0]))).I*t*origin.T  # 计算欧米伽
                norm = math.pow(np.linalg.norm((t_data.T - t.T*w)/5, ord=2), 2)  # 计算差距
                if minnorm is None or minnorm > norm:
                    min = j+1
                    minnorm = norm
            if min != label[0,(i*5+k)*2]:
                errCount += 1

    for i in range(40):
        test_data = img_data[i * 10:i * 10 + 5]
        for k in range(5):
            t_data = test_data[k]
            min = None
            minnorm = None
            for j in range(40):
                t = img_data[j * 10 + 5:j * 10 + 10]
                test = img_data[j * 10:j * 10 + 5]
                origin = (test[0] + test[1] + test[2] + test[3] + test[4])/5
                w = (t*t.T + 0.000000000001* np.matrix(np.eye(t.shape[0]))).I*t*origin.T  # 计算欧米伽
                norm = math.pow(np.linalg.norm((t_data.T - t.T*w)/5, ord=2), 2)  # 计算差距
                if minnorm is None or minnorm > norm:
                    min = j+1
                    minnorm = norm
            if min != label[0,(i*5+k)*2]:
                errCount += 1

    print('correct count:', 400-errCount,
          'correct ratio:', 1-errCount/400,
          ',error count:', errCount,
          ',cerror ratio:', errCount/400)