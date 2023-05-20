# coding=UTF-8
import functools

from skimage.measure import compare_ssim, compare_psnr, compare_mse
import scipy.misc
from glob import glob
from utils import *

dataset_name = "hyperspectra"
ypre_name = "11_1"

def evaluate_one(origin_img, pre_img, method="PSNR"):
    """
    evaluate between origin and predict images
    Args:
        origin_img: the path of origin image
        pre_img: the path of predict image
        method: evaluate method, string: PSNR, SSIM, MSE

    Returns: the value of evaluate with the method

    """

    y = scipy.misc.imread(origin_img, flatten=True)
    y_pre = scipy.misc.imread(pre_img, flatten=True)
    # print(y.shape)
    m, n = y.shape
    y = scipy.misc.imresize(y[(m-n): , :], [256, 256])

    if method == "PSNR":
        return compare_psnr(y, y_pre)
    elif method == "SSIM":
        return compare_ssim(y, y_pre)  # 对于多通道图像(RGB,HSV等)关键词multichannel要设置为True
    elif method == "MSE":
        return compare_mse(y, y_pre)
    else:
        print("method error")

def save_evaluate_res(path, method, output_name, scores):
    with open(path, 'a') as f:
        f.write(method + " " + output_name + " ")
        for i in scores:
            f.write(str(i) + " ")
        f.write("\n")


if __name__ == '__main__':
    '''
    # test one
    path_B = "./test_ouput/7_1/test_0001.png"
    path_A = "./datasets/hyperspectra/test/test_Y/10-1.jpg"
    res = evaluate_one(path_A, path_B)
    print('the res of evaluate is :{}'.format(res))
    ### the res of evaluate is :17.1508396762
    '''
    #---------------------get all scores of samples------------------
    y_paths = glob("./datasets/{}/test/test_Y/*.jpg".format(dataset_name))
    y_paths = np.array(sorted(y_paths, key=functools.cmp_to_key(compare)))
    ypre_paths = glob("./test_ouput/{}/*.png".format(ypre_name))
    print(len(y_paths), len(ypre_paths))  # (30, 30)
    print(y_paths)
    print(ypre_paths)

    methods = ["PSNR", "SSIM", 'MSE']
    for method in methods:
        res = []
        for i in range(len(y_paths)):
            res.append(evaluate_one(y_paths[i], ypre_paths[i], method=method))
        print(res)
        save_evaluate_res("./evaluate_res.txt", method, ypre_name, res)
