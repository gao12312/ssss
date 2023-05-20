# from scipy.misc import imread
# X = imread('./datasets/hyperspectra/train/X/1-1-1.jpg')
# Y = imread('./datasets/hyperspectra/train/Y/1-1.jpg')
# print (X.shape)
# print (Y.shape)
import time
import numpy as np

x_origin = np.zeros((314, 1749,1600))      #dim*height*width
INPUT_C_DIM = 314
start_time = time.time()
print("x_origin :", x_origin.shape)



x_processed = []
for eve_batch in range(1):
    # x_t = np.expand_dims(x_origin[eve_batch * INPUT_C_DIM], axis=-1)  # (1024, 543)
    x_t = [np.expand_dims(x_origin[i], -1) for i in range(eve_batch * INPUT_C_DIM, (eve_batch+1) * INPUT_C_DIM)]
    #expand dim for evey x_origin
    print("x_t shape is :", np.array(x_t).shape)

    x_t = np.concatenate(x_t, -1)

    # for i in range(1, INPUT_C_DIM):
    #
    #     print("time1 :", time.time() - start_time)
    #     start_time = time.time()
    #     img = x_origin[eve_batch * INPUT_C_DIM + i]
    #     print("time2 :", time.time() - start_time)
    #     start_time = time.time()
    #     img_ = np.expand_dims(img, axis=-1)
    #     print("time3 :", time.time() - start_time)
    #     start_time = time.time()
    #     x_t = np.concatenate((x_t, img_), axis=-1)
    #     print("time4 :", time.time() - start_time)
    #     start_time = time.time()
    x_processed.append(x_t)

x_processed = np.array(x_processed)
print(x_processed.shape)
