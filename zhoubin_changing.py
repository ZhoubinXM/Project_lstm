'''
2019.6.25 更改代码
'''
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, LSTM, GRU, Merge
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.merge import Concatenate
from keras.models import load_model
from keras.layers import merge
from keras.layers.core import Permute
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
import time
import cv2
from utils import circle_group_model_input, log_group_model_input, group_model_input
from utils import get_traj_like, get_obs_pred_like
from utils import person_model_input, model_expected_ouput, preprocess
from vehicle_utils import preprocess_vehicle,get_traj_like_vehicle,get_obs_vehicle_like,vehicle_model_input,vehicle_model_expected_ouput,veh2ped_group_model_input,veh2ped_circle_group_model_input
from keras.callbacks import History
import heapq
import data_process as dp
import veh_data_process as vdp
import os
from keras.utils import plot_model, multi_gpu_model
import pickle
import matplotlib.pyplot as plt
from keras import backend as k
from keras.applications.vgg16 import VGG16

os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 只有编号为1的GPU对程序是可见的，在代码中gpu[0]指的就是这块儿GPU

def calculate_FDE(test_label, predicted_output, test_num, show_num):
    total_FDE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])

    show_FDE = heapq.nsmallest(show_num, total_FDE)
    show_FDE = np.reshape(show_FDE, [show_num, 1])

    return np.average(show_FDE)

def calculate_ADE(test_label, predicted_output, test_num, predicting_frame_num, show_num):
    total_ADE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        ADE_temp = 0.0
        for j in range(predicting_frame_num):
            ADE_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
        ADE_temp = ADE_temp / predicting_frame_num
        total_ADE[i] = ADE_temp

    show_ADE = heapq.nsmallest(show_num, total_ADE)
    show_ADE = np.reshape(show_ADE, [show_num, 1])

    return np.average(show_ADE)

def image_tensor(data_dir,data_str, frame_ID):
    #print('Frame id: ')
    #print(frame_ID)
    length=6
    id='frame'
    b=str(frame_ID)
    s=len(b)
    for i in range(0,6-s):
        # id=id+'0'
        id=id
    id=id+str(frame_ID)
    #print(id)
    #img_dir = data_dir + data_str + str(frame_ID) + '.jpg'
    img_dir = data_dir+'/'+id+'.jpg'
    #print(img_dir)
    img = cv2.imread(img_dir)
    #print(img)
    img = cv2.resize(img, (720, 576))
    # img = cv2.resize(img, (224, 224))
    #    out = tf.stack(img)
    return img

def CNN(img_rows, img_cols, img_channels=3):
    model = Sequential()
    img_shape = (img_rows, img_cols, img_channels)
    model.add(Conv2D(96, kernel_size=11, strides=4, input_shape=img_shape, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    #    model.add(Conv2D(384, kernel_size=3, strides=1, padding="same"))
    #    model.add(Conv2D(384, kernel_size=3, strides=1, padding="same"))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    return model

observed_frame_num = 8
predicting_frame_num = 12

hidden_size = 128
tsteps = observed_frame_num#8
dimensions_1 = [720, 576] #[720,576] #eth hotel
# dimensions_1 = [224, 224]
dimensions_2 = [640, 480]  #eth univ
img_width_1 = 720
img_height_1 = 576
img_width_2 = 640
img_height_2 = 480

batch_size = 20

veh_neighborhood_size=128
grid_veh_size=4
neighborhood_size = 32
grid_size = 4
neighborhood_radius = 32
grid_radius = 4
# grid_radius_1 = 4
grid_angle = 45
circle_map_weights = [1, 1, 1, 1, 1, 1, 1, 1]

opt = optimizers.RMSprop(lr=0.003)

for i in range(1,11):
    data_dir='data_dir_'+str(i)
    veh_data='veh_data_'+str(i)
    numveh  = 'numveh_' +str(i)
    raw_data='raw_data_'+str(i)
    numPeds = 'numPeds_'+str(i)
    check_veh='check_veh_'+str(i)
    check='check_'+str(i)
    obs_veh='obs_veh_'+str(i)
    obs='obs_'+str(i)
    pred_veh='pre_veh_'+str(i)
    pred='pred_'+str(i)
    vehicle_input='vehicle_input_'+str(i)
    person_input='person_input_'+str(i)
    group_circle='group_circle_'+str(i)
    gruop_grid_veh2ped='gruop_grid_veh2ped_'+str(i)
    vehicle_expect_output='vehicle_expect_output_'+str(i)
    expected_ouput='expected_ouput_'+str(i)
    locals()[data_dir] = r'C:\Users\asus\Desktop\lstm项目\ss-lstm_0529\ss-lstm_0529\datadut\0'+str(i)
    locals()[veh_data], locals()[numveh] = preprocess_vehicle(locals()[data_dir])
    locals()[raw_data], locals()[numPeds]= preprocess(locals()[data_dir])
    locals()[check_veh] = vdp.veh_DataProcesser(locals()[data_dir], observed_frame_num, predicting_frame_num)
    locals()[check] = dp.DataProcesser(locals()[data_dir], observed_frame_num, predicting_frame_num)
    locals()[obs_veh] = locals()[check_veh].obs
    locals()[obs] = locals()[check].obs
    locals()[pred_veh]= locals()[check_veh].pred
    locals()[pred] = locals()[check].pred
    locals()[vehicle_input] = vehicle_model_input(locals()[obs_veh], observed_frame_num)
    locals()[person_input] = person_model_input(locals()[obs], observed_frame_num)
    locals()[group_circle] = circle_group_model_input(locals()[obs], observed_frame_num, neighborhood_size, dimensions_1,
                                                      neighborhood_radius, grid_radius, grid_angle, circle_map_weights,
                                                      locals()[raw_data])
    locals()[gruop_grid_veh2ped] = veh2ped_circle_group_model_input(locals()[obs], observed_frame_num, dimensions_1,
                                                                    veh_neighborhood_size, grid_radius, grid_angle,
                                                                    locals()[raw_data],locals()[veh_data])  # 圆形区域，若要矩形区域改成veh2ped_grid_model_input
    locals()[vehicle_expect_output] = vehicle_model_expected_ouput(locals()[pred_veh], predicting_frame_num) # 期望输出
    locals()[expected_ouput] = model_expected_ouput(locals()[pred], predicting_frame_num)


# 矩形区域只写一个做存档
group_grid_1 = group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_1)
group_log_3 = log_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                    grid_radius, grid_angle, circle_map_weights, raw_data_3)    #logmap同样是存档
# print(data_dir_1)

def all_run(epochs, predicting_frame_num, leave_dataset_index, map_index, show_num, min_loss):
    if map_index == 1:
         group_input_1 = group_circle_1
         group_input_2 = group_circle_2
         group_input_3 = group_circle_3
         group_input_4 = group_circle_4
         group_input_5 = group_circle_5
         group_input_6 = group_circle_6
         group_input_7 = group_circle_7
         group_input_8 = group_circle_8
         group_input_9 = group_circle_9
         group_input_10 = group_circle_10
         veh2ped_group_input_1 = gruop_grid_veh2ped_1
         veh2ped_group_input_2 = gruop_grid_veh2ped_2
         veh2ped_group_input_3 = gruop_grid_veh2ped_3
         veh2ped_group_input_4 = gruop_grid_veh2ped_4
         veh2ped_group_input_5 = gruop_grid_veh2ped_5
         veh2ped_group_input_6 = gruop_grid_veh2ped_6
         veh2ped_group_input_7 = gruop_grid_veh2ped_7
         veh2ped_group_input_8 = gruop_grid_veh2ped_8
         veh2ped_group_input_9 = gruop_grid_veh2ped_9
         veh2ped_group_input_10 = gruop_grid_veh2ped_10

    person_input = np.concatenate(
        (person_input_1, person_input_2, person_input_3, person_input_4, person_input_5, person_input_6, person_input_7, person_input_8, person_input_9))
    expected_ouput = np.concatenate(
        (expected_ouput_1, expected_ouput_2, expected_ouput_3, expected_ouput_4,expected_ouput_5,expected_ouput_6,expected_ouput_7,expected_ouput_8,expected_ouput_9))
    group_input = np.concatenate((group_input_1, group_input_2, group_input_3, group_input_4, group_input_5, group_input_6, group_input_7, group_input_8, group_input_9))
    vehicle_expect_output=np.concatenate((vehicle_expect_output_1,vehicle_expect_output_2,vehicle_expect_output_3,vehicle_expect_output_4,vehicle_expect_output_5,vehicle_expect_output_6,vehicle_expect_output_7,vehicle_expect_output_8,vehicle_expect_output_9))
    veh2ped_group_input=np.concatenate((veh2ped_group_input_1,veh2ped_group_input_2,veh2ped_group_input_3,veh2ped_group_input_4,veh2ped_group_input_5,veh2ped_group_input_6,veh2ped_group_input_7,veh2ped_group_input_8,veh2ped_group_input_9))
#    scene_input = np.concatenate((img_1, img_2, img_3, img_4))
    #test_input = [img_5, group_input_5, person_input_5]
    test_input = [group_input_10, person_input_10]
    test_output = expected_ouput_10
    print('data load done!')

    scene_scale = CNN(dimensions_1[1], dimensions_1[0])
    scene_scale.add(RepeatVector(tsteps))
    scene_scale.add(GRU(hidden_size,
                        input_shape=(tsteps, 512),
                        batch_size=batch_size,#all_run(epochs, predicting_frame_num, leave_dataset_index, map_index, show_num, min_loss)
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))

    group_model = Sequential()
    group_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 64)))#全连接层
    group_model.add(GRU(hidden_size,
                        input_shape=(tsteps, int(neighborhood_radius / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
    veh2ped_group_model = Sequential()
    veh2ped_group_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 256)))  # 全连接层
    veh2ped_group_model.add(GRU(hidden_size,
                        input_shape=(tsteps, int(veh_neighborhood_size / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
    person_model = Sequential()#定义时间序列
    person_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 2)))
    person_model.add(GRU(hidden_size,
                         input_shape=(tsteps, 2),
                         batch_size=batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.2))
    #myself
    # model.add(Merge([scene_scale, group_model, person_model], mode='sum'))
    model = Sequential()
    model.add(Merge([ group_model,
                      person_model], mode='sum'))
    model.add(RepeatVector(predicting_frame_num))
    model.add(GRU(128,
                input_shape=(predicting_frame_num, 2),
                batch_size=batch_size,
                return_sequences=True,
                stateful=False,
                dropout=0.2))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=opt)
    # parallel=multi_gpu_model(model,gpus=2)
    print(model.summary())
    for i in range(epochs):
        # history = model.fit([scene_input, group_input, person_input], expected_ouput,
                            #batch_size=batch_size,
                            #epochs=1,
                            #verbose=1,
                            #shuffle=False)
        history = model.fit([group_input, person_input], expected_ouput,
                            batch_size=batch_size,
                            epochs=1,
                            verbose=1,
                            shuffle=False)
        loss = history.history['loss']
        if loss[0] < min_loss:
            break
        else:
            continue
        model.reset_states()
        # parallel.reset_states()

    # model.save('ss_map_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'testing_seg.h5')
    # plot_model(model, to_file='model.png')
    # model.save('testing_SS_LSTM_logmap_Zara1_1000epoc_batchsize_20.h5')
    model.save_weights("model_weights_ss_lstm_ZARA2_1000epochs")
    print('Predicting...')
    predicted_output = model.predict(test_input, batch_size=batch_size)
    with open("traj_SS_LSTM_logmap_1000_Zara2", "wb+") as f:
          pickle.dump(predicted_output, f)   # 将对象predicted_output保存到f文件中去
    print('Predicting Done!')
    print('Calculating Predicting Error...')
    mean_FDE = calculate_FDE(test_output, predicted_output, len(test_output), show_num)
    mean_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, show_num)
    all_FDE = calculate_FDE(test_output, predicted_output, len(test_output), len(test_output))
    all_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, len(test_output))
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'mean ADE:', mean_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'mean FDE:', mean_FDE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all ADE:', all_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all FDE:', all_FDE)

    return predicted_output, mean_ADE, mean_FDE, all_ADE, all_FDE
all_run(1000, predicting_frame_num, 0, 1, 1, 0)