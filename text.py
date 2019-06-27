import numpy
import pickle
for i in range(1,11):
    name='data_dir_'+str(i)
    locals()[name] = r'C:\Users\asus\Desktop\lstm项目\ss-lstm_0529\ss-lstm_0529\datadut\0'+str(i)
    # print(name)
    # print(locals()[name])
    print(data_dir_1)



# filename = 'traj_SS_LSTM_logmap_1000_Zara2'
# with open(filename,'wb+') as f:
#     contents=f.read()
#     # print(contents)
#     words=contents.split()
#     num=len(words)
#     print(num)