####VISUALIXE############
import pickle
import matplotlib.pyplot as plt
# file = open("/home/asyed/SS-LSTM/traj_segnet_1000_zara1",'rb')
# file1 = open("/home/asyed/SS-LSTM/traj_zara1",'rb')

file = open("E:/lstm预测/ss-lstm_0528/traj_SS_LSTM_logmap_1000_Zara2_p2v_0528",'rb+')
#file1 = open("/home/asyed/SS-LSTM/traj_zara1",'rb')
#object_file = pickle.load(file)
object_file1=pickle.load(file)
for i in range(0,97):
 #plt.plot(person_input_1[i][:,0], person_input_1[i][:,1],"b+",label='observed')
 #plt.plot(object_file[i][:,0],object_file[i][:,1],"r+",label='Segnet')
 #plt.plot(predicted_output[i][:,0],predicted_output[i][:,1],"r+",label='Segnet')
 #plt.plot(expected_ouput_1[i][:,0],expected_ouput_1[i][:,1],"g+",label='ground_truth' )
#
#plt.plot(person_input_1[i][:, 0], person_input_1[i][:, 1], "b+", label='observed')
  plt.plot(object_file1[i][:, 0], object_file1[i][:, 1], "y+", label='predicted_SS_LSTM')
 # plt.plot(expected_ouput_1[i][:, 0], expected_ouput_1[i][:, 1], "g+", label='ground_truth')
#
plt.title(i)
plt.legend()
plt.show()
########################