import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
from control.matlab import *
import module_simulation as sim


# load infusion rate
t_f = 1200                      # I think it should be less than 1400
script_dir = os.path.dirname(__file__)
file_name = 'Eleveld_PDs/u_data.mat'
file_path = os.path.join(script_dir, file_name)
data = loadmat(file_path, squeeze_me=True)
u_arr = data['u_data'][5]
u_arr = u_arr[60:t_f]
u_arr[np.where(u_arr < 0)[0]] = 0
u_concentration = 10
u_arr = u_arr*u_concentration/60/60
# plt.plot(u_arr)
# plt.show()


# Load the  PD model
script_dir = os.path.dirname(__file__)
file_name = 'Eleveld_PDs/pd_model_datasets.csv'
file_path = os.path.join(script_dir, file_name)
df = pd.read_csv(file_path)
df['BMI'] = df['weight']/(df['height']/100)**2
df = df.replace({'gender': {0: 'male', 1: 'female'}})

subject_num = 5
gender = df['gender'][subject_num]  # zero for male and one for female
age = df['age'][subject_num]
height = df['height'][subject_num]
wgt = df['weight'][subject_num]
e0 = df['E0'][subject_num]
ke0 = df['ke0'][subject_num]
t_d = df['T_d'][subject_num]
ce50 = df['Ce50'][subject_num]
gamma = df['gamma'][subject_num]
# print(df)

# Simulation
n_u = len(u_arr)
t = np.linspace(0, t_f, n_u)
# print(u_arr)
# plt.plot(t, u_arr)
# plt.show()

pk_model = sim.eleveld_pk_model(gender, age, wgt, height)
pk_model = ct.series(sim.gm_filter(), pk_model)
cp, t_cp, x_cp = lsim(pk_model, U=u_arr, T=t)
pk_pd_lin = ct.series(pk_model, sim.pd_linear_model(ke0, ce50, t_d))
yout, tout, xout = lsim(pk_pd_lin, U=u_arr, T=t)
yout = sim.pd_model_hillfunction(yout, e0, gamma)
yout = 100 - 100*yout

u_arr = u_arr*60*60/u_concentration
figure, axis = plt.subplots(2)
axis[0].plot(tout, u_arr)
axis[0].set_xlim(0, 1200)
axis[0].set_ylim(0, 550)
axis[0].set_ylabel('Infusion rate (ml/hr)')

axis[1].plot(tout, yout)
axis[1].set_xlim(0, 1200)
axis[1].set_ylabel("DoH (WAV_cns)")
axis[1].set_xlabel("Time (seconds)")
plt.show()

# save the data
data = np.vstack((u_arr, cp, yout))
data = np.transpose(data)
columns = ['u', 'cp', 'E']
lstm_dataset = pd.DataFrame(data, columns=columns)
lstm_dataset.to_csv('lstm_dataset.csv', index=False)
print(lstm_dataset)


