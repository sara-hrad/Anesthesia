import control as ct
import gym
import numpy as np
from control.matlab import *
from gym import spaces
from matplotlib import pyplot as plt


class AnestheisaEnv(gym.Env):
    def __init__(self, age, weight, height, lbm, t_s):
        super(AnestheisaEnv, self).__init__()
        # Define the demographic parameters of patient simulator
        self.age = age
        self.weight = weight
        self.height = height
        self.lbm = lbm
        self.t_s = t_s
        self.observation_space = spaces.Box(low=0, high=100, shape=(101,), dtype=np.float32)

        # Define the continuous action space
        self.action_space = spaces.Box(low=0, high=1.5, shape=(16,), dtype=np.float32)

        # Initialize the state
        self.state = 0
        self.concentrations = np.zeros((5, ), dtype=np.float32)

    def pk_model(self):
        v1p = 4.27    # [l]
        v2p = 18.9 - 0.391 * (self.age - 53)   # [l]
        v3p = 238     # [l]
        cl1p = 1.89 + 0.0456 * (self.weight - 77) - 0.0681 * (self.lbm - 59) + 0.0264 * (self.height - 177)   # [l/min]
        cl2p = 1.29 - 0.024 * (self.age - 53)   # [l/min]
        cl3p = 0.836  # [l/min]

        clearance = np.array([cl1p, cl2p, cl3p])
        volume = np.array([v1p, v2p, v3p])
        n = len(clearance)
        k1 = clearance / volume[0]
        k2 = clearance[1:] / volume[1:]
        a = np.vstack((np.hstack((-np.sum(k1), k1[1:])), np.hstack((np.transpose(k2)[:, None], -np.diag(k2)))))
        b = np.vstack(([1 / volume[0]], np.zeros((n - 1, 1))))
        c = np.array([[1, 0, 0]])
        d = np.array([[0]])
        a = a / 60
        pk_sys = ct.ss(a, b, c, d)
        return pk_sys

    def pd_linear_model(self):
        ke0 = 0.456    # [min^(-1)]
        t_d = 20
        num = np.array([ke0])
        den = np.array([1, ke0])
        sys = ct.tf(num, den)
        time_delay_pad_app = ct.tf(ct.pade(t_d)[0], ct.pade(t_d)[1])
        pd_lin_sys = ct.series(sys, time_delay_pad_app)
        return pd_lin_sys

    def pd_model_hillfunction(self, ce):
        e0 = 100
        gamma = 2
        ce50 = 4.16
        ce = 0 if ce < 0 else ce
        e = e0 - e0*ce**gamma/(ce**gamma + ce50**gamma)
        return e

    def step(self, action):
        # Clip the action to be within the action space bounds
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])

        # Update the state based on the action
        pk_pd_lin = ct.series(self.pk_model(), self.pd_linear_model())

        yout, tout, xout = lsim(pk_pd_lin,
                                U=action*np.ones(self.t_s+1),
                                T=np.linspace(0, self.t_s, self.t_s+1),
                                X0=self.concentrations)
        self.concentrations = xout[-1]
        self.state = np.array(self.pd_model_hillfunction(yout[-1]))
        self.state = (100-self.state)/100
        self.state = np.clip(self.state, self.observation_space.low[0], self.observation_space.high[0])

        # Define the reward function (example: reward is higher when state is closer to 1)
        reward = 0.5 - np.abs(self.state-0.5) - 0.2*action - 0.2*np.max([0, self.state - 0.5])
        epsilon = 0.05
        # Check if the episode is done
        if np.abs(self.state-0.5) < epsilon:
            done = True
        else:
            done = False
        return np.array([self.state]), np.array(reward), done

    def reset(self):
        # Reset the environment to a random initial state
        self.state = 0.01*np.random.rand(1)
        self.concentrations = np.zeros((5,), dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        # Implement rendering here if needed
        pass

    def close(self):
        # Implement any cleanup code here if needed
        pass


def main():
    t_s = 1
    t_f = 600
    age = 45     #yr
    weight = 64  #kg
    height = 171 #cm
    lbm = 52     #kg/m^2
    env = AnestheisaEnv(age, weight, height, lbm, t_s)
    obs = env.reset()
    # print(type(obs))
    doh_arr = []
    infusion_arr = []

    for i in range(t_f):
        if i < 20:
            action = 0.6
        else:
            action = 0.23
        obs, reward, done = env.step(action)
        doh_arr.append(obs)
        infusion_arr.append(action)
        print(f"State: {obs}, Reward: {reward}, Done: {done}, Action:{action}")
    print(type(doh_arr))
    figure, axis = plt.subplots(2)
    axis[0].plot(range(t_f), infusion_arr)
    axis[0].set_xlim(0, t_f)
    axis[0].set_ylim(0, 1.6)
    axis[0].set_ylabel('Infusion rate (mg/s)')

    axis[1].plot(range(t_f), doh_arr)
    axis[1].axhspan(0.45, 0.55, color='green', alpha=0.75, lw=0)
    axis[1].set_xlim(0, t_f)
    axis[1].set_ylabel("DoH (0 - 1)")
    axis[1].set_xlabel("Time (seconds)")
    plt.show()
    env.close()


if __name__ == "__main__":
    main()


