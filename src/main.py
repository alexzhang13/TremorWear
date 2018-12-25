#!/usr/bin/env python

# make libs folder visible
import sys
sys.path.append('../')

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250 import mpu9250
from libs import mcp3008
import time

import numpy as np
import matplotlib.pyplot as plt

from src import preprocess

# CONSTANTS #
DEG_TO_RAD = np.pi/180
G_TO_MPERS = 9.80665
SAMPLE_RATE = 1000

def main():
    imu = mpu9250()
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    processor = preprocess.SignalProcessor(sample_rate=SAMPLE_RATE)

    start = time.time()

    for i in range(1000):
        a = imu.accel
        axt, ayt, azt = a
        ax.append(axt*G_TO_MPERS)
        ay.append(ayt*G_TO_MPERS)
        az.append(azt*G_TO_MPERS)

        b = imu.gyro
        #b = (0, 10 * np.cos(5 * np.pi * (1/SAMPLE_RATE) * i) + 5 * np.sin(10 * np.pi * (1/SAMPLE_RATE) * i), 0)
        gxt, gyt, gzt = b
        gx.append(gxt*DEG_TO_RAD)
        gy.append(gyt*DEG_TO_RAD)
        gz.append(gzt*DEG_TO_RAD)

        #print("t {:.3f}".format(i))
        #print("a {:.3f} {:.3f} {:.3f}".format(*a))
        #print("g {:.3f} {:.3f} {:.3f}".format(*b))

    end = time.time()
    print("Elapsed: {:.3f}\tFreq(hz): {:.3f}".format(end-start, SAMPLE_RATE/(end-start)))

    plots = [ax, ay, az, gx, gy, gz]
    processor.FourierTest(gy, "Gyro Y")
    plot(plots)

def plot(plots):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" Original Graph in Spatial Domain ", fontsize=18)
    ax.set_ylabel("Gyro: [rad/s], Accel: [m/s]")
    ax.set_xlabel("Time [ms]")

    for i in range(len(plots)):
        plt.plot(plots[i])
    plt.legend(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], loc='upper left')

    plt.show()
    fig.savefig("../imgs/original" + ".png")

if __name__ == "__main__":
    main()
