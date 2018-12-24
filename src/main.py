#!/usr/bin/env python

# make libs folder visible
import sys
sys.path.append('../')

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250 import mpu9250
from libs import mcp3008
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from src import preprocess

# CONSTANTS #
DEG_TO_RAD = np.pi/180
G_TO_MPERS = 9.80665

def main():
    imu = mpu9250()
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    processor = preprocess.SignalProcessor(sample_rate=0.25)

    for i in range(50):
        a = imu.accel
        #a = (1, 1, 1)
        axt, ayt, azt = a
        ax.append(axt*G_TO_MPERS)
        ay.append(ayt*G_TO_MPERS)
        az.append(azt*G_TO_MPERS)

        b = imu.gyro
        #b = (1.1 * np.sin(3 * i), 6 * np.sin(32 * i), 1.3 * np.sin(1.2 * i))
        gxt, gyt, gzt = b
        gx.append(gxt*DEG_TO_RAD)
        gy.append(gyt*DEG_TO_RAD)
        gz.append(gzt*DEG_TO_RAD)

        print("a {:.3f} {:.3f} {:.3f}".format(*a))
        print("g {:.3f} {:.3f} {:.3f}".format(*b))

        sleep(0.25)

    plots = [ax, ay, az, gx, gy, gz]
    #processor.FourierSequences(plots)
    plot(plots)

def plot(plots):
    for i in range(len(plots)):
        plt.plot(plots[i])
    plt.legend(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
