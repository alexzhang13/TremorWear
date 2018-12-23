#!/usr/bin/env python

# change root to one up
import os
os.chdir(os.getcwd())
os.chdir("..")

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250.mpu9250 import mpu9250
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

imu = mpu9250()

def main():
    ax, ay, az, gx, gy, gz = []

    for i in range(50):
        a = imu.accel
        print("a {:.3f} {:.3f} {:.3f}".format(*a))
        axt, ayt, azt = a
        ax.append(axt)
        ay.append(ayt)
        az.append(azt)

        b = imu.gyro
        print("g {:.3f} {:.3f} {:.3f}".format(*b))
        gxt, gyt, gzt = b
        gx.append(gxt)
        gy.append(gyt)
        gz.append(gzt)

        sleep(0.25)

    plots = [ax, ay, az, gx, gy, gz]
    plot(plots)

def plot(plots):
    for i in range(len(plots)):
        plt.plot(plots[i])
    plt.legend(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
