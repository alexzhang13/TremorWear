#!/usr/bin/env python

# make libs folder visible
import sys
sys.path.append('../')

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250 import mpu9250
from libs import mcp3008

import time
import argparse
import datetime

from scanf import scanf
import numpy as np
import matplotlib.pyplot as plt

from src import preprocess

# CONSTANTS #
DEG_TO_RAD = np.pi/180
G_TO_MPERS = 9.80665
SAMPLE_RATE = 500


# Arg Parse
parser = argparse.ArgumentParser(description='Main Class for TremorWear Training and Testing')
parser.add_argument("--agent", type=str, default="LSTM", help="Agent to Run")
parser.add_argument("--length", type=int, default=10000, help="Length of Tremor Recording")
parser.add_argument("--plabel", type=int, default=0, help="Patient Label")
parser.add_argument("--record", dest="record_data", action="store_true", help="Record Patient Data")
parser.add_argument("--read", dest="record_data", action="store_false", help="Playback Patient Data from File")
parser.set_defaults(record_data=True)
parser.add_argument("--readfile", type=str, default="saved_data_0", help="Name of Recording for Playback")
parser.add_argument("--plot", dest="plot_data", action="store_false", help="Plot Data")
parser.set_defaults(plot_data=False)

args = parser.parse_args()

def main():
    processor = preprocess.SignalProcessor(sample_rate=SAMPLE_RATE)

    if args.record_data is True:
        print("Flag: Recording Data to PNumber: {}".format(args.pnumber))
        imu = mpu9250()
        print("IMU Initialized")
        start = time.time()

        # Record IMU Data
        ax, ay, az, gx, gy, gz = record(imu, args.length)

        end = time.time()
        print("Elapsed: {:.3f}\tAvg Freq(hz): {:.3f}".format(end - start, args.length / (end - start)))

    else:
        print("Flag: Reading Data from File: {}".format(args.readfile))
        ax, ay, az, gx, gy, gz = read(args.readfile)

    plots = [ax, ay, az, gx, gy, gz]
    save(plots, "../data/{}".format(args.plabel))

    if args.plot_data is True:
        processor.FilterTest(gx, "Gyro X")
        plot([gx])


def record(imu, length):
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    for i in range(length):
        a = imu.accelgyro
        (axt, ayt, azt), (gxt, gyt, gzt) = a
        ax.append(axt*G_TO_MPERS)
        ay.append(ayt*G_TO_MPERS)
        az.append(azt*G_TO_MPERS)
        gx.append(gxt*DEG_TO_RAD)
        gy.append(gyt*DEG_TO_RAD)
        gz.append(gzt*DEG_TO_RAD)

    return ax, ay, az, gx, gy, gz

def read(filename):
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    with open("../data/" + filename + ".txt") as data:
        for line in data:
            # time[ms], ax, ay, az, gx, gy, gz
            _, axt, ayt, azt, gxt, gyt, gzt = scanf("%f %f %f %f %f %f %f", line)
            ax.append(axt)
            ay.append(ayt)
            az.append(azt)
            gx.append(gxt)
            gy.append(gyt)
            gz.append(gzt)

    return ax, ay, az, gx, gy, gz


def plot(plots):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" Original Graph in Spatial Domain ", fontsize=18)
    ax.set_ylabel("Gyro: [rad/s], Accel: [m/s]")
    ax.set_xlabel("Time [ms]")

    for i in range(len(plots)):
        plt.plot(np.arange(0.0, args.length*1000/(1000*SAMPLE_RATE), 1/SAMPLE_RATE), plots[i])
    plt.legend(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], loc='upper left')

    plt.show()
    fig.savefig("../imgs/original" + ".png")

def save(data, name):
    now = datetime.datetime.now()

    file = open(name + "_{}.txt".format(now.isoformat()), 'w')
    for i in range(len(data[0])):
        file.write("{} {} {} {} {} {} {}\n".format((1000*i/SAMPLE_RATE), data[0][i], data[1][i], data[2][i], data[3][i],
                                                 data[4][i], data[5][i]))

if __name__ == "__main__":
    main()
