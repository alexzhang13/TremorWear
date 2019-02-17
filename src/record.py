#!/usr/bin/env python

# make libs folder visible
import os
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
sdata_dict = {
    0: "Accel X",
    1: "Accel Y",
    2: "Accel Z",
    3: "Gyro X",
    4: "Gyro Y",
    5: "Gyro Z"
}

# Arg Parse
parser = argparse.ArgumentParser(description='Main Class for TremorWear Training and Testing')
parser.add_argument("--agent", type=str, default="LSTM", help="Agent to Run")
parser.add_argument("--length", type=int, default=10000, help="Length of Tremor Recording")
parser.add_argument("--plabel", type=str, default="Patient1", help="Patient Label")
parser.add_argument("--record", dest="record_data", action="store_true", help="Record Patient Data")
parser.add_argument("--read", dest="record_data", action="store_false", help="Playback Patient Data from File")
parser.set_defaults(record_data=True)
parser.add_argument("--path", type=str, default="PatientName_Action", help="Name/Folder of Recording for Playback")
parser.add_argument("--plot", dest="plot_data", action="store_true", help="Plot Data")
parser.set_defaults(plot_data=False)
parser.add_argument("--plotall", dest="plot_all", action="store_true", help="Plot All Data in a Folder")
parser.set_defaults(plot_all=False)
parser.add_argument("--window", dest="window", action="store_true", help="Print Windowed FFT of Sequence")
parser.set_defaults(window=False)
parser.add_argument("--window_num", type=int, default=0, help="Number denoting type of Sensor Data to Window"
                "(Only Applies When --window is Called: 0 --> ax, 1 --> ay, 2 --> az, 3 --> gx, 4 --> gy, >=5 --> gz")
parser.add_argument("--window_size", type=int, default=250, help="Shifting Window Size")

args = parser.parse_args()


def main():
    processor = preprocess.SignalProcessor(sample_rate=SAMPLE_RATE)
    print("Flag: Recording Data to PLabel: {}".format(args.plabel))

    if args.record_data is True:
        imu = mpu9250()
        print("IMU Initialized")
        start = time.time()

        # Record IMU Data
        ax, ay, az, gx, gy, gz = record(imu, args.length)

        end = time.time()
        print("Elapsed: {:.3f}\tAvg Freq(hz): {:.3f}".format(end - start, args.length / (end - start)))

        plots = [ax, ay, az, gx, gy, gz]
        save(plots, "../data/{}".format(args.plabel), args.length / (end - start))
    else:
        graph_imu(args.plot_all, args.path, args.plot_data, args.window, args.window_num, args.window_size, processor)


# Function for Graphing IMU Data - Option to Plot One File or All Files from One Folder
def graph_imu(plot_all, filepath, plot_data, window, window_num, window_size, processor):
    if plot_all:
        print("Flag: Reading All Data from Folder: {}".format(filepath))
        path = "../data/" + filepath
        for filename in os.listdir(path):
            freq, ax, ay, az, gx, gy, gz = read_frompath(path + "/" + filename)
            plotsa = [ax, ay, az]
            plotsg = [gx, gy, gz]
            plot(plotsa, plotsg, filename)
    else:
        print("Flag: Reading Data from File: {}".format(args.path))
        freq, ax, ay, az, gx, gy, gz = read(args.path)
        plotsa = [ax, ay, az]
        plotsg = [gx, gy, gz]
        plot(plotsa, plotsg, "Filler Name")
        if plot_data is True:
            plotG(gx, "Gyro X")
            processor.FilterTest(gx, "Gyro X")
            plotG(gy, "Gyro Y")
            processor.FilterTest(gy, "Gyro Y")
            plotG(gz, "Gyro Z")
            processor.FilterTest(gz, "Gyro Z")
        if window is True:
            if window_num > 2:
                data = plotsg[min(5, window_num) % 3]
            else:
                data = plotsa[window_num]
            window_FFT(data, window_size, sdata_dict[window_num])

# Function for Recording IMU Data
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


# Function for Graphing a Shifting FFT Window of Input Data [One Parameter]
def window_FFT(sensor_data, window_size, name):
    pass


# Given File Name from Data Folder
def read(filename):
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    with open("../data/" + filename + ".txt") as data:
        freq = scanf("%f", data.readline())
        for line in data:
            # time[ms], ax, ay, az, gx, gy, gz
            _, axt, ayt, azt, gxt, gyt, gzt = scanf("%f %f %f %f %f %f %f", line)
            ax.append(axt)
            ay.append(ayt)
            az.append(azt)
            gx.append(gxt)
            gy.append(gyt)
            gz.append(gzt)

    return freq, ax, ay, az, gx, gy, gz

# Given Full Path
def read_frompath(filename):
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    with open(filename) as data:
        freq = scanf("%f", data.readline())
        for line in data:
            # time[ms], ax, ay, az, gx, gy, gz
            _, axt, ayt, azt, gxt, gyt, gzt = scanf("%f %f %f %f %f %f %f", line)
            ax.append(axt)
            ay.append(ayt)
            az.append(azt)
            gx.append(gxt)
            gy.append(gyt)
            gz.append(gzt)

    return freq, ax, ay, az, gx, gy, gz

def plot(plotsa, plotsg, name):
    fig = plt.figure(figsize=(12.0, 10.0))
    ax = fig.add_subplot(2, 1, 1)

    ax.set_title(" Accelerometer Readings in Spatial Domain ", fontsize=18)
    ax.set_ylabel("Accel: [m/s2]")
    ax.set_xlabel("Time [s]")

    for i in range(len(plotsa)):
        plt.plot(np.arange(0.0, args.length*1000/(1000*SAMPLE_RATE), 1/SAMPLE_RATE), plotsa[i])
    plt.legend(['ax', 'ay', 'az'], loc='upper left')

    bx = fig.add_subplot(2, 1, 2)

    bx.set_title(" Gyroscope Readings in Spatial Domain ", fontsize=18)
    bx.set_ylabel("Gyro: [rad/s]")
    bx.set_xlabel("Time [s]")

    for i in range(len(plotsg)):
        plt.plot(np.arange(0.0, args.length * 1000 / (1000 * SAMPLE_RATE), 1 / SAMPLE_RATE), plotsg[i])
    plt.legend(['gx', 'gy', 'gz'], loc='upper left')

    plt.show()
    fig.savefig("../img/original_" + name + ".png")

def plotG(plot, name):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" Gyroscope Readings in Spatial Domain: " + name, fontsize=18)
    ax.set_ylabel("Gyro: [rad/s], Accel: [m/s]")
    ax.set_xlabel("Time [s]")

    plt.plot(np.arange(0.0, args.length * 1000 / (1000 * SAMPLE_RATE), 1 / SAMPLE_RATE), plot)

    plt.show()
    fig.savefig("../img/original_" + name + ".png")


def save(data, name, freq):
    now = datetime.datetime.now()
    file = open(name + "_{}.txt".format(now.isoformat()), 'w')
    file.write("{}\n".format(freq))
    for i in range(len(data[0])):
        file.write("{} {} {} {} {} {} {}\n".format((1000*i/SAMPLE_RATE), data[0][i], data[1][i], data[2][i], data[3][i],
                                                 data[4][i], data[5][i]))

if __name__ == "__main__":
    main()
