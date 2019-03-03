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
parser.add_argument("--plabel", type=str, default="Patient1", help="Patient Label")
parser.add_argument("--sqn_path", type=str, default="PatientName_Action", help="Path to Store Filtered Sequences")
parser.add_argument("--path", type=str, default="PatientName_Action", help="Name/Folder of Recording for Playback")
parser.add_argument("--length", type=int, default=10000, help="Length of Tremor Recording")
parser.add_argument("--window_num", type=int, default=0, help="Number denoting type of Sensor Data to Window"
                "(Only Applies When --window is Called: 0 --> ax, 1 --> ay, 2 --> az, 3 --> gx, 4 --> gy, >=5 --> gz")
parser.add_argument("--window_size", type=int, default=250, help="Shifting Window Size")
parser.add_argument("--record", dest="record_data", action="store_true", help="Record Patient Data")
parser.add_argument("--read", dest="record_data", action="store_false", help="Playback Patient Data from File")
parser.set_defaults(record_data=True)
parser.add_argument("--save_sqn", dest="save_sqn", action="store_true", help="Store Filtered and Voluntary Sequences")
parser.set_defaults(save_sqn=False)
parser.add_argument("--plot", dest="plot_data", action="store_true", help="Plot Data")
parser.set_defaults(plot_data=False)
parser.add_argument("--plotall", dest="plot_all", action="store_true", help="Plot All Data in a Folder")
parser.set_defaults(plot_all=False)
parser.add_argument("--fft", dest="plot_fft", action="store_true", help="Plot All (FFT Mode)")
parser.set_defaults(plot_fft=False)
parser.add_argument("--window", dest="window", action="store_true", help="Print Windowed FFT of Sequence")
parser.set_defaults(window=False)
parser.add_argument("--index", dest="index", action="store_true", help="Get Tremor/Voluntary Comfort Index")
parser.set_defaults(index=False)
parser.add_argument("--is_tremor", dest="is_tremor", action="store_true", help="Is the Input Data Tremor or Voluntary")
parser.set_defaults(is_tremor=False)
parser.add_argument("--graph", dest="graph", action="store_true", help="Graph IMU Data")
parser.set_defaults(graph=False)

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
        if args.graph is True:
            graph_imu(args.plot_all, args.plot_fft, args.path, args.save_sqn, args.sqn_path, args.plot_data, args.window,
                  args.window_num, args.window_size, processor)
        if args.index is True:
            print("Flag: Reading Data from File: {}".format(args.path))
            freq, _, _, _, gx, gy, gz = read(args.path)
            plotsg = [gx, gy, gz]
            print_idx_metric(plotsg, processor, args.is_tremor)

# Function for Graphing IMU Data - Option to Plot One File or All Files from One Folder
def graph_imu(plot_all, fft, filepath, save_sqn, sqn_path, plot_data, window, window_num, window_size, processor):
    if plot_all:
        print("Flag: Reading All Data from Folder: {}".format(filepath))
        path = "../data/" + filepath
        for filename in os.listdir(path):
            freq, ax, ay, az, gx, gy, gz = read_frompath(path + "/" + filename)
            plotsa = [ax, ay, az]
            plotsg = [gx, gy, gz]

            if fft:
                if window_num > 2:
                    fourier, freq = processor.Fourier(plotsg[min(5, window_num) % 3])
                else:
                    fourier, freq = processor.Fourier(plotsa[window_num])
                processor.SaveFFTGraph(fourier, freq, sdata_dict[window_num]+" - "+erase_ts(filename))
            else:
                plot(plotsa, plotsg, filename)
    else:
        print("Flag: Reading Data from File: {}".format(filepath))
        freq, ax, ay, az, gx, gy, gz = read(filepath)
        plotsa = [ax, ay, az]
        plotsg = [gx, gy, gz]
        if plot_data is True:
            gx_filt, _ = processor.Bandpass_Filter(gx, 3, 13, 5)
            plotFilt(gx, gx_filt, "Gx")
            # plot(plotsa, plotsg, "Filler Name")
            # plotG(gx, "Gyro X")
            # processor.FilterTest(gx, "Gyro X")
            # plotG(gy, "Gyro Y")
            # processor.FilterTest(gy, "Gyro Y")
            # plotG(gz, "Gyro Z")
            # processor.FilterTest(gz, "Gyro Z")
        if save_sqn is True:
            processor.SaveSequences(gx, gy, gz, sqn_path)
        if window is True:
            if window_num > 2:
                data = plotsg[min(5, window_num) % 3]
            else:
                data = plotsa[window_num]
            processor.WindowFourier(data, window_size, sdata_dict[window_num])

# Calculate Accuracy (Tremor/Voluntary Comfort Index) of Sequence
def print_idx_metric(plots, processor, is_tremor):
    gx_f, _ = processor.Bandpass_Filter(plots[0], 3, 13, 5)
    gy_f, _ = processor.Bandpass_Filter(plots[1], 3, 13, 5)
    gz_f, _ = processor.Bandpass_Filter(plots[2], 3, 13, 5)

    if is_tremor is True:
        print("Tremor Reduction Loss Measure, Gx: {}%".format(idx_metric(plots[0], gx_f)))
        print("Tremor Reduction Loss Measure, Gy: {}%".format(idx_metric(plots[1], gy_f)))
        print("Tremor Reduction Loss Measure, Gz: {}%".format(idx_metric(plots[2], gz_f)))
    else:
        print("Voluntary Motion Reduction Measure, Gx: {}%".format(idx_metric(plots[0], gx_f)))
        print("Voluntary Motion Reduction Measure, Gy: {}%".format(idx_metric(plots[1], gy_f)))
        print("Voluntary Motion Reduction Measure, Gz: {}%".format(idx_metric(plots[2], gz_f)))

# Tremor/Voluntary Motion Index Measure
def idx_metric(sqn, filt_sqn):
    orig_sum = 0.0
    filt_sum = 0.0 # Sum of all values difference (original - filtered)

    for i in range(len(sqn)):
        orig_sum += np.abs(sqn[i])
        if sqn[i] != 0:
            filt_sum += filt_sqn[i]/sqn[i]

    return 100*filt_sum/len(sqn)


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
    fig.savefig("../img/" + name + "_Original.png")

def plotFilt(seq, filt_seq, name):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" Gyroscope Readings in Spatial Domain: " + name, fontsize=18)
    ax.set_ylabel("Gyro: [rad/s], Accel: [m/s]")
    ax.set_xlabel("Time [s]")

    plt.plot(np.arange(0.0, args.length * 1000 / (1000 * SAMPLE_RATE), 1 / SAMPLE_RATE), seq)
    plt.plot(np.arange(0.0, args.length * 1000 / (1000 * SAMPLE_RATE), 1 / SAMPLE_RATE), filt_seq)
    plt.legend(['orig', 'filt'], loc='upper left')

    plt.show()
    fig.savefig("../img/" + name + "_OrigFilt.png")

def save(data, name, freq):
    now = datetime.datetime.now()
    file = open(name + "_{}.txt".format(now.isoformat()), 'w')
    file.write("{}\n".format(freq))
    for i in range(len(data[0])):
        file.write("{} {} {} {} {} {} {}\n".format((1000*i/SAMPLE_RATE), data[0][i], data[1][i], data[2][i], data[3][i],
                                                 data[4][i], data[5][i]))

# Erase Timestamp from Previously Saved File Name (Remove Indices past second underscore _)
def erase_ts(str):
    idx = str.find("_2019")
    return str[0:idx]

if __name__ == "__main__":
    main()
