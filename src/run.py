#!/usr/bin/env python

# make libs folder visible
import sys
sys.path.append('../')

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250 import mpu9250
from libs import mcp3008

import time
import argparse
from collections import deque

from scanf import scanf
import numpy as np
import matplotlib.pyplot as plt

from src import preprocess, stabilizer

# CONSTANTS #
DEG_TO_RAD = np.pi/180
G_TO_MPERS = 9.80665
SAMPLE_RATE = 500
WINDOW_SIZE = 500


# Arg Parse
parser = argparse.ArgumentParser(description='Main Class for Running TremorWear')

args = parser.parse_args()

def main():
    processor = preprocess.SignalProcessor(sample_rate=SAMPLE_RATE)
    imu = mpu9250()
    raw_data = Motion_Data()

    try:
        while True:
            # Record IMU Data
            axt, ayt, azt, gxt, gyt, gzt = record(imu)
            raw_data.add_data(axt, ayt, azt, gxt, gyt, gzt)

            if raw_data.cnt > WINDOW_SIZE:
                # Filter and Feed into LSTM
                ft_seq = processor.Bandpass_All(raw_data.unpack(), 3, 12, 5)

    except KeyboardInterrupt:
        print("Program Terminated")

def record(imu):
    a = imu.accelgyro
    (ax, ay, az), (gx, gy, gz) = a
    return ax*G_TO_MPERS, ay*G_TO_MPERS, az*G_TO_MPERS, gx*DEG_TO_RAD, gy*DEG_TO_RAD, gz*DEG_TO_RAD

if __name__ == "__main__":
    main()


class Motion_Data():
    def __init__(self):
        self.ax = deque()
        self.ay = deque()
        self.az = deque()
        self.gx = deque()
        self.gy = deque()
        self.gz = deque()
        self.cnt = 0

    def add_data(self, ax, ay, az, gx, gy, gz):
        self.ax.append(ax)
        self.ay.append(ay)
        self.az.append(az)
        self.gx.append(gx)
        self.gy.append(gy)
        self.gz.append(gz)
        self.cnt += 1

        if self.cnt > WINDOW_SIZE:
            self.ax.popleft()
            self.ay.popleft()
            self.az.popleft()
            self.gx.popleft()
            self.gy.popleft()
            self.gz.popleft()
            self.cnt -= 1

    def unpack(self):
        return [list(self.ax), list(self.ay), list(self.az), list(self.gx), list(self.gy), list(self.gz)]
