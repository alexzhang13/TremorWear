#!/usr/bin/env python

# python driver for mpu9250: https://github.com/MomsFriendlyRobotCompany/mpu9250
from libs.mpu9250.mpu9250.mpu9250 import mpu9250
from time import sleep

imu = mpu9250()

def main():
	try:
		while True:
			a = imu.accel
			print("Accel" + a[0] + a[1] + a[2])
			sleep(0.5)
	except KeyboardInterrupt:
		print("end")

if __name__ == "__main__":
	main()
