# Modified to not include Magnetometer

from __future__ import print_function
from __future__ import division

try:
    import smbus2 as smbus
except ImportError:
    print('WARNING: Using fake hardware')
    from .fakeHW import smbus
# from fake_rpi import smbus

from time import sleep
import struct
import ctypes  # for signed int

MPU9250_ADDRESS = 0x68
DEVICE_ID = 0x71
WHO_AM_I = 0x75
PWR_MGMT_1 = 0x6B
INT_PIN_CFG = 0x37
INT_ENABLE = 0x38
# --- Accel ------------------
ACCEL_DATA = 0x3B
ACCEL_CONFIG = 0x1C
ACCEL_CONFIG2 = 0x1D
ACCEL_2G = 0x00
ACCEL_4G = (0x01 << 3)
ACCEL_8G = (0x02 << 3)
ACCEL_16G = (0x03 << 3)
# --- Temp --------------------
TEMP_DATA = 0x41
# --- Gyro --------------------
GYRO_DATA = 0x43
GYRO_CONFIG = 0x1B
GYRO_250DPS = 0x00
GYRO_500DPS = (0x01 << 3)
GYRO_1000DPS = (0x02 << 3)
GYRO_2000DPS = (0x03 << 3)

# --- Sample Rate -------------
SMPLRT_DIV = 0x19
CONFIG = 0x1A

class mpu9250(object):
    def __init__(self, bus=1):
        """
		Setup the IMU

		reg 0x25: SAMPLE_RATE= Internal_Sample_Rate / (1 + SMPLRT_DIV)
		reg 0x29: [2:0] A_DLPFCFG Accelerometer low pass filter setting
			ACCEL_FCHOICE 1
			A_DLPF_CFG 4
			gives BW of 20 Hz
		reg 0x35: FIFO disabled default - not sure i want this ... just give me current reading

		might include an interface where you can change these with a dictionary:
			setup = {
				ACCEL_CONFIG: ACCEL_4G,
				GYRO_CONFIG: AK8963_14BIT | AK8963_100HZ
			}
		"""
        self.bus = smbus.SMBus(bus)

        # let's double check we have the correct device address
        ret = self.read8(MPU9250_ADDRESS, WHO_AM_I)
        if ret is not DEVICE_ID:
            raise Exception('MPU9250: init failed to find device')

        self.write(MPU9250_ADDRESS, PWR_MGMT_1, 0x00)  # turn sleep mode off
        sleep(0.2)
        self.bus.write_byte_data(MPU9250_ADDRESS, PWR_MGMT_1, 0x01)  # auto select clock source

        # set sample speed to 1khz
        self.write(MPU9250_ADDRESS, SMPLRT_DIV, 0)
        self.write(MPU9250_ADDRESS, CONFIG, 0x03)

        # set max range for gyro and accel
        self.write(MPU9250_ADDRESS, ACCEL_CONFIG, ACCEL_2G)
        self.write(MPU9250_ADDRESS, GYRO_CONFIG, GYRO_250DPS)

        # You have to enable the other chips to join the I2C network
        # then you should see 0x68 and 0x0c from:
        # sudo i2cdetect -y 1
        self.write(MPU9250_ADDRESS, INT_PIN_CFG, 0x22)
        self.write(MPU9250_ADDRESS, INT_ENABLE, 0x01)
        sleep(0.1)

        # normalization coefficients
        self.alsb = 2.0 / 32760  # ACCEL_2G
        self.glsb = 250.0 / 32760  # GYRO_250DPS

    # self.mlsb = 4800.0 / 32760 # MAGNET range +-4800


    def __del__(self):
        self.bus.close()

    def write(self, address, register, value):
        self.bus.write_byte_data(address, register, value)

    def read8(self, address, register):
        data = self.bus.read_byte_data(address, register)
        return data

    def read16(self, address, register):
        data = self.bus.read_i2c_block_data(address, register, 2)
        return self.conv(data[0], data[1])

    def read_xyz(self, address, register, lsb):
        """
		Reads x, y, and z axes at once and turns them into a tuple.
		"""
        # data is MSB, LSB, MSB, LSB ...
        data = self.bus.read_i2c_block_data(address, register, 6)

        # data = []
        # for i in range(6):
        # 	data.append(self.read8(address, register + i))

        x = self.conv(data[0], data[1]) * lsb
        y = self.conv(data[2], data[3]) * lsb
        z = self.conv(data[4], data[5]) * lsb

        # print('>> data', data)
        # ans = self.convv.unpack(*data)
        # ans = struct.unpack('<hhh', data)[0]
        # print('func', x, y, z)
        # print('struct', ans)

        return (x, y, z)

    def conv(self, msb, lsb):
        value = lsb | (msb << 8)

        return ctypes.c_short(value).value

    @property
    def accel(self):
        return self.read_xyz(MPU9250_ADDRESS, ACCEL_DATA, self.alsb)

    @property
    def gyro(self):
        return self.read_xyz(MPU9250_ADDRESS, GYRO_DATA, self.glsb)

    @property
    def temp(self):
        """
		Returns chip temperature in C

		pg 33 reg datasheet:
		pg 12 mpu datasheet:
		Temp_room 21
		Temp_Sensitivity 333.87
		Temp_degC = ((Temp_out - Temp_room)/Temp_Sensitivity) + 21 degC
		"""
        temp_out = self.read16(MPU9250_ADDRESS, TEMP_DATA)
        temp = ((temp_out - 21.0) / 333.87) + 21.0  # these are from the datasheets

        return temp


