import numpy as np
import matplotlib.pyplot as plt

# Process 3 sequences of spatial information (gx, gy, gz)
class SignalProcessor():
    def __init__(self, sample_rate):
        self.srate = sample_rate

    # Run Kalman Filter on IMU Data
    def KFilter(self):
        pass

    def Fourier(self, spatial_sequence):
        fourier = np.fft.fft(spatial_sequence)
        freq = np.linspace(0.0, self.srate, len(spatial_sequence)/2)
        return fourier, freq

    def IFourier(self, fourier_sequence, window_size):
        ifourier = np.fft.ifft(fourier_sequence)
        time_window = np.linspace(0, window_size*(1/self.srate), window_size)

        return ifourier, time_window

    def SaveIFFTGraph(self, ifourier, window, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Composed Graph: " + name, fontsize=18)
        ax.set_ylabel("Angular V(t) [rad/s]")
        ax.set_xlabel("Time [s]")

        # ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        ax.plot(window, ifourier.real)
        ax.plot(window, ifourier.imag)

        plt.legend(['real', 'imaginary'], loc='upper left')
        plt.show()

        fig.savefig("../imgs/" + name + ".png")

    def SaveFFTGraph(self, fourier, freq, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Decomposed " + name, fontsize=18)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency [Hz]")

        ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        plt.xlim(0, 250)

        plt.show()

        fig.savefig("../imgs/" + name + ".png")

    def FilterTremor(self, fourier, freq):
        pass

    def FourierTest(self, sequence, name):
        fourier, freq = self.Fourier(sequence)
        self.SaveFFTGraph(fourier, freq, name + "_FFT")
        ifourier, window = self.IFourier(fourier, len(sequence))
        self.SaveIFFTGraph(ifourier, window, name + "_C")