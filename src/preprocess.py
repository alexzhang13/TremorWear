import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Process 3 sequences of spatial information (gx, gy, gz)
class SignalProcessor():
    def __init__(self, sample_rate):
        self.srate = sample_rate

    # Simple Truncating Filter
    def SimpleFilter(self, sequence, low_freq, high_freq):
        # Convert to FFT
        fourier, freq = self.Fourier(sequence)

        # Filter
        for i in range(len(freq)):
            if freq[i] < low_freq or freq[i] > high_freq:
                fourier[i] = 0

        self.SaveFFTGraph(fourier, freq, "Filtered")

        # Return iFFT of filtered FFT
        return self.IFourier(fourier, len(sequence))

    # Simple Bandpass Filter
    def Bandpass_Filter(self, sequence, low_freq, high_freq, order):
        nyq = 0.5*self.srate # Nyquist Frequency
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_sequence = filtfilt(b, a, sequence)
        time_window = np.linspace(0, len(sequence) * (1 / self.srate), len(sequence))
        return filtered_sequence, time_window

    def Fourier(self, spatial_sequence):
        fourier = np.fft.fft(spatial_sequence)
        freq = np.linspace(0.0, self.srate, len(spatial_sequence)/2)
        return fourier, freq

    def IFourier(self, fourier_sequence, window_size):
        ifourier = np.fft.ifft(fourier_sequence)
        time_window = np.linspace(0, window_size*(1/self.srate), window_size)

        return ifourier, time_window

    def Bandpass_All(self, sequences, low_freq, high_freq, order):
        filtered_sequences = []
        for i in range(len(sequences)):
            filtered_sequences.append((self.Bandpass_Filter(sequences[i], low_freq, high_freq, order)))

        return filtered_sequences

# -------------------------------- TESTER FUNCTIONS -------------------------------- #

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

        fig.savefig("../img/" + name + ".png")

    def SaveFFTGraph(self, fourier, freq, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Decomposed " + name, fontsize=18)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency [Hz]")

        ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        plt.xlim(0, 50)

        plt.show()

        fig.savefig("../img/" + name + ".png")

    def SaveButterFilterGraph(self, y, window, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Filtered Graph: " + name, fontsize=18)
        ax.set_ylabel("Angular V(t) [rad/s]")
        ax.set_xlabel("Time [s]")

        # ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        ax.plot(window, y)
        plt.show()

        fig.savefig("../img/" + name + ".png")

    def FilterTest(self, sequence, name):
        filtered, window = self.Bandpass_Filter(sequence, 3, 12, 5)
        self.SaveButterFilterGraph(filtered, window, name + "_Filtered")

        fourier, freq = self.Fourier(sequence)
        self.SaveFFTGraph(fourier, freq, name + ": Original FFT")

        fourier, freq = self.Fourier(filtered)
        self.SaveFFTGraph(fourier, freq, name + ": Filtered FFT")

    def FourierTest(self, sequence, name):
        fourier, freq = self.Fourier(sequence)
        self.SaveFFTGraph(fourier, freq, name + "_FFT")

        ifourier, window = self.IFourier(fourier, len(sequence))
        self.SaveIFFTGraph(ifourier, window, name + "_C")

    def WindowFourier(self, sequence, window, name):
        pass