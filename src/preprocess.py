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
        freq = np.linspace(0.0, 1/(2*self.srate), len(spatial_sequence)/2)
        return fourier, freq

    def SaveFFTGraph(self, fourier, freq, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title("Decomposed " + name, fontsize=18)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency [Hz]")

        ax.plot(freq, 2.0/50 * np.abs(fourier[:50//2]))
        plt.show()

        fig.savefig("../imgs/" + name + ".png")

    def FilterTremor(self, fourier, freq):
        pass

    def FourierSequences(self, sequences):
        for i in range(len(sequences)):
            fourier, freq = self.Fourier(sequences[i])
            self.SaveFFTGraph(fourier, freq, str(i))