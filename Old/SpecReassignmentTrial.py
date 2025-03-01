import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft, get_window


def enhanced_frequency_spectrogram(wav_file, db_threshold=-80):
    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Ensure data is mono if it's stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Define a high-resolution STFT configuration
    nperseg = 4096  # Increase the FFT size for better frequency resolution
    noverlap = nperseg // 2  # 50% overlap
    window = get_window('hann', nperseg)  # Use a Hanning window

    # Perform Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(data, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)

    # Calculate the magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(Zxx))

    # Apply the dB threshold
    magnitude_db = np.maximum(magnitude_db, db_threshold)

    # Calculate frequency reassignment
    phase = np.angle(Zxx)
    reassign_freq = np.gradient(np.unwrap(phase), axis=0) / (2 * np.pi)

    # Enhanced frequency by adding the reassigned frequency
    enhanced_magnitude = np.abs(Zxx) * np.exp(1j * (2 * np.pi * reassign_freq))
    enhanced_magnitude_db = 20 * np.log10(np.abs(enhanced_magnitude))

    # Apply the dB threshold to the enhanced spectrogram
    enhanced_magnitude_db = np.maximum(enhanced_magnitude_db, db_threshold)

    # Plot the original and enhanced spectrogram on a logarithmic frequency scale (in Hz)
    plt.figure(figsize=(12, 8))

    # Original Spectrogram
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, np.log10(f + 1), magnitude_db, shading='gouraud')
    plt.title('Original Spectrogram (Logarithmic Frequency)')
    plt.ylabel('Log Frequency [log(Hz)]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Magnitude [dB]')

    # Enhanced Spectrogram
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, np.log10(f + 1), enhanced_magnitude_db, shading='gouraud')
    plt.title('Enhanced Frequency Spectrogram (Logarithmic Frequency)')
    plt.ylabel('Log Frequency [log(Hz)]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Magnitude [dB]')

    plt.tight_layout()
    plt.show()


# Usage example
wav_file = '../files/JUST_Organ.wav'  # Replace with your WAV file path
enhanced_frequency_spectrogram(wav_file, db_threshold=20)  # Adjust the threshold as needed