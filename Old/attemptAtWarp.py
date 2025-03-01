import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import scipy.interpolate
import scipy.fft

def main():
    # Load audio data
    audio_data, sample_rate = librosa.load('../files/JUST_Organ.wav', sr=None)

    # Apply window function
    windowed_signal = audio_data * np.hanning(len(audio_data))

    # Perform FFT
    fft_result = scipy.fft.fft(windowed_signal)
    fft_len = len(fft_result)
    freqs = np.fft.fftfreq(fft_len, d=1/sample_rate)

    # Define anchors and stretch factors
    anchors = [0, int(fft_len * 0.25), int(fft_len * 0.5), int(fft_len * 0.75), fft_len // 2]
    stretch_factors = [1.0, .7, 0.5, 1.2, 1.0]  # Added a fifth stretch factor

    # Create interpolation function
    target_positions = np.array(anchors) * np.array(stretch_factors)
    warp_func = scipy.interpolate.interp1d(
        anchors, target_positions, kind='linear', fill_value="extrapolate"
    )

    # Initialize a new FFT result array
    warped_fft_result = np.zeros_like(fft_result, dtype=complex)

    # Apply warping to redistribute spectral energy
    for k in range(fft_len):
        new_k = int(warp_func(k))
        if 0 <= new_k < fft_len:
            warped_fft_result[new_k] += fft_result[k]  # Distribute energy

    shift_amount = 0  # Adjust this value to shift frequencies up or down

    # Calculate the shift in terms of FFT bins
    shift_bins = int(shift_amount * len(warped_fft_result) / sample_rate)

    # Shift the frequencies by multiplying with a complex exponential
    fft_result_shifted = np.roll(warped_fft_result, shift_bins)

    # Reconstruct signal using inverse FFT
    reconstructed_signal = scipy.fft.ifft(fft_result_shifted).real

    # Normalize reconstructed signal
    reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # Save the reconstructed audio
    sf.write('../files/warped_audio.wav', reconstructed_signal, sample_rate)
    print("Frequency-warped audio saved successfully.")

    # Higher resolution STFT parameters
    n_fft = 4096  # Higher FFT size for better frequency resolution
    hop_length = 256  # Smaller hop length for better time resolution

    # Plot spectrograms
    plt.figure(figsize=(12, 8))

    # Original signal spectrogram with logarithmic frequency axis
    plt.subplot(2, 1, 1)
    plt.title("Original Signal Spectrogram (High Resolution)")
    original_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(original_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')

    # Warped signal spectrogram with logarithmic frequency axis
    plt.subplot(2, 1, 2)
    plt.title("Frequency-Warped Signal Spectrogram (High Resolution)")
    warped_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed_signal, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(warped_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()