import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.fft import rfft, irfft

def scale_indices(data, original_range, target_range):
    original_indices = np.arange(original_range[0], original_range[1] + 1)
    target_indices = np.interp(original_indices,
                               [original_range[0], original_range[1]],
                               [target_range[0], target_range[1]])
    new_data = np.zeros(target_range[1] - target_range[0] + 1, dtype=complex)

    for orig_idx, target_idx in zip(original_indices, target_indices):
        new_idx = int(round(target_idx)) - target_range[0]
        if abs(data[orig_idx - original_range[0]]) > abs(new_data[new_idx]):
            new_data[new_idx] = data[orig_idx - original_range[0]]

    return new_data

def scale_indices_real(data, original_range, target_range):
    original_indices = np.arange(original_range[0], original_range[1] + 1)
    target_indices = np.interp(original_indices,
                               [original_range[0], original_range[1]],
                               [target_range[0], target_range[1]])
    new_data = np.zeros(target_range[1] - target_range[0] + 1)

    for orig_idx, target_idx in zip(original_indices, target_indices):
        new_data[int(round(target_idx)) - target_range[0]] = data[orig_idx - original_range[0]]

    return new_data

def main():
    # Load audio data
    audio_data, sample_rate = librosa.load('files/JUST_Organ.wav', sr=None)

    # Apply window function
    # Generates an array of floating-point numbers, tapering from small to large to small (Hanning window),
    # then multiplies arrays to window the audio
    windowed_signal = audio_data * np.hanning(len(audio_data))

    # Perform rFFT
    # Gathers overall frequency content, stft is performed later in order to graph
    fft_result = rfft(windowed_signal)

    midpoint = len(fft_result) // 2

    # Split the array into two parts
    first_half_fft = fft_result[:midpoint]
    second_half_fft = fft_result[midpoint:]

    first_half_midpoint = len(first_half_fft) // 2 #This is wrong, causes it to be called at actual length
    anchor =  500




    first_original_range = [0,  first_half_midpoint]
    second_original_range = [ first_half_midpoint+1, len(first_half_fft) - 1]


    first_target_range = [0, anchor]
    second_target_range = [anchor + 1, len(first_half_fft) - 1]

    first_half = scale_indices(first_half_fft, first_original_range, first_target_range)
    second_half = scale_indices(second_half_fft, second_original_range, second_target_range)

    final_warped_first_half = np.append(first_half, second_half)
    final_warped = np.append(final_warped_first_half,second_half_fft)

    # Reconstruct signal using inverse rFFT
    reconstructed_signal = irfft(final_warped)

    # Normalize reconstructed signal
    reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # Save the reconstructed audio
    sf.write('./files/linShifted_audio.wav', reconstructed_signal, sample_rate)
    print("Frequency-shifted audio saved successfully.")

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

    # Shifted signal spectrogram with logarithmic frequency axis
    plt.subplot(2, 1, 2)
    plt.title("Frequency-Shifted Signal Spectrogram (High Resolution)")
    shifted_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed_signal, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(shifted_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    plt.axhline(y=anchor, color='blue', linestyle='--', linewidth=2, label=f'anchor: {anchor} Hz')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()