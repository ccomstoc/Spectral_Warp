import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

def main():
    # Load audio data
    audio_data, sample_rate = librosa.load('files/JUST_Organ.wav', sr=None)

    # Apply window function
    #Generates array of floating point numbers, tapering from small to large to small(The hanning window), then multiplies arrays, to window audio
    windowed_signal = audio_data * np.hanning(len(audio_data))

    # Perform FFT
    #Gathers overall freq content, stft is preformed later in order to graph
    fft_result = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/sample_rate)

    # Frequency shift amount (in Hz)
    shift_amount = 400  # Adjust this value to shift frequencies up or down

    # Calculate the shift in terms of FFT bins
    shift_bins = int(shift_amount * len(fft_result) / sample_rate)

    # Shift the frequencies by multiplying with a complex exponential
    fft_result_shifted = np.roll(fft_result, shift_bins)

    # Reconstruct signal using inverse FFT
    reconstructed_signal = np.fft.ifft(fft_result_shifted).real

    # Normalize reconstructed signal
    reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # Save the reconstructed audio
    sf.write('./files/shifted_audio.wav', reconstructed_signal, sample_rate)
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

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()