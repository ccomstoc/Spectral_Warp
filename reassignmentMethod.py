
# Specify the path to your audio file

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Specify the path to your audio file
file_path = "files/softSaw.wav"

# Load the audio file (using sr=None to keep its native sampling rate)
y, sr = librosa.load(file_path, sr=None)

# Set STFT parameters
n_fft = 4096*2
hop_length = n_fft // 4

# Compute a standard spectrogram (magnitude of the STFT)
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
mags = np.abs(D)
mags_db = librosa.amplitude_to_db(mags, ref=np.max)

# Compute the reassigned spectrogram
freqs, times, reassigned_mags = librosa.reassigned_spectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
)
reassigned_db = librosa.amplitude_to_db(reassigned_mags, ref=np.max)

# Create subplots that share the y-axis
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# Plot the standard spectrogram with a logarithmic frequency axis
img = librosa.display.specshow(mags_db, sr=sr, hop_length=hop_length,
                               x_axis='time', y_axis='log', ax=ax0)
ax0.set(title='Standard Spectrogram (Log Frequency)')

# Plot the reassigned spectrogram using a scatter plot on the same y-axis scale
ax1.scatter(times, freqs, c=reassigned_db, cmap='magma', alpha=1, s=.05)
ax1.set(title='Reassigned Spectrogram (Log Frequency)',
        xlabel='Time (s)', ylabel='Frequency (Hz)')

# Add a shared colorbar
fig.colorbar(img, ax=(ax0, ax1), format="%+2.f dB")

plt.show()