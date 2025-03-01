import numpy as np
import time
from matplotlib.colors import Normalize
import librosa
import librosa.display
import matplotlib
matplotlib.use("Qt5Agg")#Slightly faster than default
#plt = matplotlib.pyplot
import matplotlib.pyplot as plt

# Specify the path to your audio file
#file_path = "files/JUST_Organ.wav"
file_path = "files/NoOTTSaw.wav"

    #"files/NoOTTSaw.wav"

# Load the audio file (using sr=None to keep its native sampling rate)
y, sr = librosa. load(file_path, sr=None)

# Set STFT parameters
n_fft = int(4096*2) #The bigger, the less time res, more frequency res
hop_length = n_fft // 4 #smaller hop means more overlap and closer points

# Generate the Kaiser window
beta = 14  # Example beta value
window = librosa.filters.get_window(('kaiser', beta), n_fft)
print("Window Generated")
# Compute a standard spectrogram (magnitude of the STFT) using the Kaiser window
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
mags = np.abs(D)
mags_db = librosa.amplitude_to_db(mags, ref=np.max)
print("Computed a standard spectrogram")
# Compute the reassigned spectrogram using the Kaiser window


freqs, times, reassigned_mags = librosa.reassigned_spectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window
)
reassigned_db = librosa.amplitude_to_db(reassigned_mags, ref=np.max)

print("Computed the reassigned spectrogram")

# Create subplots that share the y-axis
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# Plot the standard spectrogram with a logarithmic frequency axis
img = librosa.display.specshow(mags_db, sr=sr, hop_length=hop_length,x_axis='time', y_axis='log', ax=ax0)
ax0.set(title='Standard Spectrogram (Log Frequency)')

print("Computed Graph1")

ax1.set_facecolor('black')


#Threshhold reduces number of points that need to be drawn, improving performance
threshold = -80
# Only plot points above -60 dB
mask = reassigned_db > threshold
# Plot the reassigned spectrogram using a scatter plot on the same y-axis scale
#ADJUST alpha and s for point visibility characteristics
ax1.scatter(times[mask], freqs[mask], c=reassigned_db[mask], cmap='magma', alpha=.1, s=1)
print("Computed Graph2")
ax1.set(title='Reassigned Spectrogram (Log Frequency)',
        xlabel='Time (s)', ylabel='Frequency (Hz)')

# Add a shared colorbar
fig.colorbar(img, ax=(ax0, ax1), format="%+2.f dB")
print("ColorBar")

start_time = time.time()
plt.draw()  # Renders the figure without blocking
plt.pause(0.1)  # Pause briefly to allow rendering
end_time = time.time()
print(f"plt.show() runtime (non-blocking): {end_time - start_time:.4f} seconds")

plt.show()
print("Show")