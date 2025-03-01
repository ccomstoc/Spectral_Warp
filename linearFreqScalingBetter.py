import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from fontTools.merge.util import first




def resize_data_with_priority(data, idx_range, target_size):
    #This function turns 010203 into 130, so this could be investigated, the reason for this is because of how lower and
    #upper bound ar calculated,
    # fisrt the range of indexes 012345 are converted into 0,2.5,5 with the linspace funtion
    # then the loop identifies uses this to idenfy the ranges in original array the qualify for remapping
    #     the problem arises because first mapping to 0 ar (0,1), then 2,0,3, then the last is 0, so this could be optimized but
    #     honestly might not cause issues on large scale
    start_idx = idx_range[0]
    end_idx = idx_range[1]
    # Extract the subarray to resize
    subarray = data[start_idx:end_idx + 1]
    original_size = len(subarray)

    # Initialize the new resized array with zeros
    resized_data = np.zeros(target_size, dtype=subarray.dtype)

    # Map original indices to target indices
    original_indices = np.linspace(0, original_size - 1, num=original_size)
    target_indices = np.linspace(0, original_size - 1, num=target_size)

    for target_idx in range(target_size):
        # Find the range of original indices that map to this target index
        lower_bound = target_indices[target_idx - 1] if target_idx > 0 else 0
        upper_bound = target_indices[target_idx + 1] if target_idx < target_size - 1 else original_size - 1

        # Get the original indices within this range
        relevant_indices = np.where((original_indices >= lower_bound) & (original_indices <= upper_bound))[0]

        # Use max value when compressing, or zero-fill when expanding
        if len(relevant_indices) > 0:
            resized_data[target_idx] = max(subarray[i] for i in relevant_indices)

    return resized_data



def decompose(spectrum, quarter_half):
    midpoint = len(spectrum) // 2  # spectrum actually only does to the nyquist
    first_half = spectrum[:midpoint]
    second_half = spectrum[midpoint:]

    if quarter_half == -1:
        first_half_midpoint = len(first_half) // 2  # spectrum actually only does to the nyquist
    else:
        first_half_midpoint = int(len(first_half) // quarter_half)
    first_quarter = first_half[:first_half_midpoint]
    second_quarter = first_half[first_half_midpoint:]

    return [first_quarter, second_quarter,second_half]

def main():
    # Load audio data
    file_path = "files/JUST_Organ.wav"
    audio_data, sample_rate = librosa.load( file_path, sr=None)

    # STFT parameters
    SIZE = 2048 #2048,8192,4096,1024
    anchor_grip_division_factor = 16
    n_fft = SIZE  # Window size for FFT
    quarter_n = int(SIZE/4)
    freq_res = int(44100/SIZE)#Come back to this, can graph to see where true midpoint RANGE is
    hop_length = int(SIZE/4) # Hop size for moving the window, SHOULD be half SIZE
    anchor = 1  # Frequency shift anchor

    # Compute STFT
    stft_result = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_result)
    stft_phase = np.angle(stft_result)
    middle_anchor = int(400)  # we are trying to warp bins in index 0 - 510? 511#elem see below MAX n_fft/4

    # Process each time slice in the STFT
    print(stft_magnitude.shape[1])
    for t in range(stft_magnitude.shape[1]):


        """
        ***Currently works by like pretty much shifting frequency bins, but consider looking into interpolation in the frequency domain for frequency 
        shifting as it can make things smoother and still works in fft when not shifting with integers, as it interpretes in between windows???
        
        
        The first step here is to break everything down into the warp areas, or anchors
        
        A big oversight in the past to avoid is these arrays have bins not actual frquencies, also these bins only go up to ~22khz
        The size of your stft result array, and thus magnitude and phase arrays, is (n_fft/2)+1
        
        Another large consideration is how the warp function will handle this data type, should it also warp phase?? But is spectrum already a complex number
        
        So I want my anchors to be at 0, and 11khz(half nyquist), I will leave above that untouched
        
        
        """


        # so is an array of index 0 - 1024 and a max frequency of 22.05khz
        #Half of this would half index of 0 - size:1025/ 2 = 512 elements non inclusive  for first half making it 511 and 513 number of elements
        #So first half has 511 elements
        #511/2 255#elements first quarter 256 second quarter
        spectrum = stft_magnitude[:, t]
        phase = stft_phase[:,t]

        midpoint = len(spectrum) // 2 #512 bc 1025/2



        #Helps for decomposing both phase and mag into first 2 quarters and last half in indexs 0,1,2 respectivly
        #Set to - 1 or 2 to set middle anchor GRIP half way between upper and lower bound
        #otherwise set to the fraction between you want to set grip at, use decimal for above natural middle and whole for below
        decompMag = decompose(spectrum,anchor_grip_division_factor)
        decompP = decompose(phase,anchor_grip_division_factor)
        #Still usefull

        # 0 - 10
        #11 things
        # last arg is size of arrays returning, we want a total of 511 elements so - 511 not 510 which is actual range of anchor values, or is it?
        decompMag[0] = resize_data_with_priority(decompMag[0],[0,(len(decompMag[0])-1)],middle_anchor)
        decompMag[1] = resize_data_with_priority(decompMag[1], [0, (len(decompMag[1])-1)], (quarter_n- middle_anchor))

       # print(quarter_n)
        decompP[0] = resize_data_with_priority(decompP[0], [0, (len(decompP[0]) - 1)], middle_anchor)
        decompP[1] = resize_data_with_priority(decompP[1], [0, (len(decompP[1]) - 1)], (quarter_n - middle_anchor))
        #Preform warping here, will need to do this all again for phase lame






        # Reassemble the spectrum and phase
        stft_magnitude[:, t] = np.append(np.append(decompMag[0],decompMag[1]),decompMag[2])
        stft_phase[:,t] = np.append((np.append(decompP[0],decompP[1])),decompP[2])


    # Reconstruct the complex STFT with modified magnitudes
    modified_stft = stft_magnitude * np.exp(1j * stft_phase)

    # Perform inverse STFT
    reconstructed_signal = librosa.istft(modified_stft, hop_length=hop_length)

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
    # Warp Anchor
    plt.axhline(y=(middle_anchor * freq_res), color='blue', linestyle='--', linewidth=2,
                label=f'anchor: {"warpAnchor"} Hz')
    # Upper bound
    plt.axhline(y=(11025), color='red', linestyle='--', linewidth=2, label=f'anchor: {11025 / 2} Hz')
    # Warp Grip
    plt.axhline(y=((quarter_n / anchor_grip_division_factor) * freq_res), color='green', linestyle='--', linewidth=2,
                label=f'anchor: {middle_anchor * freq_res} Hz')

    # Shifted signal spectrogram with logarithmic frequency axis
    plt.subplot(2, 1, 2)
    plt.title("Frequency-Shifted Signal Spectrogram (High Resolution)")
    shifted_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed_signal, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(shifted_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    #Warp Anchor
    plt.axhline(y=(middle_anchor*freq_res), color='blue', linestyle='--', linewidth=2, label=f'anchor: {"warpAnchor"} Hz')
    #Upper bound
    plt.axhline(y=(11025), color='red', linestyle='--', linewidth=2, label=f'anchor: {11025/2} Hz')
    #Warp Grip
    plt.axhline(y=((quarter_n/anchor_grip_division_factor)*freq_res), color='green', linestyle='--', linewidth=2, label=f'anchor: {middle_anchor*freq_res} Hz')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()