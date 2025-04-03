
import time
import numpy as np
import matplotlib
from librosa import samples_like
from numpy.random import normal

matplotlib.use("Qt5Agg")#Slightly faster than default
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import math
from fontTools.merge.util import first




def loadBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='>'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length +'-' * (length - filled_length)
    print(f'\r{prefix}|{bar}|{percent}%{suffix}',end ='\r')
    if iteration == total:
        print()

def resize_data_with_priority(data, idx_range, target_size):
    #This function turns 010203 into 130, so this could be investigated, the reason for this is because of how lower and
    #upper bound ar calculated,
    # fisrt the range of indexes 012345 are converted into 0,2.5,5 with the linspace funtion
    # then the loop identifies uses this to idenfy the ranges in original array the qualify for remapping
    #     the problem arises because first mapping to 0 ar (0,1), then 2,0,3, then the last is 0, so this could be optimized but
    #     honestly might not cause issues on large scale
    start_idx = idx_range[0]
    end_idx = idx_range[1]
    # Extract the og_data to resize
    og_data = data[start_idx:end_idx + 1]
    original_size = len(og_data)
    #return og_data

    # Initialize the new resized array with zeros
    resized_data = np.zeros(target_size, dtype=og_data.dtype)

    #Expansion
    if original_size < target_size:
        ratio = target_size/original_size
        for old_index in range(original_size):
            new_index = round(old_index*ratio)-1
            resized_data[new_index] = og_data[old_index]
        #print(f"og_data{og_data[:50]}")
        #print()
        #print(f"new {resized_data[:50]}")
        return resized_data
    else: #Contraction
        ratio = original_size/target_size
        for i in range(target_size):
            lower_bound = i*ratio
            upper_bound = (i+1)*ratio
            lower_frac, lower_int = math.modf(lower_bound)
            upper_frac, upper_int = math.modf(upper_bound)
            lower_int = int(lower_int)
            upper_int = int(upper_int)
            max_value = 0
            for og_idx in range(lower_int, upper_int+1):
                testing_value = 0
                if og_idx == lower_int:
                    testing_value = (1-lower_frac)*og_data[og_idx]
                elif og_idx == upper_int:
                    if upper_frac != 0:#edge case with last interation,
                        testing_value = upper_frac*og_data[og_idx]
                else:
                    testing_value = og_data[og_idx]
                max_value = max(testing_value, max_value, key=abs)
            resized_data[i] = max_value


        return resized_data
def decompose(spectrum, quarter_half=2):
    midpoint = len(spectrum) // 2  # spectrum actually only does to the nyquist
    first_half = spectrum[:midpoint] #with 2049, 1024
    second_half = spectrum[midpoint:]#with 2049, 1025

    first_half_midpoint = int(len(first_half) // quarter_half)
    first_quarter = first_half[:first_half_midpoint]
    second_quarter = first_half[first_half_midpoint:]

    return [first_quarter, second_quarter,second_half]

def main():
    # Load audio data

    #file_path = "files/linShifted_audio.wav"
    file_path = "files/JUST_Organ.wav"
    #file_path = "files/crowd.wav"
    #file_path = "files/NoOTTSaw.wav"

    #file_path = "./files/FREE_MALWARE_FINAL.wav"
    #file_path = "./files/FIRSME_FINAL.wav"
    file_out = './files/linShifted_audio_NEWWARP.wav'
    normalize = False;
    audio_data, sample_rate = librosa.load(file_path, sr=None)



    #--------------------------CONFIG--------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    #Best Params
    #4096 and hoplength/8 has been low noise but lower freq res than preferred

    # STFT parameters
    SIZE = (8192)#2048,8192,4096,1024 #The bigger, the less time res, more frequency res

    n_fft = SIZE  # Window size for FFT
    quarter_n = int(SIZE/4)
    freq_res = int(44100/SIZE)#Come back to this, can graph to see where true midpoint RANGE is
    hop_length = n_fft // 16 # Hop size for moving the window, Default 4, SIZE #smaller hop means more overlap and closer points, but less freq res
    anchor = 1  # Frequency shift anchor

    beta = 15  # Example beta value, 14 is a colid value
    kaiser_window = librosa.filters.get_window(('kaiser', beta), n_fft)

    # so nyquist / target freq is the ratio to be applied to size
    # If you make this number larger than the top anchor, or nyquist/2, aka ~10k, it will break
    """
    # 5512.5 is no warp, 
    """
    # warp_frequency = 200
    #
    # anchor_grip_division_factor = 64  # Controls where top anchor is, 2 puts it at ~10k, aka half of nyquist
    # middle_anchor = int((warp_frequency / (sample_rate / 2)) * (SIZE/2)) #SIZE/2 because stft returns half of size
    # print(str(middle_anchor) + " MID ANCH ")
    # middle_anchor = int(400)  # we are trying to warp bins in index 0 - 510? 511#elem see below MAX n_fft/4

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------




    # Compute STFT
    #returns with size of (size/2)+1
    stft_result = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length,window=kaiser_window)
    stft_magnitude = np.abs(stft_result)
    stft_phase = np.angle(stft_result)

    # Process each time slice in the STFT
    print(stft_magnitude.shape[1])
    console_time_count = 0
    limitPrint = 0
    warp_frequency = 4000
    anchor_grip_division_factor = 2  # Controls where top anchor is, 2 puts it at ~10k, aka half of nyquist
    #warp_frequency += 1;
    flip = True;
    for t in range(stft_magnitude.shape[1]):
        warp_speed = 15
        if flip:
            warp_frequency+=warp_speed;
        else:
            warp_frequency -=warp_speed
        if warp_frequency < 4000 or warp_frequency >8000:
            flip =  not flip
        #print(warp_frequency)
        middle_anchor = int((warp_frequency / (sample_rate / 2)) * (SIZE / 2))  # SIZE/2 because stft returns half of size
        #print(str(middle_anchor) + " MID ANCH ")

        if t %100 == 0:
            #print(f"\r" + str(console_time_count) + "/"+str(stft_magnitude.shape[1]/100), end ='\r')
            print(str(console_time_count) + "/"+str(stft_magnitude.shape[1]/100))
            console_time_count+=1


        # so is an array of index 0 - 1024 and a max frequency of 22.05khz
        #Half of this would half index of 0 - size:1025/ 2 = 512 elements non inclusive  for first half making it 511 and 513 number of elements
        #So first half has 511 elements
        #511/2 255#elements first quarter 256 second quarter

        #[:, t] - returns entire stft slice at time t, returns entire row a col t
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

        """
        4096 TOTAL
        stft returns 2048+1
        
        2049
            decomp
        1024
        1024
        
        decomp mag length is 512, I feel like this should be 1024, because we are warping the bottom 2 quarters, and quarters are 1024, when size = 4096
        
        """


        if limitPrint == 0:
            print("len spec: ")
            print(len(spectrum))
            print('decomp left/right size')
            print(len(decompMag[0]) )
            print(len(decompMag[1]))
        ogDecompMag = decompMag.copy()

        decompMag[0] = resize_data_with_priority(decompMag[0],[0,(len(decompMag[0])-1)],middle_anchor)
        decompMag[1] = resize_data_with_priority(decompMag[1], [0, (len(decompMag[1])-1)], (quarter_n- middle_anchor))

        if limitPrint == 0:
            print("data ")
            print(len(decompMag[0]))
            print("idx_range" + str([0,(len(decompMag[0])-1)]) + "target_size: " + str(middle_anchor))

        # for i in range(len(decompMag)):
        #     if decompMag[0][i] != ogDecompMag[0][i]:
        #         print(str(decompMag[0][i]) + " " + str(ogDecompMag[0][i]) + "DIFFFFFERENT!!!!!!!!!!!!!")

        limitPrint +=1


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
    if normalize:
        reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

    # Save the reconstructed audio

    sf.write(file_out, reconstructed_signal, sample_rate)
    print("Frequency-shifted audio saved successfully.")


    #-----GRAPH CONFIG-----
    # --------------------
    # --------------------


    threshold = -80# Threshhold(db) reduces number of points that need to be drawn, improving performance
    alpha = .4 #how solid are plotted points
    size = .5 #how large are plotted points
    dpi = 100 #points per inch

    top_anchor = sample_rate /4 #/2 because stft only goes to nyquist, /2 again because we are actually warping bottom 2 quarters
    bottom_anchor = 32 #it is not actually 32, it is zero, but graph only goes to 32
    non_warp_middle_anchor = (sample_rate/4)/anchor_grip_division_factor # /4 to get top anchor, /2 again to get half way between 0 and top, or division factor, to set even lower
    warped_middle_anchor = warp_frequency


    # --------------------
    # --------------------
    # --------------------

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(14, 8),dpi=dpi)


    # Define tick positions (2^6 to 2^14)
    custom_ticks = [2 ** i for i in range(5, 15)]  # [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # Set these tick positions on the y-axis of the scatter plot (ax1)
    ax0.set_yscale('log')
    ax0.set_ylim(32, 20000)  # Make sure the limits cover your desired range
    ax0.set_yticks(custom_ticks)
    # Format tick labels as "2^6", "2^7", etc. using LaTeX formatting for clarity
    ax0.set_yticklabels([str(int(freq)) for freq in custom_ticks])


    og_freqs, og_times, og_reassigned_mags = librosa.reassigned_spectrogram( # still limited in resolution, instead of the entire bin being colored in when being plotted, it offers more precise pinpointing within that bin
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window=kaiser_window

    )
    thresh_reassigned_db = librosa.amplitude_to_db(og_reassigned_mags, ref=np.max)
    mask = thresh_reassigned_db > threshold

    ax0.set_facecolor('black')
    ax0.scatter(og_times[mask], og_freqs[mask], c=thresh_reassigned_db[mask], cmap='magma', alpha=alpha, s=size)
    print("Computed Graph2")
    ax0.set(title='Original Spectrogram ', xlabel='Time (s)', ylabel='Frequency (Hz)')

    #----------Draw Marking Lines----------
    ax0.axhline(y=top_anchor, color='yellow', linestyle='--', linewidth=2, label='Top Anchor')
    ax0.axhline(y=bottom_anchor, color='yellow', linestyle='--', linewidth=2, label='Bottom Anchor')
    ax0.axhline(y=non_warp_middle_anchor, color='green', linestyle='--', linewidth=2, label='Non-Warp Middle Anchor')
    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # -------------------------------------

    mod_freqs, mod_times, mod_reassigned_mags = librosa.reassigned_spectrogram(
        y=reconstructed_signal,
        S=modified_stft,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window=kaiser_window

    )
    thresh_reassigned_db = librosa.amplitude_to_db(mod_reassigned_mags, ref=np.max)
    mask = thresh_reassigned_db > threshold

    # Plot the reassigned spectrogram using a scatter plot on the same y-axis scale
    # ADJUST alpha and s for point visibility characteristics
    ax1.set_facecolor('black')
    ax1.scatter(mod_times[mask], mod_freqs[mask], c=thresh_reassigned_db[mask], cmap='magma', alpha=alpha, s=size)
    print("Computed Graph2")
    ax1.set(title='Mod Spectrogram ', xlabel='Time (s)', ylabel='Frequency (Hz)')

    # ----------Draw Marking Lines----------
    ax1.axhline(y=top_anchor, color='yellow', linestyle='--', linewidth=2, label='Top Anchor')
    ax1.axhline(y=bottom_anchor, color='yellow', linestyle='--', linewidth=2, label='Bottom Anchor')
    ax1.axhline(y=warped_middle_anchor, color='red', linestyle='--', linewidth=2, label='Warp Middle Anchor')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # -------------------------------------


    print("Drawing, please wait...")
    start_time = time.time()
    plt.draw()  # Renders the figure without blocking
    plt.pause(0.1)  # Pause briefly to allow rendering
    end_time = time.time()
    print(f"plt.show() runtime: {end_time - start_time:.4f} seconds")
    plt.show()
    print("Drawn")

if __name__ == "__main__":
    main()

    """
        Next Steps:
            dive into warping technique and improve, should not be changing data if no anchors are changed
            try to understand how I could better use interpolation to warp more accurately, as well as re-think using max value,
    
    
    """


    """
    Regarding reassignment method: 

    So what if I have 2 sin waves generating a tone so similar that they would be represented by the same bin, would the reassignment method, 
    and the result of the function represent that, or would it take the average of those frequencies and display them as one point in that bin. 
    Because if the size of reassigned freqs mirrors that of the number of bins, wouldn't it be limited to one frequency per bin? 

    Yes? it seems so... so the reassignment method does not offer necessarily higher resolution, but presents more accurate points within the current resolution


        '
        More advanced methods try to go beyond that limitation. For example:

            Synchrosqueezing Transform (SST): This technique refines the time–frequency representation further by reassigning energy more adaptively. It can sometimes separate components that overlap in the basic STFT grid by concentrating the energy around their true instantaneous frequencies.

            Parametric or Subspace Methods: Techniques like MUSIC, ESPRIT, or even high-resolution spectral estimation methods model the signal as a sum of sinusoids. They can estimate the parameters (frequency, amplitude, phase) of each sinusoid even if they are closer than what the basic STFT resolution would allow.

            Sparse Representations/Matching Pursuit: These methods decompose a signal into a sum of basis functions (or atoms). If the signal is sparse in a particular dictionary, overlapping components might be separated more effectively than by using a fixed grid.
        '

    """