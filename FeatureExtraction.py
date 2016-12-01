import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Useful Constants
NOTE_RATIO = 2**(1./12.)
C_0_FREQ = 16.35

# Returns frequency of note number n, where n is the number of half steps above C_0
def note_freq(n):
    return C_0_FREQ * (NOTE_RATIO ** n)

# Create a frequency window/bin surrounding note n which grows exponentially
def note_freq_window(n):
    return [(C_0_FREQ/2.)*(NOTE_RATIO**(n-1))*(1+NOTE_RATIO), (C_0_FREQ/2.)*(NOTE_RATIO**n)*(1+NOTE_RATIO)]

def plot(x, y, xlabel, ylabel, title):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# DATA SMOOTHERS
# Smooths the data by applying a convolution with a "box" function
def smooth(y, width):
    frame = np.ones(width)/width
    out = np.convolve(y, frame, mode = 'same')
    return out

# Smooths data using a running average of the data
def run_avg(data, width):
    return [np.mean(chunk) for chunk in list_chunks(data,width)]

# Yields pieces of the list l which are length n
def list_chunks(l,n):
    # First pad the list until its length is divisible by n
    while len(l) % n != 0:
        l = np.append(l,0)
    for i in range(0,len(l),n):
        yield l[i:i+n]

# UNIT CONVERTERS
# Returns the value in Hz associated with each FFT_data entry
def FFT_indices_to_hz(FFT_data, rate):
    return [i*float(rate)/(2*float(len(FFT_data))) for i in range(len(wav_freq))]

# Returns the time in seconds associated with each wav_data entry
def wav_indices_to_sec(wav_data, rate):
    return [float(x)/float(rate) for x in range(len(wav_data))]

# Returns the magnitude of the real FFT of the wav_data
def get_FFT_data(wav_data, rate):
    return np.abs(np.fft.rfft(wav_data))

# FEATURE EXTRACTION METHODS
# Naively chooses the highest num_vals values in the data
def top_n_vals(data, x_vals, num_vals):
    result = []
    for _ in range(num_vals):
        index = np.argmax(data)
        result.append(x_vals[index])
        print x_vals[index]
        data = np.delete(data, index)
        x_vals = np.delete(x_vals, index)
    return result

# Finds all points which are taller than the points to their immediate left and right
def find_peaks(data, x_vals, thresh):
    f1 = np.abs(data[1:-1])
    f2 = np.abs(data[:-2])
    f3 = np.abs(data[2:])
    peaks = [x_vals[i] for i in range(len(data)-2) if f1[i] > f2[i] and f1[i] > f3[i] and f1[i] > thresh]
    return peaks

# Finds the average y value (intensity) between the x values low and high (frequency values)
def avg_intensity_in_window(low, high, x_vals, FFT_data):
    indices_btwn_low_high = [n for n,item in enumerate(x_vals) if item >= low and item <= high]
    if not(indices_btwn_low_high):
        return 0
    return np.mean(FFT_data[indices_btwn_low_high[0]:indices_btwn_low_high[-1]])

# Finds the max y value (intensity) between the x values low and high (frequency values)
def max_intensity_in_window(low, high, x_vals, FFT_data):
    indices_btwn_low_high = [n for n,item in enumerate(x_vals) if item >= low and item <= high]
    if not(indices_btwn_low_high):
        return 0
    return np.max(FFT_data[indices_btwn_low_high[0]:indices_btwn_low_high[-1]])

# Returns a list of the total avg intensity of each note's frequency bins
def total_avg_note_intensity(x_vals, FFT_data):
    # Obtains the frequency window and avg intensity for 96 notes ranging between 16 Hz and 7900 Hz
    windows = [note_freq_window(i) for i in range(1, 97)]
    avgs = [avg_intensity_in_window(i[0], i[1], x_vals, FFT_data) for i in range(1,97)]
    # Sums the averages among the same notes across different octaves (12 notes total)
    return [np.sum([avgs[i+12*j] for j in range(0,8)]) for i in range(0,12)]

def total_max_note_intensity(x_vals, FFT_data):
    # Obtains the frequency window and max intensity for 96 notes ranging between 16 Hz and 7900 Hz
    windows = [note_freq_window(i) for i in range(1, 97)]
    maxs = [max_intensity_in_window(i[0], i[1], x_vals, FFT_data) for i in range(1,97)]
    # Sums the max intensities among the same notes across different octaves (12 notes total)
    return [np.sum([maxs[i+12*j] for j in range(0,8)]) for i in range(0,12)]

for i in range(1,100):
    print note_freq(i), ' : ', note_freq_window(i)

rate, wav_data = wavfile.read("../Samples/841.wav")
plot_wav(wav_data, rate, "Original Audio")
plot_FFT(wav_data, rate, "FFT of Original Audio")