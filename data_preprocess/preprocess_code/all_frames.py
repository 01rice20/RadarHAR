import numpy as np
from numpy import pi
from time import time, sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as C
from scipy.signal import butter, filtfilt
from datetime import datetime


def get_frame_from_file(frame_head_size, frame):
    frame_head = []
    frame_body = []

    timestamp_line = next(line for line in frame if b'Timestamp' in line)
    timestamp = int(timestamp_line.split(b'=')[1])
    datetime_obj = datetime.fromtimestamp(timestamp / 1000.0)
    # time_str = datetime_obj.strftime('%H:%M:%S.%f')
    frame_body = frame[frame_head_size:]

    return datetime_obj, frame_body

def rangefft(final_frame, sample_number):
    iq_data = [float(x.decode().strip()) for x in final_frame]
    complex_data = np.array(iq_data[::2]) + 1j * np.array(iq_data[1::2])
    fft_data = np.fft.fftshift(np.fft.fft(complex_data))
    fft_data = fft_data[(sample_number):]
    fft_data = np.array(fft_data).reshape(sample_number, 1)

    return fft_data

def mti_filter(iq_mat):
    order = 4
    cutoff = 0.01
    fs = 2e3

    b, a = butter(order, cutoff, 'high', fs=fs)
    Data_RTI_complex_MTIFilt = np.zeros_like(iq_mat, dtype=np.complex128)
    for k in range(iq_mat.shape[0]):
        Data_RTI_complex_MTIFilt[k, :] = filtfilt(b, a, iq_mat[k, :])

    return Data_RTI_complex_MTIFilt

# Test if the frame_body length is correct
def test(file):
    rest = []
    for line in file:
        rest.append(line.strip())
    
    print("rest: ", rest)

    return

def main():

    frame = []
    frame_body = []
    fft_data = []
    fft_data_per_frame = []
    frame_title_size = 20
    frame_body_size = 519
    frame_head_size = 7
    frame_number = 156  # 10 s for 78 frames

    # Radar Parameters
    center_rf_frequency_khz = 6.1044e7
    sampling_frequency_hz = 2e3
    adc_resolution_bits = 12
    number_of_samples = 128
    samples_per_frame = 512
    frame_period_sec = 0.128
    c = 3e8
    bandwidth = 0.41e9
    max_distance = 7
    index = np.arange(1, number_of_samples+1)
    range_bins = np.linspace(0, max_distance, number_of_samples)
    time_bins = np.array([frame_period_sec * i for i in range(frame_number)])
    fft_data = np.empty((number_of_samples, 0))

    # BGT60LTR11AIP_waveForwardBackward_20240502_163953.txt
    # BGT60LTR11AIP_waveLeftRight_20240502_164031.txt
    # BGT60LTR11AIP_normal_20240502_163517.txt
    # BGT60LTR11AIP_withoutBreathing_20240502_163737.txt

    with open('../data/radar/BGT60LTR11AIP_waveForwardBackward_20240502_163953.txt', 'rb') as file:
        frame_title = [next(file) for _ in range(frame_title_size)]
        
        for i in range (frame_number):
            frame = [next(file) for _ in range(frame_body_size)]
            timestamp, frame_body = get_frame_from_file(frame_head_size, frame)
            fft_data_per_frame = rangefft(frame_body, number_of_samples)
            fft_data = np.hstack((fft_data, fft_data_per_frame))  
        
        # test(file)

    file.close()

    mti_filtered_data = mti_filter(np.real(fft_data))

    fig, ax = plt.subplots()
    img = ax.imshow(np.abs(mti_filtered_data[0]).reshape(-1, 1), aspect='auto', extent=[0, number_of_samples, range_bins[-1], range_bins[0]])
    plt.colorbar(img, ax=ax, label='Magnitude')
    # ax.set_xticks([]) 
    # ax.set_yticks([])
    ax.set_xlabel('Sample')
    ax.set_ylabel('Range (m)')
    ax.set_title('Range-Time Map')

    def update(frame):
        img.set_array(np.abs(mti_filtered_data[frame]).reshape(-1, 1))
        if frame == frame_number - 1:
            ani.event_source.stop()

        return img,

    ani = FuncAnimation(fig, update, frames=range(frame_number), interval=128, blit=True)
    plt.show()
    
if __name__ == '__main__':
    main()