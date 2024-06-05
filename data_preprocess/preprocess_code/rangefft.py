import numpy as np
from numpy import pi
from time import time, sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as C
from scipy.signal import butter, filtfilt
from datetime import datetime
import math


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
    iq_data = np.array([float(x.decode().strip()) for x in final_frame])
    complex_data = iq_data[::2] + 1j * iq_data[1::2]
    fft_data = np.fft.fftshift(np.fft.fft(complex_data))
    fft_data = fft_data[(sample_number):]
    fft_data = np.array(fft_data).reshape(sample_number, 1)

    return fft_data

def dopplerfft(final_frame, sample_number, sampling_frequency_hz):
    iq_data = np.array([float(x.decode().strip()) for x in final_frame])
    complex_data = iq_data[::2] + 1j * iq_data[1::2]
    complex_data = np.reshape(complex_data, (sample_number, -1))

    fft_data = np.fft.fftshift(np.fft.fft2(complex_data))
    fft_data = np.abs(fft_data)

    return fft_data


def mti_filter(iq_mat):
    order = 7
    cutoff = 0.01
    fs = 2e3

    b, a = butter(order, cutoff, 'high', fs=fs)
    Data_RTI_complex_MTIFilt = np.zeros_like(iq_mat, dtype=np.complex128)
    for k in range(iq_mat.shape[0]):
        Data_RTI_complex_MTIFilt[k, :] = filtfilt(b, a, iq_mat[k, :])

    return Data_RTI_complex_MTIFilt

def showplt_fft(range_bins, time_bins, fft_data, name):
    plt.figure(figsize=(10, 6), num=name)
    plt.imshow(np.abs(fft_data), aspect='auto', extent=[time_bins[0], time_bins[-1], range_bins[-1], range_bins[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('Range-Time Map')
    plt.show()

    return

def showplt_fft2(doppler_bins, time_bins, doppler_data, name):
    plt.figure(figsize=(10, 6), num=name)
    plt.imshow(doppler_data, aspect='auto', extent=[time_bins[0], time_bins[-1], doppler_bins[-1], doppler_bins[1]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Doppler (Hz)')
    plt.title('Doppler-Time Map')
    plt.show()

    return

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
    fft2_data = []
    fft_data_per_frame = []
    fft_data_per_frame2 = []
    frame_title_size = 20
    frame_body_size = 519
    frame_head_size = 7
    frame_number = 78  # 10 s for 78 frames

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
    range_bins = np.linspace(0, max_distance, number_of_samples)
    doppler_bins = np.fft.fftfreq(samples_per_frame, d=1/sampling_frequency_hz)
    time_bins = np.array([frame_period_sec * i for i in range(frame_number)])
    fft_data = np.empty((number_of_samples, 0))
    doppler_data = np.empty((number_of_samples, 0))


    # BGT60LTR11AIP_waveForwardBackward_20240502_163953
    # BGT60LTR11AIP_waveLeftRight_20240502_164031
    # BGT60LTR11AIP_WalkBackward_20240515_141527
    # BGT60LTR11AIP_WalkForward_20240520_151537

    name = "BGT60LTR11AIP_WalkBackward_20240515_141527"
    file_name = f'./{name}.txt'

    with open(file_name, 'rb') as file:
        frame_title = [next(file) for _ in range(frame_title_size)]
        
        for i in range (frame_number):
            frame = [next(file) for _ in range(frame_body_size)]
            timestamp, frame_body = get_frame_from_file(frame_head_size, frame)
            fft_data_per_frame = rangefft(frame_body, number_of_samples)
            fft_data = np.hstack((fft_data, fft_data_per_frame))  
            doppler_data_per_frame = dopplerfft(frame_body, number_of_samples, sampling_frequency_hz)
            doppler_data = np.hstack((doppler_data, doppler_data_per_frame))  

        # test(file)

    file.close()
    print("fft_data shape: ", fft_data.shape)
    mti_filtered_data = mti_filter(np.real(fft_data))
    showplt_fft(range_bins, time_bins, mti_filtered_data, name)
    showplt_fft2(doppler_bins, time_bins, doppler_data, name)
    
if __name__ == '__main__':
    main()