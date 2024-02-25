import numpy as np
import cv2
import matplotlib.pyplot as plt
# video_path = './newVideo.mp4'
# x,y,w,h = (605,198,25,28);
def roiFreq(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    n = len(frames)
    fft_frames = []
    # upd_frames = []
    # for i in range(n):
    #     upd_frames.append(frames[i][y:y + h + 1, x:x + w + 1])
    # frames = np.array(upd_frames)
    # print(frames.shape)
    frames = np.array(frames)
    # print(frames.shape)
    for i in range(n):
        fft_frames.append(np.fft.fft2(frames[i]))
    imag = [np.imag(fft_frame) for fft_frame in fft_frames]
    real = [np.real(fft_frame) for fft_frame in fft_frames]

    phi = [np.arctan(imag[i]/real[i]) for i in range(len(fft_frames))]

    x0 = int(len(frames[0]) / 2)
    y0 = int(len(frames[0][0]) / 2)

    del_phi_del_x = np.diff(phi, axis=2)
    del_phi_del_y = np.diff(phi, axis=1)
    del_phi_del_t = np.diff(phi, axis=0)
    del_phi_del_x = np.concatenate((np.zeros((n, len(frames[0]), 1)), del_phi_del_x), axis=2)
    del_phi_del_y = np.concatenate((np.zeros((n, 1, len(frames[0][0]))), del_phi_del_y), axis=1)
    del_phi_del_t = np.concatenate((np.zeros((1, len(frames[0]), len(frames[0][0]))), del_phi_del_t), axis=0)
    omega_x = del_phi_del_x[:, x0, :]
    omega_y = del_phi_del_y[:, :, y0]

    del2_phi_del_x_del_t = np.diff(del_phi_del_x, axis=0)
    del2_phi_del_y_del_t = np.diff(del_phi_del_y, axis=0)
    del2_phi_del_x2 = np.diff(del_phi_del_x, axis=2)
    del2_phi_del_y2 = np.diff(del_phi_del_y, axis=1)
    del2_phi_del_x_del_t = np.concatenate((np.zeros((1, len(frames[0]), len(frames[0][0]))), del2_phi_del_x_del_t),
                                          axis=0)
    del2_phi_del_y_del_t = np.concatenate((np.zeros((1, len(frames[0]), len(frames[0][0]))), del2_phi_del_y_del_t),
                                          axis=0)
    del2_phi_del_x2 = np.concatenate((np.zeros((n, len(frames[0]), 1)), del2_phi_del_x2), axis=2)
    del2_phi_del_y2 = np.concatenate((np.zeros((n, 1, len(frames[0][0]))), del2_phi_del_y2), axis=1)

    # del2_phi_del_x2[del2_phi_del_x2 == 0] = 1e-10
    # del2_phi_del_y2[del2_phi_del_y2 == 0] = 1e-10
    # disp_x = np.sum(np.sum(del2_phi_del_x_del_t/del2_phi_del_x2, axis=2), axis=0)
    # disp_y = np.sum(np.sum(del2_phi_del_y_del_t/del2_phi_del_y2, axis=1), axis=0)

    del2_phi_del_x2_nonzero = np.where(del2_phi_del_x2 == 0, 1, del2_phi_del_x2)
    del2_phi_del_y2_nonzero = np.where(del2_phi_del_y2 == 0, 1, del2_phi_del_y2)

    disp_x = np.sum(
        np.sum(np.where(del2_phi_del_x2_nonzero != 0, del2_phi_del_x_del_t / del2_phi_del_x2_nonzero, 0), axis=2),
        axis=1)
    disp_y = np.sum(
        np.sum(np.where(del2_phi_del_y2_nonzero != 0, del2_phi_del_y_del_t / del2_phi_del_y2_nonzero, 0), axis=1),
        axis=1)

    disp_norm_x = ((disp_x - min(disp_x)) / (max(disp_x) - min(disp_x))) * 10
    disp_norm_y = ((disp_y - min(disp_y)) / (max(disp_y) - min(disp_y))) * 10

    fft_disp_norm_x = np.fft.fft(disp_norm_x)
    freqs_x = np.fft.fftfreq(len(frames), d=(1 / (4 * fps)))
    fft_disp_norm_y = np.fft.fft(disp_norm_y)
    freqs_y = np.fft.fftfreq(len(frames), d=(1 / (4 * fps)))
    # print(freqs_x[freqs_x >= 0])
    # print(freqs_y[freqs_y >= 0])
    # print(np.abs(fft_disp_norm_x[freqs_x >= 0]))
    # print(np.abs(fft_disp_norm_y[freqs_y >= 0]))
    # fft_disp_norm_x = np.fft.fft(disp_norm_x)
    # freqs_x = np.fft.fftfreq(len(disp_norm_x), d=(1/(2*fps)))
    # print(fps)
    # fft_disp_norm_y = np.fft.fft(disp_norm_y)
    # freqs_y = np.fft.fftfreq(len(disp_norm_y), d=(1/(2*fps)))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(freqs_x[freqs_x >= 0][1:], np.abs(fft_disp_norm_x[freqs_x >= 0])[1:] / 2,
             label='Amplitude vs. Frequency - X')
    plt.title('Amplitude vs. Frequency - X')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs_y[freqs_y >= 0][1:], np.abs(fft_disp_norm_y[freqs_y >= 0])[1:] / 2,
             label='Amplitude vs. Frequency - Y')
    plt.title('Amplitude vs. Frequency - Y')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.specgram(disp_norm_x, Fs=fps, cmap='viridis', aspect='auto')
    plt.title('Spectrogram - X')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.specgram(disp_norm_y, Fs=fps, cmap='viridis', aspect='auto')
    plt.title('Spectrogram - Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # plt.figure(figsize=(12, 6))
    # time = int(len(frames)/fps)+1
    # plt.subplot(2, 1, 1)
    # plt.plot(np.abs(fft_disp_norm_x[freqs_x >= 0]), [i for i in range(time)], label='Amplitude vs. Time - X')
    # plt.title('AMPLITUDE VS TIME X')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude X')

    # plt.subplot(2, 1, 2)
    # plt.plot(np.abs(fft_disp_norm_y[freqs_y >= 0]), [i for i in range(time)], label='Amplitude vs. Time - Y')
    # plt.title('AMPLITUDE VS TIME Y')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude Y')

    plt.tight_layout()
    plt.show()
    return freqs_x[freqs_x >= 0], freqs_y[freqs_y >= 0]