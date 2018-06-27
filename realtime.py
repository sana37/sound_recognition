import numpy as np
import pyaudio
import time
from wavdataset import DatasetManager
from model import Model


DEVICE_NAME = 'USB Device'
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
FRAMES_PER_BUFFER = 128 * 10

TIME = 128 / 48000


def main():
    model = Model()
    model.load()

    audio = pyaudio.PyAudio()
    n_device = audio.get_device_count()

    dev_id = -1
    for i in range(n_device):
        dev_name = audio.get_device_info_by_index(i)['name']

        if DEVICE_NAME in dev_name:
            dev_id = i
            print(dev_name)

    if dev_id == -1:
        print('required device is not found')
        quit()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=dev_id,
        frames_per_buffer=FRAMES_PER_BUFFER)


    spectrum_buf = [[0] * (DatasetManager.n_frame // 2)] * DatasetManager.image_size
    run_nn_idx = 0

    while True:
        available = stream.get_read_available()

        if available >= DatasetManager.n_frame:
            raw_bytes = stream.read(DatasetManager.n_frame)
            data = np.frombuffer(raw_bytes, dtype=np.int16) / 32768

            spectrum = DatasetManager.get_spectrum(data)

            spectrum_buf.pop(0)
            spectrum_buf.append(spectrum)

            run_nn_idx += 1
            if run_nn_idx % 3 == 0:
                run_nn(model, spectrum_buf)
                run_nn_idx = 0


        else:
            time.sleep(0.0005)

    stream.stop_stream()
    stream.close()

    audio.terminate()


def run_nn(model, image):
    image_array = np.array(image)
    image_array = np.reshape(image_array, [-1, Model.n_frame, Model.n_frequency, 1])
    result = model.run(image_array)
    if result[0] >= 0.8:
        print(" {0:.3} ".format(result[0]))


def getMaxFrequency(image, sampleTime):
    spectrum = image[-1]
    maxIdx = spectrum.index(max(spectrum))
    frequency = maxIdx / sampleTime
    print(frequency)


if __name__ == '__main__':
    main()
