import wave
import numpy as np
import scipy.fftpack as fftpack
import random
import os


class DatasetManager:
    n_frame = 128
    image_size = 64
    true_dir_list = [
        "wavdataset/v2/mix/true"]
#        "wavdataset/v2/mix/true2",
#        "wavdataset/v2/mix/true_noise",
#        "wavdataset/v2/mix/true2_noise"]
    false_dir_list = [
        "wavdataset/v2/mix/false_noise"]
#        "wavdataset/v2/mix/false_noise_noise",
 #       "wavdataset/v2/mix/false_high",
  #      "wavdataset/v2/mix/false_high_variety",
   #     "wavdataset/v2/mix/false_high_variety2",
    #    "wavdataset/v2/mix/false_clap",
#        "wavdataset/v2/mix/false_clap_variety"]
    n_true_max = 750
    n_false_max = 1500
    max_spectrum = 10


    def __init__(self):
        self.images = []
        self.labels = []

        print("now, fetch dataset")
        self.fetch_dataset()

        self.n_data = len(self.labels)
        self.batch_idx = 0

        self.shuffle_dataset()

        print("{0} images fetched.".format(self.n_data))
#        print(self.labels)


    def release_dataset(self):
        self.images = []
        self.labels = []
        self.n_data = 0
        self.batch_idx = 0


    def get_batch(self, batch_size = 20):
        if batch_size > self.n_data:
            print("batch size which you required is too big!!")
            raise Exception()

        if self.batch_idx + batch_size <= self.n_data:
            batch_images = self.images[self.batch_idx : self.batch_idx + batch_size]
            batch_labels = self.labels[self.batch_idx : self.batch_idx + batch_size]
        else:
            batch_images = self.images[self.batch_idx : ] + self.images[ : (self.batch_idx + batch_size) % self.n_data]
            batch_labels = self.labels[self.batch_idx : ] + self.labels[ : (self.batch_idx + batch_size) % self.n_data]
            self.shuffle_dataset()

        self.batch_idx = (self.batch_idx + batch_size) % self.n_data

        return (
            np.array(batch_images),
            np.array(batch_labels)
        )


    def shuffle_dataset(self):
        random_idx_list = list(range(self.n_data))
        random.shuffle(random_idx_list)

        self.images = [self.images[i] for i in random_idx_list]
        self.labels = [self.labels[i] for i in random_idx_list]


    def fetch_dataset(self):
        n_true = 0
        for dir in DatasetManager.true_dir_list:
            for file_name in os.listdir(dir):
                new_images = DatasetManager.read_file(dir + '/' + file_name, 1)

                self.images.extend(new_images)
                self.labels.extend([1])

                n_true += 1
#                if n_true >= DatasetManager.n_true_max:
#                    break


        n_false = 0
        for dir in DatasetManager.false_dir_list:
            for file_name in os.listdir(dir):
                new_images = DatasetManager.read_file(dir + '/' + file_name, 1)

                self.images.extend(new_images)
                self.labels.extend([0])

                n_false += 1
#                if n_false >= DatasetManager.n_false_max:
#                    break


        '''
        for file_name in os.listdir(DatasetManager.directory_name_false):
            new_images = DatasetManager.read_file(DatasetManager.directory_name_false + '/' + file_name, 1)
#            new_images = DatasetManager.read_file(
#                DatasetManager.directory_name_false + '/' + file_name,
#                DatasetManager.n_false_max - n_false)

            self.images.extend(new_images)
            self.labels.extend([0])
#            self.labels.extend([0] * len(new_images))

            n_false += 1
#            n_false += len(new_images)
            if n_false >= DatasetManager.n_false_max:
                break
        '''

        print("true data size: {0}, false data size: {1}".format(n_true, n_false))


    def read_file(file_name, n_image_max = None):
        wf = wave.open(file_name, "rb")
        framerate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channel = wf.getnchannels()

        n_allframe = wf.getnframes()
        buf = wf.readframes(n_allframe)
        wf.close()

        if sampwidth != 2 or channel != 1 or framerate != 48000:
            print("wav file format is incorrect")
            raise Exception()

        wav_data = np.frombuffer(buf, dtype=np.int16) / 32768

        images = []
        spectrums_image = []
        spectrum_idx = 0

        for i in np.arange(0, n_allframe, DatasetManager.n_frame): # n_frame // 2
            if i + DatasetManager.n_frame > n_allframe:
                break

            spectrum = DatasetManager.get_spectrum(wav_data[i : i + DatasetManager.n_frame])
            spectrums_image.append(spectrum)
            spectrum_idx += 1

            if spectrum_idx >= DatasetManager.image_size:
                images.append(spectrums_image)

                if (n_image_max is not None) and (len(images) == n_image_max):
                    break

                spectrums_image = []
                spectrum_idx = 0

        if images == []:
            print("the wav file is too small!!")
            raise Exception()

        return images


    def get_spectrum(data):
        windowed_data = data * np.hamming(DatasetManager.n_frame)

        dft = fftpack.fft(windowed_data)[:DatasetManager.n_frame // 2]
        spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) / DatasetManager.max_spectrum for c in dft]

        return spectrum
