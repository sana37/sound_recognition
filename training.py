import numpy as np
import matplotlib.pyplot as plt
from model import Model
from wavdataset import DatasetManager

def main():
    model = Model()
    model.setup_training()
    model.setup_accuracy()
    model.initialize_variables()

    dataset_manager = DatasetManager()

    '''
    # training
    for i in range(1000):
        if i % 50 == 0:
            batch_x, batch_y = dataset_manager.get_batch(batch_size=200)
            batch_x = np.reshape(batch_x, [-1, Model.n_frame, Model.n_frequency, 1])
            print("accuracy({0}):".format(i), model.calc_accuracy(batch_x, batch_y))

        batch_x, batch_y = dataset_manager.get_batch(batch_size=50)
        batch_x = np.reshape(batch_x, [-1, Model.n_frame, Model.n_frequency, 1])
        model.exec_training(batch_x, batch_y, 0.5)

    # evaluate
    batch_x, batch_y = dataset_manager.get_batch(batch_size=50)
    batch_x = np.reshape(batch_x, [-1, Model.n_frame, Model.n_frequency, 1])
    print("final accuracy:", model.calc_accuracy(batch_x, batch_y))

    model.save()
    '''

## test
    for i in range(20):
        batch_x, batch_y = dataset_manager.get_batch(batch_size=1)
        batch_x_re = np.reshape(batch_x, [-1, Model.n_frame, Model.n_frequency, 1])
        print(model.run(batch_x_re))
        print('answer:', batch_y)
#        print('max:', max([max(vec) for vec in batch_x[0]]))
#        print('min:', min([min(vec) for vec in batch_x[0]]))
        print()
        plt.imshow(batch_x[0], vmin=0.0, vmax=2.0)
        plt.colorbar()
        plt.show()
##

#    dataset_manager.release_dataset()


if __name__ == '__main__':
    main()
