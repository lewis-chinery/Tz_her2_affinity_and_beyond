# training code adapted from https://github.com/dahjan/DMS_opt
# final optimised params from SI https://www.nature.com/articles/s41551-021-00699-9

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from src.classifiers.cnn.model import create_cnn
import matplotlib.pyplot as plt

MASON_BATCH_SIZE = 16
MASON_PARAMS = [['CONV', 400, 5, 1],
                ['DROP', 0.2],
                ['POOL', 2, 1],
                ['FLAT'],
                ['DENSE', 300]]


def fit_CNN(X_train, y_train, X_val, y_val, input_shape=(10,20), params=MASON_PARAMS,
                  shuffle=True, epochs=10, verbose=1, class_weight=None, learning_rate=0.000075,
                  batch_size=MASON_BATCH_SIZE, patience=5):
    """
    Classification of data with a convolutional neural
    network, followed by plotting of ROC and PR curves.

    Parameters
    ---
    params: optional; if provided, should specify the optimized
        model parameters that were determined in a separate model
        tuning step. If None, model parameters are hard-coded.
    """

    # Create the CNN with above-specified parameters
    CNN_classifier = create_cnn(params, input_shape, 'relu', None)

    # Compiling the CNN
    opt = Adam(learning_rate=learning_rate)
    CNN_classifier.compile(optimizer=opt, loss='binary_crossentropy',
                           metrics=['accuracy'])

    # Fit the CNN to the training set
    history = CNN_classifier.fit(
                x=X_train, y=y_train, shuffle=shuffle, validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight=class_weight,
                callbacks=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience))

    return CNN_classifier, history


def plot_loss(history, title="", figsize=(4, 3)):
    '''
    '''
    _, _ = plt.subplots(figsize=figsize)
    plt.rcParams["figure.facecolor"] = "white"
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"])
    plt.show()
