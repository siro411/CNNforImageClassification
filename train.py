from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from scipy.io import loadmat
import pandas as pd
import numpy as np
from keras import backend as K
from wide_resnet import WideResNet




def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008



def main():


    input_path = "/Users/annaying/final project/UTKFace.mat"
    batch_size = 64
    nb_epochs = 20
    lr = 0.1
    depth = 10
    k = 8
    validation_split = 0.1
    output_path = Path(__file__).resolve().parent.joinpath("/Users/annaying/final project")
    output_path.mkdir(parents=True, exist_ok=True)



    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender)[:,[0,1]]
    y_data_a = np_utils.to_categorical(age, 101)

    model = WideResNet(image_size, depth=depth, k=k)()
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    #"Model summary..."
    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",  #02d 宽度2，自动补齐左边（右对齐）.2f 保留小数点后两位
                                 monitor="val_loss",
                                 verbose=1,   #有进度条为1
                                 save_best_only=True, #设置当为True时，将只保存在验证集上性能最好的模型
                                 mode="auto")
                 ]

    #"Running training..."

    data_num = len(X_data)
    indices = np.arange(data_num)
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_data_g = y_data_g[indices]
    y_data_a = y_data_a[indices]
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]



    hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                         validation_data=(X_test, [y_test_g, y_test_a]))

    #"Saving history..."
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()