import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

from resins.preparing import UNet

IMG_HEIGHT = 512
IMG_WIDTH = 512
ORG_IMG_HEIGHT = 2272
ORG_IMG_WIDTH = 1704
IMG_CHANNELS = 3


def read_image(img_path, mask_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    image = np.asarray(image)
    mask = np.asarray(mask)
    label = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1))
    for i, row in enumerate(mask):
        for j, col in enumerate(row):
            if sum(col) > 600:
                label[i][j][0] = 255
    image = image / 255
    label = label / 255
    return image, label


def model_train():
    model = UNet().create_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    train_image, train_mask = read_image("train/img_train.png", "train/label_train.png")

    val_image, val_mask = read_image("train/img_val.png", "train/label_val.png")

    model.fit(np.array([train_image, val_image]), np.array([train_mask, val_mask]), batch_size=1, epochs=200,
              validation_split=0.5)

    model.save("UNetW.h5")


def main():
    # model_train()
    model = keras.models.load_model("UNetW.h5")
    val_image, val_mask = read_image("train/img_train.png", "train/label_train.png")

    result = model.predict(np.array([val_image]))
    result = result > 0.5

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.reshape(val_mask * 255, (IMG_WIDTH, IMG_HEIGHT)), cmap="gray")

    ax = fig.add_subplot(1, 2, 2)
    cv2.imwrite("result.png",
                cv2.resize(np.float32(result[0] * 255), (ORG_IMG_HEIGHT, ORG_IMG_WIDTH), interpolation=cv2.INTER_AREA))
    ax.imshow(np.reshape(result[0] * 255, (IMG_WIDTH, IMG_HEIGHT)), cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
