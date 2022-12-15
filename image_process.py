import cv2
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def crop_image(img, cords):
    img_croped = img[int(cords[1]): int(cords[3]), int(cords[0]): int(cords[2])]
    return img_croped


def renforce_contours(img, thresh):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    return thresh_img


def image_resize(img, img_size):
    w, h = img_size
    image = tf.image.resize(
        img,
        size= (h, w),
        preserve_aspect_ratio=True
    )

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        pad_height_top = pad_height//2 + 1
        pad_height_bottom = pad_height//2
    else:
        pad_height_top = pad_height_bottom = pad_height//2

    if pad_width % 2 != 0:
        pad_width_left = pad_width // 2 + 1
        pad_width_right = pad_width // 2
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0]
        ]
    )

    image = tf.transpose(
        image,
        perm=[1, 0, 2]
    )

    image = tf.image.flip_left_right(image)

    return image


def preprocess_image(image_path, img_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = image_resize(image, img_size)
    image = tf.cast(
        image, tf.float32
    ) / 255.0
    return image


def show_preprocessed_image(img_preprocessed):
    img_dec = tf.image.flip_left_right(img_preprocessed)
    img_dec = tf.transpose(img_dec, perm=[1, 0, 2])
    img_dec = (img_dec * 255.0).numpy().clip(0, 255).astype(np.uint8)
    img_dec = img_dec[:, :, 0]

    cv2.imshow('sample image resized', img_dec)