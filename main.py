import cv2
import numpy as np
import os

import image_process
import text_detector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# img_path = "D:\\Documents\\perso\\ML\\Projet Text Recongnization\\Dataset\\lines\\a02\\a02-004\\a02-004-05.png"
img_path = "D:\\Documents\\perso\\ML\\pythonProject\\Dataset\\test_imgs\\img_4.png"
model_input_shape = (128, 32)
outputBoxes = []
outputCropImages = []

img = cv2.imread(img_path)

cv2.imshow('sample image', img)

img = text_detector.prepare_img(img)
cv2.imshow('sample image prepared', img)
detections = text_detector.detect(img,
                                  kernel_size=17,  # valeur initiale 25
                                  sigma=11,
                                  theta=7,
                                  min_area=220)  # valeur initiale 100

line = text_detector.sort_multiline(detections)[0]

for i, result in enumerate(line):
    outputBoxes.append(result.bbox)  # recuperer les boxes pour afficher les rect sur l'image
    expended_img = np.expand_dims(result.img, 2) # pour avoir un shape de dimensions (x, y, 1)
    outputCropImages.append(image_process.image_resize(expended_img, model_input_shape)) # recuperer les mots extraits pour predire la sortie avec le model

for cords in outputBoxes:
    cv2.rectangle(img, (int(cords.x), int(cords.y)), (int(cords.x + cords.w), int(cords.y + cords.h)), (0, 0, 255), 2)

cv2.imshow('sample image with rects', img)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()

'''
datasetPreprocessor = dataset_preprocess.DatasetPreprocessor()
datasetPreprocessor.set_train_test_val()
test_set = datasetPreprocessor.test_ds
model = tf.keras.models.load_model("Model\\handwritting_recognization_model.hdf5", custom_objects={'CTCLayer': CTCLayer})
test = None
for batch in test_set.cache().take(1).repeat():
    test = batch
    break
print(test_set.cache().take(1).repeat())
pred = model.predict(test)
pred_text = dataset_preprocess.decode_batch_predictions(pred, datasetPreprocessor)
print(pred_text)



plt.subplot(len(line), 1, 1)
plt.imshow(img, cmap='gray')
for i, word in enumerate(line):
  output.append(word.bbox)
  print(word.bbox)
  plt.subplot(len(line), 1, i + 1)
  plt.imshow(word.img, cmap='gray')
plt.show()


'''
