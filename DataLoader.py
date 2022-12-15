import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import image_process

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Sample = namedtuple('Sample', 'text, file_path')
AUTOTUNE = tf.data.AUTOTUNE


class DataLoaderClass:

    def __init__(self,
                 data_dir,
                 batch_size: int = 10,
                 train_split: float = 0.8
                 ) -> None:

        # on verifie si le repertoire du dataset existe sinon on leve une exception
        assert os.path.getsize(
            os.path.join(data_dir, "words.txt")), "Le repertoire vers les donnees n'existe pas ou est incorrect"
        self.batch_size = batch_size
        self.samples = []
        self.max_len = 0
        self.chars = set()
        self.dictionnary = set()
        self.padding_val = 99  # value to complete a encoding sentence matrix when len < to max len

        lines_in_text = open(os.path.join(data_dir, "words.txt"), "r").readlines()

        # on parcours chaque ligne de notre fichier
        for line in lines_in_text:

            # on verifie si la ligne n'est pas une ligne vide ou ne commence pas
            # par # ce qui signifie que c'est un commentaire

            if not line or line[0] == "#":
                continue

            line = line.strip()
            elem = line.split(" ")
            img_name = elem[0]
            img_path_elems = img_name.split("-")
            text = elem[-1]
            img_path = os.path.join(data_dir, "words", img_path_elems[0],
                                    img_path_elems[0] + "-" + img_path_elems[1],
                                    img_name + ".png")

            if os.path.getsize(img_path) and elem[1] != "err":
                self.chars = self.chars.union(set(list(text)))
                for word in text.split(" "):
                    self.dictionnary.add(word)
                self.samples.append(Sample(text, img_path))
                self.max_len = max(self.max_len, len(text))  # pour retenir la longueur du plus long texte

            len_trainset = int(train_split * len(self.samples))
            len_valset = int((len(self.samples) - len_trainset) / 2)
            len_testset = len(self.samples) - len_trainset - len_valset

            self.trainSample = self.samples[0: len_trainset]
            self.valSample = self.samples[len_trainset: len_trainset + len_testset]
            self.testSample = self.samples[len_trainset + len_testset:len(self.samples)]

        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=list(self.chars), mask_token=None
        )

        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

        self.trainSet = self.prepare_datasets(self.trainSample)
        self.valSet = self.prepare_datasets(self.valSample)

    # fonction permettant de traiter et encoder l'image et le texte
    def encode_single_sample(self, img_path, text):
        # on traite l'image
        img = image_process.preprocess_image(img_path, (1024, 128))
        # on traite le label
        # label = self.char_to_num(tf.strings.unicode_split(text, input_encoding="UTF-8"))
        label = self.convert_label_to_num(text)
        return {"image": img, "label": label}

    # pour convertir une phrase en donnees numeriques
    def convert_label_to_num(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=self.padding_val)
        return label

    # pour convertir une phrase encodee en texte
    def convert_num_to_label(self, enc_label):
        indices = tf.gather(enc_label, tf.where(tf.math.not_equal(enc_label, self.padding_val)))
        label = tf.strings.reduce_join(self.num_to_char(indices))
        label = label.numpy().decode("UTF-8")
        return label

    # decoder un batch de donnees encodees
    def decode_batch_predictions(self, predictions):
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        results = keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0][:,
                  :self.max_len]
        output_text = [self.convert_num_to_label(res) for res in results]
        return output_text

    # fonction qui va nous creer un dataset sous la forme image, label avec les 2 donnees deja traites pretes
    # pour le model
    def prepare_datasets(self, samples):
        x = []
        y = []

        for item in samples:
            x.append(item.file_path)
            y.append(item.text)

        x = np.array(x)
        y = np.array(y)

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(
            self.encode_single_sample, num_parallel_calls=AUTOTUNE
        )

        return dataset.batch(self.batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    def visualize_data(self):
        _, ax = plt.subplots(4, 4, figsize=(10, 5))
        for batch in self.trainSet.take(1):
            images = batch["image"]
            labels = batch["label"]
            for i in range(16):
                img = (images[i] * 255).numpy().astype("uint8")
                label = self.convert_num_to_label(labels[i])
                print(label)
                ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
                ax[i // 4, i % 4].set_title(label)
                ax[i // 4, i % 4].axis("off")
        plt.show()
