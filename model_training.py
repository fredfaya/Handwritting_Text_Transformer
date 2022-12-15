from DataLoader import DataLoaderClass
import os

from Model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_path = "Dataset"
path = os.path.join(dataset_path, "words.txt")
model_input_shape = (128, 32)

myDataLoader = DataLoaderClass(dataset_path, 10)
print("Finishing with dataset preprocess")
Model(model_input_shape, myDataLoader).train_model(100)
