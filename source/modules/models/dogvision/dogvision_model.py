import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import numpy as np
import pandas as pd

#Create a function for preprocessing images
def process_image_from_bytes(file_bytes, img_size=224):
    image = tf.io.decode_jpeg(file_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_size, img_size])
    return image

def create_data_batches(X, batch_size=32):
  #If the data is a test dataset, we probably don't have labels
  print("Creating test data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) #only bytes (no labels)
  data_batch = data.map(process_image_from_bytes).batch(batch_size)
  return data_batch

def get_pred_label(prediction_probabilities):
  labels_csv = pd.read_csv('modules/models/dogvision/labels.csv')
  labels = labels_csv['breed'].to_numpy()
  unique_breeds = np.unique(labels)

  return unique_breeds[np.argmax(prediction_probabilities)]


def load_model(model_path):
  print(f'Loading saved model from {model_path}')

  #model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
  model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

  return model

loaded_model = load_model('modules/models/dogvision/dogVision.h5')


def get_dogs_breed(contents: list[bytes]) -> list[str]:
    labels = []
    custom_predictions = loaded_model.predict(create_data_batches(contents), verbose=1)
    for i in custom_predictions:
        custom_pred_labels = get_pred_label(i)
        labels.append(custom_pred_labels)
        
    return labels
        
        
#Test code
#with open("dog.jpg", "rb") as a, open("dog2.jpg", "rb") as b :
#    get_dogs_breed([a.read(), b.read()])