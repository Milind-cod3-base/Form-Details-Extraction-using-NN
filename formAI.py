# module to hold the methods required for the AI


from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# defining a list of characters containing lower case, upper case nd special characters
characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', 
            '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

max_len = 21

AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

image_width = 128
image_height = 32
batch_size = 64
padding_token = 99
        

class HandAI():


    
    def __init__(self, name):
        self.name = name
        
        

    
    def distortion_free_resize(self, image, img_size):
      w, h = img_size
      image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

      # Check tha amount of padding needed to be done.
      pad_height = h - tf.shape(image)[0]
      pad_width = w - tf.shape(image)[1]

      # Only necessary if you want to do same amount of padding on both sides.
      if pad_height % 2 != 0: 
          height = pad_height // 2
          pad_height_top = height + 1
          pad_height_bottom = height
      else:
          pad_height_top = pad_height_bottom = pad_height // 2

      if pad_width % 2 != 0:
          width = pad_width // 2
          pad_width_left = width + 1
          pad_width_right = width
      else:
          pad_width_left = pad_width_right = pad_width // 2

      image = tf.pad(
          image,
          paddings=[
              [pad_height_top, pad_height_bottom],
              [pad_width_left, pad_width_right],
              [0, 0],
          ],
      )

      image = tf.transpose(image, perm=[1, 0, 2])
      image = tf.image.flip_left_right(image)
      return image


    def preprocess_image(self, image_path, img_size=(image_width, image_height)):
      image = tf.io.read_file(image_path)
      image = tf.image.decode_png(image, 1)
      image = self.distortion_free_resize(image, img_size)
      image = tf.cast(image, tf.float32) / 255.0
      return image


    def vectorize_label(self, label):
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
        return label


    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}


    def prepare_dataset(self, image_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
            self.process_images_labels, num_parallel_calls=AUTOTUNE
        )
        return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


        
class CTCLayer(keras.layers.Layer):


    def __init__(self, name=None,**kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred



# def calculate_edit_distance(labels, predictions):
#     # Get a single batch and convert its labels to sparse tensors.
#     saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

#     # Make predictions and convert them to sparse tensors.
#     input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
#     predictions_decoded = keras.backend.ctc_decode(
#         predictions, input_length=input_len, greedy=True
#     )[0][0][:, :max_len]
#     sparse_predictions = tf.cast(
#         tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
#     )

#     # Compute individual edit distances and average them out.
#     edit_distances = tf.edit_distance(
#         sparse_predictions, saprse_labels, normalize=False
#     )
#     return tf.reduce_mean(edit_distances)


# class EditDistanceCallback(keras.callbacks.Callback):
#     def __init__(self, pred_model):
#         super().__init__()
#         self.prediction_model = pred_model

#     def on_epoch_end(self, epoch, logs=None):
#         edit_distances = []

#         for i in range(len(validation_images)):
#             labels = validation_labels[i]
#             predictions = self.prediction_model.predict(validation_images[i])
#             edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

#         print(
#             f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
#         )


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text




def calculate_results(y_true, y_pred):
  
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results







reconstructed_model = keras.models.load_model("my_h5_model-1.h5", custom_objects={"CTCLayer": CTCLayer})

np.random.seed(42)
dir1_path = '/content/kuch_bhi'
def get_image_paths_and_labels1(dir1_path):
  paths1 = []
  corrected_samples1 = []
  for f in os.listdir(dir1_path):
    path = dir1_path + '/' + f
    sample = f.split(".")
    paths1.append(path)
    corrected_samples1.append(sample[0])
  return paths1, corrected_samples1



paths, samples = get_image_paths_and_labels1(dir1_path)


a = HandAI(name=123)
test_ds = a.prepare_dataset(paths, samples)

preds = reconstructed_model.predict(test_ds)


preds_text = decode_batch_predictions(preds)


#print(preds_text)