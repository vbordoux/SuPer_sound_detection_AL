# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classification over embeddings."""

import dataclasses
from typing import Sequence

from chirp.inference.classify import data_lib
from chirp.models import metrics
import numpy as np
import tensorflow as tf
from ml_collections import config_dict
from chirp.inference import embed_lib
import pandas as pd
import os


@dataclasses.dataclass
class ClassifierMetrics:
  test_logits: dict[str, np.ndarray]
  test_labels: np.ndarray
  test_auc_roc: float



def train_embedding_model(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    train_ratio: float | None,
    train_examples_per_class: int | None,
    num_epochs: int,
    random_seed: int,
    batch_size: int,
    learning_rate: float | None = None,
    exclude_eval_classes: Sequence[int] = (),
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""

  # Create training and test splits
  train_locs, test_locs, _ = merged.create_random_train_test_split(
      train_ratio, train_examples_per_class, random_seed,
      exclude_eval_classes=exclude_eval_classes,
  )

  # Train the model and return metrics 
  test_metrics = train_from_locs(
      model=model,
      merged=merged,
      train_locs=train_locs,
      test_locs=test_locs,
      num_epochs=num_epochs,
      batch_size=batch_size,
      learning_rate=learning_rate,
  )
  return test_metrics



def train_from_locs(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    train_locs: Sequence[int],
    test_locs: Sequence[int],
    num_epochs: int,
    batch_size: int,
    learning_rate: float | None = None,
    use_bce_loss: bool = True,
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""
  if use_bce_loss:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  else:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=loss,
      metrics=[
          tf.keras.metrics.Precision(top_k=1, name='top1prec'),
          tf.keras.metrics.AUC(
              curve='ROC', name='auc', from_logits=True, multi_label=True
          ),
          tf.keras.metrics.RecallAtPrecision(0.9, name='recall0.9'),
      ],
  )

  train_ds = merged.create_keras_dataset(train_locs, True, batch_size)
  test_ds = merged.create_keras_dataset(test_locs, False, batch_size)

  model.fit(train_ds, epochs=num_epochs, verbose=0)

  # Compute overall metrics to avoid online approximation error in Keras.
  test_logits = model.predict(test_ds, verbose=0, batch_size=8)
  test_labels_hot = merged.data['label_hot'][test_locs]
  test_labels = merged.data['label'][test_locs]

  # Create a dictionary of test logits for each class.
  test_logits_dict = {}
  for k in set(test_labels):
    lbl_locs = np.argwhere(test_labels == k)[:, 0]
    test_logits_dict[k] = test_logits[lbl_locs, k]

  auc_roc = metrics.roc_auc(test_logits, test_labels_hot)

  return ClassifierMetrics(
      test_logits,
      test_labels,
      auc_roc,
  )


def train_from_index(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    num_epochs: int,
    train_locs: Sequence[int],
    test_locs: Sequence[int],
    batch_size: int,
    learning_rate: float | None = None,
    exclude_eval_classes: Sequence[int] = (),
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""

  # Train the model and return metrics 
  test_metrics = train_from_locs(
      model=model,
      merged=merged,
      train_locs=train_locs,
      test_locs=test_locs,
      num_epochs=num_epochs,
      batch_size=batch_size,
      learning_rate=learning_rate,
  )
  return test_metrics



def choose_embedding_model(model_name, models_folder_path="./models/"):
    '''
    Load model, corresponding paramaters and run a test according to the model selected.
    Args:
        model_name: str. Name of the model to load (surfperch, perch, birdnet)
        models_folder_path: str. Path to the folder containing the models
    Returns:
        embed_fn: Embedding function
        config: ConfigDict
    '''

    # Load and test the model corresponding to the name given in input
    config = config_dict.ConfigDict()
    embed_fn_config = config_dict.ConfigDict()
    model_config = config_dict.ConfigDict()

    # SurfPerch
    if model_name.lower() == 'surfperch':
        embed_fn_config.model_key = 'taxonomy_model_tf'
        model_config.window_size_s = 5.0
        model_config.hop_size_s = 5.0
        model_config.sample_rate = 32000
        model_config.model_path = models_folder_path + 'SurfPerch-model/'
    
    # Perch
    elif model_name.lower() == 'perch':
        embed_fn_config.model_key = 'taxonomy_model_tf'
        model_config.window_size_s = 5.0
        model_config.hop_size_s = 5.0
        model_config.sample_rate = 32000
        model_config.model_path = models_folder_path + 'Perch-model/'

    # BirdNET
    elif model_name.lower() == 'birdnet':
        embed_fn_config.model_key = 'birdnet'
        model_config.window_size_s = 3.0
        model_config.hop_size_s = 3.0
        model_config.sample_rate = 48000
        model_config.model_path = models_folder_path + 'BirdNET-model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
        model_config.num_tflite_threads = 16
        model_config.class_list_name = 'birdnet_v2_1'

    # Name error
    else:
        print(f"No model corresponding to the input {model_name}, possible choices: surfperch, perch, birdnet")
        return [], []
    
    embed_fn_config.write_embeddings = True
    embed_fn_config.write_logits = False
    embed_fn_config.write_separated_audio = False
    embed_fn_config.write_raw_audio = False
    config.embed_fn_config = embed_fn_config
    embed_fn_config.model_config = model_config
    # Number of parent directories to include in the filename. This allows us to
    # process raw audio that lives in multiple directories.
    config.embed_fn_config.file_id_depth = 1
    
    # Loading model and testing on zeros
    # The embed_fn object is a wrapper over the model.
    embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
    print(f'\n\nLoading {model_name}...')
    embed_fn.setup()

    print('Test-run of model...')
    z = np.zeros([int(model_config.sample_rate * model_config.window_size_s)])
    embed_fn.embedding_model.embed(z)
    print('\nSetup complete!')
        
    return embed_fn, config



# CSV to Raven format
def eval_to_Raven(prediction_file, wind_dur, keep_only_call=True, freq_bands=[0, 16000]):
    """ Convert a detections csv file into several files, one file per audio files in the Raven Pro format
        return None, save all files in a subdirectory "Raven file" where the prediction file is.
    Args:
     - prediction_file: path to the file containing the detections
     - wind_dur: size of the detection window to calculate the end of the annotations
     - keep_only_call: set to false to keep the annotation of noise class as well
     - freq_bands: define the frequency range of the annotations in the raven annotation files
     Output:
     - None
      """
    
    predicted_events = pd.read_csv(prediction_file, sep=',')
    if keep_only_call:
        predicted_events = predicted_events.loc[predicted_events[' label'] != ' Unknown']

    source_dir, filename = os.path.split(prediction_file)
    raven_dir = os.path.join(source_dir, 'Raven file/')
    if not os.path.exists(raven_dir):
        os.makedirs(raven_dir)

    list_predicted_files = predicted_events.groupby(['filename'])

    for _idx, predicted_file  in list_predicted_files:

        raven_file_df = pd.DataFrame({'Begin Time (s)': predicted_file[' timestamp_s'], 'End Time (s)': predicted_file[' timestamp_s']+wind_dur})
        raven_file_df.index.name = 'Selection'
        raven_file_df['View'] = 'Spectrogram 1'
        raven_file_df['Channel'] = 1
        raven_file_df['Low Freq (Hz)'] = freq_bands[0]
        raven_file_df['High Freq (Hz)'] = freq_bands[1]
        raven_file_df['Type'] = predicted_file[' label']
        # raven_file_df['Logit'] = predicted_file[' logit']
        raven_file_df['Confidence Score'] = predicted_file[' confidence']*100 # Confidence score to percentage


        if raven_file_df.index[0] == 0: # Raven does not support index starting at 0
            raven_file_df.index += 1

        dest_filepath = raven_dir + os.path.basename(predicted_file['filename'].iloc[0])[:-4] +'.txt'
        try:
            raven_file_df.to_csv(dest_filepath, sep='\t')
            # print('Raven annotation saved at:', dest_filepath)
        except Exception as e:
            print("Could not save the annotations in a csv file:", e)

    print("Raven files saved at:", raven_dir)