# %% md

## DeepExplain - Tensorflow example
### MNIST with a 2-layers MLP

# %%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os

sys.path.insert(0, os.path.abspath('..'))

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import tqdm
import glob
import cv2
import numpy as np
import csv


# %%


model_path = "saved_models/1579825617"
base_dir = '/Users/sal/Documents/phenomenal-face'

model_path = os.path.join(base_dir, model_path)

# %%

# extract input tensor by name


# %%

# get the images
input_files = glob.glob(os.path.join(base_dir, 'face-evaluation/*.jpg'))
image_tensor_stack = {}
input_image_size = (299, 299)

sess_img = tf.Session()
for idx, image_filename in enumerate(tqdm(input_files)):
    i_n = image_filename.split('/')[-1]
    decoded_image = (cv2.imread(image_filename) / 255) * 2 - 1
    result_img = cv2.resize(decoded_image, input_image_size, interpolation=cv2.INTER_AREA)
    result_img = np.expand_dims(result_img, 0)

    # result_image = tf.squeeze(resized_image, 0)
    image_tensor_stack[i_n] = result_img


# get the labels
label_dict = {}
label_types = ['heights', 'weights', 'genders', 'ages']


# filter the strings into float format
def filter_fn(d):
    for i in range(len(d)):
        if d[i] != 'None':
            if d[i] in ['male', 'female']:
                d[i] = np.array([[0,1]] if (d[i] == 'male') else [[1,0]], dtype=np.int32)

            else:
                if 'jpg' not in d[i]:
                    d[i] = np.array([float(d[i])], dtype=np.float32)
        else:
            d[i] = float(0)

    return d


with open(os.path.join(base_dir, 'face-evaluation/labels.csv'), newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = list(map(filter_fn, data))

for d in data:
    label_dict[d[0]] = {label_types[idx]: val for idx, val in enumerate(d[1:])}

# %%

image_tensor_stack.keys()

# %%


# %%

from deepexplain.tensorflow import DeepExplain

tf.reset_default_graph()
result_dict_output = {}
sess = tf.Session()
limit  = 2

with DeepExplain(session=sess) as de:
    tf.saved_model.loader.load(de.session, [tf.saved_model.tag_constants.SERVING], model_path)
    # sess.run(init_op)
    X = de.graph.get_tensor_by_name('serving-input-placeholder:0')

    Y = [de.graph.get_tensor_by_name(f'custom_layers/{key}:0') for key in label_types if key == 'ages']
    Y = tf.math.add_n(Y)
    for i, image_filename in enumerate(tqdm(input_files[:limit])):
        i_n = image_filename.split('/')[-1]
        labels = label_dict[i_n]
        xi = image_tensor_stack[i_n]
        print(xi.shape)
        attributions = {'DeepLIFT (Rescale)': de.explain('intgrad', Y, X, xi )}
        '''
        attributions = {
                            # Gradient-based
                            'Saliency maps':        de.explain('saliency', Y, X, xi),
                            'Gradient * Input':     de.explain('grad*input', Y, X, xi),
                            'Integrated Gradients': de.explain('intgrad', Y, X, xi),
                            'Epsilon-LRP':          de.explain('elrp', Y, X, xi),
                            'DeepLIFT (Rescale)':   de.explain('deeplift', Y, X, xi),
                            #Perturbation-based
                            '_Occlusion [1x1]':      de.explain('occlusion', Y, X, xi),
                            '_Occlusion [3x3]':      de.explain('occlusion', Y, X, xi, window_shape=(3,))
                        }
        '''
        result_dict_output[image_filename] = {'grad': attributions, 'input': [xi]*2, 'output': None, 'label': labels}

# %%


A = 1