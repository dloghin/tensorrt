import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  # start = time.time_ns()
  start = time.time()
  output_dict = model(input_tensor)
  # end = time.time_ns()
  end = time.time()
  # print("Inference time: {}".format((end-start)/(10**9)))
  print("Inference time: {} s".format(end-start))

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # return output_dict

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict


def main(args):
  if len(args) < 4:
    print("Usage: {} <model_path> <label_path> <image_path>")
    exit(1)

  model_path = args[1]
  labels_path = args[2]
  image_path = args[3]

  model = tf.saved_model.load(model_path)
  model = model.signatures['serving_default']

  image_np = np.array(Image.open(image_path))

  category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

  output_dict = run_inference_for_single_image(model, image_np)

  print(output_dict)

  result = vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  result = Image.fromarray((result * 255).astype(np.uint8))
  result.save("output.jpg")

  # display(Image.fromarray(image_np))


if __name__ == '__main__':
  main(sys.argv)
