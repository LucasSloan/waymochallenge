from __future__ import print_function
from waymo_open_dataset import label_pb2 as open_label
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils

import os
import tensorflow as tf
import math
import numpy as np
import itertools
import hashlib
import json
import random

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Directory to read input tfrecords from.')
flags.DEFINE_string('output_path', '', 'Path to write the tensorflow dectection api tfrecords to.')
FLAGS = flags.FLAGS


tf.enable_eager_execution()


camera_name_map = {
    open_dataset.CameraName.UNKNOWN: 'unknown',
    open_dataset.CameraName.FRONT: 'front',
    open_dataset.CameraName.FRONT_LEFT: 'front_left',
    open_dataset.CameraName.FRONT_RIGHT: 'front_right',
    open_dataset.CameraName.SIDE_LEFT: 'side_left',
    open_dataset.CameraName.SIDE_RIGHT: 'side_right',
}


def build_example(sequence_name, camera_image, camera_labels, calibrations):
    for calibration in calibrations:
        if calibration.name != camera_image.name:
            continue

        width = calibration.width
        height = calibration.height

    filename = "{}_{}_{}".format(sequence_name, camera_name_map[camera_image.name], camera_image.pose_timestamp)
    with open(FLAGS.output_path + filename + ".txt", "w") as f:
        for cl in camera_labels:
            # Ignore camera labels that do not correspond to this camera.
            if cl.name != camera_image.name:
                continue

            # Iterate over the individual labels.
            for label in cl.labels[:100]:
                class_label = int(label.type) - 1

                xcenter = label.box.center_x / width
                ycenter = label.box.center_y / height

                width = label.box.length / width
                height = label.box.width / height

                f.write("{} {} {} {} {}\n".format(class_label, xcenter, ycenter, width, height))

    with open(FLAGS.output_path + filename + ".jpg", "wb") as f:
        f.write(camera_image.image)



def main():
    i = 0
    for tfrecord in tf.data.Dataset.list_files(FLAGS.data_dir + '/*.tfrecord'):
        i += 1

        dataset = tf.data.TFRecordDataset(tfrecord)
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,
            range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
                frame)

            for image in frame.images:
                if image.name != open_dataset.CameraName.FRONT:
                    continue
                build_example(frame.context.name,
                    image, frame.camera_labels, frame.context.camera_calibrations)

if __name__ == '__main__':
    main()
