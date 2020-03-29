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

tf.enable_eager_execution()


label_type_map = {
    open_label.Label.TYPE_UNKNOWN: 'unknown',
    open_label.Label.TYPE_VEHICLE: 'vehicle',
    open_label.Label.TYPE_PEDESTRIAN: 'pedestrian',
    open_label.Label.TYPE_SIGN: 'sign',
    open_label.Label.TYPE_CYCLIST: 'cyclist',
}

def build_example(camera_image, camera_labels, calibrations):
    for calibration in calibrations:
        if calibration.name != camera_image.name:
            continue

        width = calibration.width
        height = calibration.height


    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    for cl in camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if cl.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in cl.labels:
            # Draw the object bounding box.
            xmin = float(label.box.center_x - 0.5 * label.box.length) / width
            ymin = float(label.box.center_y - 0.5 * label.box.width) / height

            xmax = float(label.box.center_x + 0.5 * label.box.length) / width
            ymax = float(label.box.center_y + 0.5 * label.box.width) / height

            class_text = label_type_map[label.type]

            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            classes.append(int(label.type))
            classes_text.append(class_text.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        # 'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[camera_image.image])),
        # 'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def main():
    writer = tf.io.TFRecordWriter('./data/waymo_validation.tfrecord')
    # FILENAME = '/mnt/Bulk/Waymo/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    dataset = tf.data.Dataset.list_files('/mnt/Bulk/Waymo/validation/*')
    dataset = tf.data.TFRecordDataset(dataset)
    i = 1
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections,
        range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)

        for index, image in enumerate(frame.images):
            if image.name != open_dataset.CameraName.FRONT:
                continue

            tf_example = build_example(image, frame.camera_labels, frame.context.camera_calibrations)
            writer.write(tf_example.SerializeToString())

        if i % 100 == 0:
            print("processed {} frames".format(i))
        i += 1

    writer.close()


if __name__ == '__main__':
    main()
