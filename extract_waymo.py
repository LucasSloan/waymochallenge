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


label_type_map = {
    open_label.Label.TYPE_UNKNOWN: 'unknown',
    open_label.Label.TYPE_VEHICLE: 'vehicle',
    open_label.Label.TYPE_PEDESTRIAN: 'pedestrian',
    open_label.Label.TYPE_SIGN: 'sign',
    open_label.Label.TYPE_CYCLIST: 'cyclist',
}

camera_name_map = {
    open_dataset.CameraName.UNKNOWN: 'unknown',
    open_dataset.CameraName.FRONT: 'front',
    open_dataset.CameraName.FRONT_LEFT: 'front_left',
    open_dataset.CameraName.FRONT_RIGHT: 'front_right',
    open_dataset.CameraName.SIDE_LEFT: 'side_left',
    open_dataset.CameraName.SIDE_RIGHT: 'side_right',
}


GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.

def filename_to_int(filename):
    """Convert a string to a integer."""
    # Warning: this function is highly specific to pascal filename!!
    # Given filename like '2008_000002', we cannot use id 2008000002 because our
    # code internally will convert the int value to float32 and back to int, which
    # would cause value mismatch int(float32(2008000002)) != int(2008000002).
    # COCO needs int values, here we just use a incremental global_id, but
    # users should customize their own ways to generate filename.
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID


def build_example(sequence_name, camera_image, camera_labels, calibrations, ann_json_dict=None):
    for calibration in calibrations:
        if calibration.name != camera_image.name:
            continue

        width = calibration.width
        height = calibration.height

    key = hashlib.sha256(camera_image.image).hexdigest()

    filename = "{}_{}_{}".format(sequence_name, camera_name_map[camera_image.name], camera_image.pose_timestamp)
    image_id = filename_to_int(filename)
    if ann_json_dict:
        image = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id,
        }
        ann_json_dict['images'].append(image)

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

            if ann_json_dict:
                abs_xmin = int(label.box.center_x - 0.5 * label.box.length)
                abs_ymin = int(label.box.center_y - 0.5 * label.box.width)
                abs_xmax = int(label.box.center_x + 0.5 * label.box.length)
                abs_ymax = int(label.box.center_y + 0.5 * label.box.width)
                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin
                ann = {
                    'area': abs_width * abs_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                    'category_id': int(label.type),
                    'id': get_ann_id(),
                    'ignore': 0,
                    'segmentation': [],
                }
                ann_json_dict['annotations'].append(ann)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            str(image_id).encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[camera_image.image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
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
    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }

    for class_id, class_name in label_type_map.items():
        cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
        ann_json_dict['categories'].append(cls)

    num_examples = 0
    i = 0
    for tfrecord in tf.data.Dataset.list_files(FLAGS.data_dir + '/*'):
        i += 1

        examples = []
        dataset = tf.data.TFRecordDataset(tfrecord)
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,
            range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
                frame)

            for image in frame.images:
                tf_example = build_example(frame.context.name,
                    image, frame.camera_labels, frame.context.camera_calibrations, ann_json_dict=ann_json_dict)
                examples.append(tf_example.SerializeToString())

        random.shuffle(examples)
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '-%05d.tfrecord' % (i))
        for example in examples:
                writer.write(example)
                num_examples += 1
        writer.close()
        print("processed {} tfrecords".format(i))

    json_file_path = os.path.join(
        os.path.dirname(FLAGS.output_path),
        'json_' + os.path.basename(FLAGS.output_path) + '.json')
    with open(json_file_path, 'w') as f:
        json.dump(ann_json_dict, f)

    print("wrote {} examples in {} tfrecords".format(num_examples, i))

if __name__ == '__main__':
    main()
