import tensorflow as tf
from google.protobuf.json_format import MessageToJson

for example in tf.python_io.tf_record_iterator("/mnt/Bulk/Waymo/comparison/train-00001.tfrecord"):
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    print(jsonMessage)
