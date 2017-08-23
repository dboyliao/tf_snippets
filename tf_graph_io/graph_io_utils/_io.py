# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util

__all__ = ["chkp_to_freeze_graph", "import_pb_file"]

def chkp_to_freeze_graph(chkp_path,
                         output_node_names,
                         clear_devices=False,
                         model_file_name=None):
    """Convert Checkpoints Meta File to Serialized Model File

    params
    ======
    - chkp_path: string, path prefix of the checkpoint files
    - output_node_names: list of names of nodes to be output, each name in the list is
        the name string of the output node
    - clear_devices (Optional): Boolean. Clear devices of the nodes if True, keep otherwise (default: False)
    - model_file_name (Optional): string. If it's given, write the serialization data to given file (default: None)
    """
    # restore graph
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(chkp_path+".meta",
                                           clear_devices=clear_devices)
        graph_def = graph.as_graph_def()

    # restore session
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, chkp_path)
        # get the freezed graph def
        freeze_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     graph_def,
                                                                     output_node_names)
    # save the model to disk
    if model_file_name:
        with tf.gfile.GFile(model_file_name, "wb") as fid:
            fid.write(freeze_graph_def.SerializeToString())
    return freeze_graph_def


def import_pb_file(pb_file, **kwargs):
    """Load Tensorflow Protobuf Model File

    params
    ======
    - pb_file: string, the protobuf file to be loaded
    - **kwargs: keyword arguments for tf.import_graph_def

    return
    ======
    - graph: tf.Graph, loaded graph
    """
    graph = tf.Graph()
    graph_def = graph.as_graph_def()
    with tf.gfile.GFile(pb_file, "rb") as fid:
        graph_def.ParseFromString(fid.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, **kwargs)
    return graph
