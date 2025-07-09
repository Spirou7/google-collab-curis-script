import tensorflow as tf

# TensorFlow 2.19.0 Compatibility:
# - tf.tpu.experimental.TPUSystemMetadata is still supported in TF 2.19.0
# - tf.train.ClusterSpec continues to be available in TF 2.19.0
# - tf.distribute.cluster_resolver.TPUClusterResolver remains stable

class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
  """LocalTPUClusterResolver."""

  def __init__(self):
    self._tpu = ""
    self.task_type = "worker"
    self.task_id = 0

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    return None

  def cluster_spec(self):
    return tf.train.ClusterSpec({})

  def get_tpu_system_metadata(self):
    # TensorFlow 2.19.0: tf.tpu.experimental.TPUSystemMetadata still supported
    return tf.tpu.experimental.TPUSystemMetadata(
        num_cores=8,
        num_hosts=1,
        num_of_cores_per_host=8,
        topology=None,
        devices=tf.config.list_logical_devices())

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    return {"TPU": 8}

