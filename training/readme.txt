To perform training you will need to download and unpack training data: https://www.dropbox.com/s/vsxhs6qm3mk2dlq/data.zip?dl=0

Also to perform training using Tensorflow 1.3 you will need Tensorflow 1.3 installed and running and the following things:
1. Checkout "tensorflow/models" branch "r1.5"
2. Apply the following patches:

ubuntu@ip-172-31-43-173:~/work/tf/models/research/object_detection/builders$ git diff
diff --git a/research/object_detection/builders/model_builder.py b/research/object_detection/builders/model_builder.py
index 5467a91..255e0a9 100644
--- a/research/object_detection/builders/model_builder.py
+++ b/research/object_detection/builders/model_builder.py
@@ -105,7 +105,7 @@ def _build_ssd_feature_extractor(feature_extractor_config, is_training,
   depth_multiplier = feature_extractor_config.depth_multiplier
   min_depth = feature_extractor_config.min_depth
   pad_to_multiple = feature_extractor_config.pad_to_multiple
-  batch_norm_trainable = feature_extractor_config.batch_norm_trainable
+  batch_norm_trainable = False # feature_extractor_config.batch_norm_trainable
   conv_hyperparams = hyperparams_builder.build(
       feature_extractor_config.conv_hyperparams, is_training)

diff --git a/research/object_detection/export_inference_graph.py b/research/object_detection/export_inference_graph.py
index 279d1d1..6df6703 100644
--- a/research/object_detection/export_inference_graph.py
+++ b/research/object_detection/export_inference_graph.py
@@ -93,9 +93,9 @@ flags.DEFINE_string('trained_checkpoint_prefix', None,
                     'path/to/model.ckpt')
 flags.DEFINE_string('output_directory', None, 'Path to write outputs.')

-tf.app.flags.mark_flag_as_required('pipeline_config_path')
-tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
-tf.app.flags.mark_flag_as_required('output_directory')
+#tf.app.flags.mark_flag_as_required('pipeline_config_path')
+#tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
+#tf.app.flags.mark_flag_as_required('output_directory')
 FLAGS = flags.FLAGS


diff --git a/research/object_detection/exporter.py b/research/object_detection/exporter.py
index ef8fe19..198b188 100644
--- a/research/object_detection/exporter.py
+++ b/research/object_detection/exporter.py
@@ -69,7 +69,7 @@ def freeze_graph_with_def_protos(
     if optimize_graph:
       logging.info('Graph Rewriter optimizations enabled')
       rewrite_options = rewriter_config_pb2.RewriterConfig(
-          layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
+          optimize_tensor_layout=True) #layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
       rewrite_options.optimizers.append('pruning')
       rewrite_options.optimizers.append('constfold')
       rewrite_options.optimizers.append('layout')
