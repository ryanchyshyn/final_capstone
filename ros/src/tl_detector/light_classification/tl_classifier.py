import rospy
import numpy as np
import tensorflow as tf
from PIL import Image
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.counter = 0

        ssd_inception_model = 'frozen_inference_graph.pb'
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(ssd_inception_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

            # Definite input and output Tensors for detection_graph
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_np_expanded = np.expand_dims(image, axis=0)

        # save image
        #img = Image.fromarray(image)
        #img.save('image' + str(self.counter) + ".png", 'png')
        #self.counter = self.counter + 1

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(classes) < 1:
            return TrafficLight.UNKNOWN

        # get higher score
        higher_score_index = scores.argmax(axis=0)
        return self.class_to_traffic_light(classes[higher_score_index])

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    @staticmethod
    def class_to_traffic_light(obj_class):
        if obj_class == 1:
            return TrafficLight.GREEN
        elif obj_class == 2:
            return TrafficLight.RED
        elif obj_class == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN
