# imgclassifywrapper.py

import os.path
import numpy as np
import tensorflow as tf
from os import path




from node_lookup import NodeLookup

class Imgclassifywrapper():

    def __init__(self, model_dir='model', model_file='classify_image_graph_def.pb'):

        self.model_name = os.path.join(model_dir, model_file)


    def classify(self, image):


        self.run_inference_on_image(image)


    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.

        print('\n model_name : ' + self.model_name)
        with tf.gfile.FastGFile(self.model_name, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image(self, image):
        """Runs inference on an image.

        Args:
          image: Image file name.

        Returns:
          Nothing
        """
        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = tf.gfile.FastGFile(image, 'rb').read()

        # Creates graph from saved GraphDef.
        self.create_graph()

        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()

            #top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
            top_k = predictions.argsort()[-3:][::-1]

            toppredictions = []
            index = 0
            top_score = 0.0
            top_index = 0
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                toppredictions.append(human_string)
                score = predictions[node_id]

                if (score > top_score):
                    top_score = score
                    top_index = index

                print('\n%s (score = %.5f)' % (human_string, score))
                index = index + 1
            # print("************************")
            print ("\n\n Recommended " + toppredictions[top_index] )

            return toppredictions[top_index]


# end class Imgclassifywrapper
###############################################################################




class TestImgclassifywrapper():

    def test(self, image):

        if path.exists(image) ==  False :
            print ('Error file not found!')
            return 'Error file not found!'
        img_cls_wrapper = Imgclassifywrapper()

        print ( '** Result :  ' + img_cls_wrapper.run_inference_on_image(image))



# end class TestImgclassifywrapper
###############################################################################


test_obj = TestImgclassifywrapper()
test_obj.test("phone.jpg")
