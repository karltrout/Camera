import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)


class ImageClass:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.data)

    def decode(self) -> object:
        float_caster = tf.cast(self.data, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resize = tf.image.resize_bilinear(dims_expander, [160, 160])
        normalized = tf.divide(tf.subtract(resize, [0]), [255])
        sess = tf.Session()
        result = sess.run(normalized)
        return tf.constant(result)


class Classifier:
    def __init__(self, model_path, classifier_path):
        """
        Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
         a classifier outputted to :param output_path

        :param model_path: Path to protobuf graph file for facenet model
        :param classifier_path: Path to write pickled classifier
        """
        start_time = time.time()

        self.classifier_filename = classifier_path
        self.load_model(model_filepath=model_path)
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        with open(self.classifier_filename, 'rb') as f:
            self.model, self.class_names = pickle.load(f)

        print('Completed init in {} seconds'.format(time.time() - start_time))

    def classify_image(self, image):
        sTime = time.time()
        arg = tf.convert_to_tensor(image, dtype=tf.float32)

        image = tf.image.per_image_standardization(arg)
        new_image = self.session.run([image])
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init_op)

        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

        emb_array = \
            self.session.run(embedding_layer,
                             feed_dict={images_placeholder: new_image, phase_train_placeholder: False})



        predictions = self.model.predict_proba(emb_array, )
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        name='Unknown'
        probability= .5

        for i in range(len(best_class_indices)):
        #if (best_class_probabilities[i] > .5):
            print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))
            if (best_class_probabilities[i] > probability):
                name = self.class_names[best_class_indices[i]]
                probability = best_class_probabilities[i]
        #else:
         #   print('{} Unidentified: {}'.format(i, best_class_probabilities[i]))

        print('Completed ID process in {} seconds'.format(time.time() - sTime))

        return name, probability

        #predictions = np.squeeze(predictions)

            #best_class_indices = np.argmax(predictions)
            #print(best_class_indices)
            #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            #for i in range(len(best_class_indices)):
            #    print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))

            #print(predictions)

            #embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #coord = tf.train.Coordinator()

            #threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            #emb_array, label_array = self.create_embeddings(embedding_layer, images, labels,
            #                                            images_placeholder, phase_train_placeholder, sess)
            #coord.request_stop()
            #coord.join(threads=threads)
            #logger.info('Created {} embeddings'.format(len(emb_array)))

            #self.evaluate_classifier(emb_array)

    @staticmethod
    def load_model(model_filepath):
        """
        Load frozen protobuf graph
        :param model_filepath: Path to protobuf graph
        :type model_filepath: str
        """
        model_exp = os.path.expanduser(model_filepath)
        if os.path.isfile(model_exp):
            logging.info('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            logger.error('Missing model file. Exiting')
            sys.exit(-1)

    def create_embeddings(self, embedding_layer, images, labels, images_placeholder, phase_train_placeholder, sess):
        """
        Uses model to generate embeddings from :param images.
        :param embedding_layer:
        :param images:
        :param labels:
        :param images_placeholder:
        :param phase_train_placeholder:
        :param sess:
        :return: (tuple): image embeddings and labels
        """
        emb_array = None
        label_array = None

        try:
            i = 0
            while True:
                batch_images, batch_labels = sess.run([images, labels])
                logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
                emb = sess.run(embedding_layer,
                               feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

                emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
                label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
                i += 1

        except tf.errors.OutOfRangeError:
            pass

        return emb_array, label_array

    def evaluate_classifier(self, emb_array):
        """

        :param emb_array:
        """
        logger.info('Evaluating classifier on {} images'.format(len(emb_array)))

        predictions = self.model.predict_proba(emb_array, )
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        help='Input path of data to train on', default=128)
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                        help='Number of threads to utilize for queue')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                        help='Path to output trained classifier model')
    parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', default=0.7,
                        help='Ratio to split train/test dataset')
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=10,
                        dest='min_images_per_class', help='Minimum number of images per class')
    parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path',
                        help='Path to output trained classifier model')
    parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')

    args = parser.parse_args()

#   main(input_directory=args.input_dir, model_path=args.model_path, classifier_output_path=args.classifier_path,
#batch_size=args.batch_size, num_threads=args.num_threads, num_epochs=args.num_epochs,
#min_images_per_labels=args.min_images_per_class, split_ratio=args.split_ratio, is_train=args.is_train)
