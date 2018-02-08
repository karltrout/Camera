from unittest import TestCase
from time import time
import cv2
import os

from align_dlib import AlignDlib

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

from classifier import  Classifier

class TestImageClass(TestCase):
    def test_decode(self):

        classifier = \
            Classifier(model_path='/Users/karltrout/Documents/DockerHome/etc/20170511-185253/20170511-185253.pb',
                       classifier_path='/Users/karltrout/Documents/DockerHome/output/classifier.pkl')

        cap = cv2.VideoCapture(0)
        # file_contents = \
        #     tf.read_file('/Users/karltrout/Documents/DockerHome/output/intermediate/'
        #                  'Karl_Trout/Karl_Trout_3005.jpg')
        #   #               'George_Clooney/George_Clooney_0001.jpg')
        # file_contents = tf.image.decode_image(file_contents,channels=3)

        sTime = time()
        count = 30;
        while True:
            captured, frame = cap.read()
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            TestCase.assertIs(self, captured, True, msg="Did not grab a frame.")
            bb, frame = _align_image(frame, 160)
            if frame is not None:
                #count = count + 1
                fileName = 'Karl_Trout_{}.jpg'.format(count)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #cv2.imwrite(fileName, frame)
                break

        print('Completed Capture of image in {} seconds'.format(time() - sTime))
        classifier.classify_image(image=frame)


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    #if aligned is not None:
        #aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return bb, aligned
