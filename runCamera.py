import cv2
import os
import threading

from classifier import Classifier
from queue import Queue

#classifier = \
 #   Classifier(model_path='/Users/karltrout/Documents/DockerHome/etc/20170511-185253/20170511-185253.pb',
  #             classifier_path='/Users/karltrout/Documents/DockerHome/output/classifier.pkl')


from align_dlib import AlignDlib
align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))
bBox = None
threadLock = threading.Lock()
q = Queue(maxsize=1)

def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return bb, aligned


def classifier_thread_worker(q):
    """thread worker function"""
    classifier = \
        Classifier(model_path='/Users/karltrout/Documents/DockerHome/etc/20170511-185253/20170511-185253.pb',
                   classifier_path='/Users/karltrout/Documents/DockerHome/output/classifier.pkl')
    while True:
        try:
            image_in = q.get_nowait()
            if image_in is None:
                break
            classifier.classify_image(image=image_in)
            q.task_done()
        except:
            pass


cap = cv2.VideoCapture(0)
crop_dim = 180

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

t = threading.Thread(target=classifier_thread_worker, args=(q,))
t.start()

while cap.isOpened:
    ret, frame = cap.read()

    if ret is True:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        smallFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

        if image is not None:
            bb, aligned_image = _align_image(image, crop_dim)

            if bb is not None:
                    bBox = bb
            else:
                bBox = None

            if bBox is not None :
                cv2.rectangle(frame,
                              pt1=(bBox.left()*2,
                                   bBox.top()*2),
                              pt2=(bBox.right()*2,
                                   bBox.bottom()*2),
                              color=(0, 255, 0),
                              thickness=1)

            cv2.imshow('frame', frame)
            if aligned_image is not None :
                # do awesome ML Look up stuff here.
                #classifier_thread_worker(aligned_image)
                try:
                    q.put_nowait(aligned_image)
                except:
                    pass

        else:
            raise IOError('Error buffering image.')
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            q.put(None)
            t.join()
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
