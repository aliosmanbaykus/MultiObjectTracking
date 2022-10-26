import cv2
import numpy as np

OBJECT_TRACKERS = {
    'csrt' : cv2.legacy.TrackerCSRT_create,
    'mosse' : cv2.legacy.TrackerMOSSE_create,
    'kcf' : cv2.legacy.TrackerKCF_create,
    'medianflow' : cv2.legacy.TrackerMedianFlow_create,
    'nil' : cv2.legacy.TrackerMIL_create,
    'tld' : cv2.legacy.TrackerTLD_create,
    'boosting' : cv2.legacy.TrackerBoosting_create
}

class ObjectTracking:
    def __init__(self, path):
        self.path = str(path)

                
        self.trackers = cv2.legacy.MultiTracker_create()
        self.cap = cv2.VideoCapture(self.path)
   
        while True:
            frame = self.cap.read()[1]

            if frame is None:
                break

            frame = cv2.resize(frame, (750, 550))

            success, boxes = self.trackers.update(frame)

            for box in boxes:
                x, y, w, h = [int(c) for c in box]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)

            cv2.imshow('Tracking', frame)
            k = cv2.waitKey(30)

            if k == ord('s'):
                roi = cv2.selectROI('Tracking', frame)
                tracker = OBJECT_TRACKERS['kcf']()
                
                self.trackers.add(tracker, frame, roi)
    

        self.cap.release()
        cv2.destroyAllWindows()


