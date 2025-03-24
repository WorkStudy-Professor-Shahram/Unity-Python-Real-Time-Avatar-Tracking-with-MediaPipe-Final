import mediapipe as mp
import cv2
import time
import global_vars
import struct
from clientUDP import ClientUDP

class BodyTracker:
    def __init__(self):
        self.cap = None
        self.client = None
        self.pipe = None
        self.data = ""

    def setup_capture(self):
        self.cap = cv2.VideoCapture(global_vars.CAM_INDEX)
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_vars.HEIGHT)
        print("Opened Capture @ %s fps" % str(self.cap.get(cv2.CAP_PROP_FPS)))

    def run(self):
        self.setup_capture()
        self.setup_comms()

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.5, model_complexity=global_vars.MODEL_COMPLEXITY, static_image_mode=False, enable_segmentation=True) as pose:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Image transformations
                frame = cv2.flip(frame, 1)
                frame.flags.writeable = global_vars.DEBUG
                
                # Process the image
                results = pose.process(frame)

                # Render results
                if global_vars.DEBUG:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
                    cv2.imshow('Body Tracking', frame)
                    if cv2.waitKey(3) & 0xFF == 27:
                        break

                # Prepare and send data
                self.prepare_data(results)
                self.send_data(self.data)

        self.cleanup()

    def prepare_data(self, results):
        self.data = ""
        if results.pose_world_landmarks:
            for i in range(0, 33):
                landmark = results.pose_world_landmarks.landmark[i]
                self.data += "{}|{}|{}|{}\n".format(i, landmark.x, landmark.y, landmark.z)

    def setup_comms(self):
        if not global_vars.USE_LEGACY_PIPES:
            self.client = ClientUDP(global_vars.HOST, global_vars.PORT)
            self.client.start()
        else:
            print("Using Pipes for interprocess communication (not supported on OSX or Linux).")

    def send_data(self, message):
        if not global_vars.USE_LEGACY_PIPES:
            self.client.sendMessage(message)
        else:
            # this is for MacOs. 
            print("Using Pipes for interprocess communication (not supported on OSX or Linux).")
            pass  

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.pipe:
            self.pipe.close()

if __name__ == "__main__":
    tracker = BodyTracker()
    tracker.run()