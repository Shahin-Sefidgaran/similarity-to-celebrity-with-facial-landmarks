#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os.path as osp
import sys
import time
import datetime

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from .ie_module import InferenceContext
from .landmarks_detector import LandmarksDetector
from .face_detector import FaceDetector
from .faces_database import FacesDatabase
from .face_identifier import FaceIdentifier

import settings

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']
MATCH_ALGO = ['HUNGARIAN', 'MIN_DIST']

input_image = '0'
output_image = ""
no_show = 0
tl = False
timelapse = False
cw = 0
crop_width = 0
ch = 0
crop_height = 0
match_algo = 'HUNGARIAN'
fg = "celebrities_gallary"
run_detector = False
m_fd = "./face-detection-retail-0004.xml"
m_lm = "./landmarks-regression-retail-0009.xml"
# m_lm = 'facial-landmarks-35-adas-0002.xml'
m_reid = "./face-reidentification-retail-0095.xml"
fd_iw = 0
fd_input_width = 0
fd_ih = 0
fd_input_height = 0
d_fd = 'CPU'
d_lm = 'CPU'
d_reid = 'CPU'
cpu_lib = ""
gpu_lib = ""
verbose = False
pc = False
perf_stats = False
t_fd = 0.7
t_id = 0.3
exp_r_fd = 1.15
allow_grow = False
new_face_name = ""
possible_labels = []

class FrameProcessor:
    QUEUE_SIZE = 32

    def __init__(self):
        used_devices = set([d_fd, d_lm, d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, cpu_lib, gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if perf_stats else "NO"})

        log.info("Loading models")
        face_detector_net = self.load_model(m_fd)
        
        assert (fd_input_height and fd_input_width) or \
               (fd_input_height==0 and fd_input_width==0), \
            "Both -fd_iw and -fd_ih parameters should be specified for reshape"
        
        if fd_input_height and fd_input_width :
            face_detector_net.reshape({"data": [1, 3, fd_input_height,fd_input_width]})
        landmarks_net = self.load_model(m_lm)
        face_reid_net = self.load_model(m_reid)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=t_fd,
                                          roi_scale_factor=exp_r_fd)

        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
                                              match_threshold=t_id,
                                              match_algo = match_algo)

        self.face_detector.deploy(d_fd, context)
        self.landmarks_detector.deploy(d_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(d_reid, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        log.info("Building faces database using images from '%s'" % (fg))
        self.faces_database = FacesDatabase(fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if run_detector else None, no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

        self.allow_grow = allow_grow and not no_show

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.face_identifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns, face_identities2, face_identities3 = self.face_identifier.get_matches()

        if allow_grow:
            crop = orig_image[int(rois[0].position[1]):int(rois[0].position[1]+rois[0].size[1]), int(rois[0].position[0]):int(rois[0].position[0]+rois[0].size[0])]
            #name = self.faces_database.ask_to_save(crop)
            name = new_face_name
            if name:
                id = self.faces_database.dump_faces(crop, face_identities[0].descriptor, name)
                face_identities[0].id = id
        outputs = [rois, landmarks, face_identities, face_identities2, face_identities3]

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}


    def __init__(self,):
        self.frame_processor = FrameProcessor()
        self.display = not no_show
        self.print_perf_stats = perf_stats

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if crop_width and crop_height:
            self.input_crop = np.array((crop_width, crop_height))

        self.frame_timeout = 0 if timelapse else 1

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=2, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, roi, identity, identity2, identity3):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)
        label2 = self.frame_processor \
            .face_identifier.get_identity_label(identity2.id)
        label3 = self.frame_processor \
            .face_identifier.get_identity_label(identity3.id)
        # Draw face ROI border
        #cv2.rectangle(frame,
        #              tuple(roi.position), tuple(roi.position + roi.size),
        #              (0, 220, 0), 2)

        # Draw identity label
        text_scale = 1.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = label
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        #self.draw_text_with_background(frame, text,
        #                               roi.position - line_height * 0.5,
        #                               font, scale=text_scale)
        text2 = label2
        text2 += ' %.2f%%' % (100.0 * (1 - identity2.distance))

        text3 = label3
        text3 += ' %.2f%%' % (100.0 * (1 - identity3.distance))

        global possible_labels
        possible_labels.append(text)
        possible_labels.append(text2)
        possible_labels.append(text3)

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]
        # keypoints = [landmarks.left_eye_right,
        #     landmarks.left_eye_left,
        #     landmarks.right_eye_left,
        #     landmarks.right_eye_right,
        #     landmarks.nose_tip,
        #     landmarks.nose_nasal,
        #     landmarks.nose_left,
        #     landmarks.nose_right,
        #     landmarks.mouth_left,
        #     landmarks.mouth_right,
        #     landmarks.mouth_top,
        #     landmarks.mouth_down,
        #     landmarks.left_eyebrow_left,
        #     landmarks.left_eyebrow_middle,
        #     landmarks.left_eyebrow_right,
        #     landmarks.right_eyebrow_left,
        #     landmarks.right_eyebrow_middle,
        #     landmarks.right_eyebrow_right,
            
        #     # Face Contour
        #     landmarks.face_contour_18,
        #     landmarks.face_contour_19,
        #     landmarks.face_contour_20,
        #     landmarks.face_contour_21,
        #     landmarks.face_contour_22,
        #     landmarks.face_contour_23,
        #     landmarks.face_contour_24,
        #     landmarks.face_contour_25,
        #     landmarks.face_contour_26, # Chin Center
        #     landmarks.face_contour_27,
        #     landmarks.face_contour_28,
        #     landmarks.face_contour_29,
        #     landmarks.face_contour_30,
        #     landmarks.face_contour_31,
        #     landmarks.face_contour_32,
        #     landmarks.face_contour_33,
        #     landmarks.face_contour_34
        # ]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_detections(self, frame, detections):
        for roi, landmarks, identity, identity2, identity3 in zip(*detections):
            self.draw_detection_roi(frame, roi, identity, identity2, identity3)
            self.draw_detection_keypoints(frame, roi, landmarks)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
#        text_size, _ = self.draw_text_with_background(frame,
#                                                      "Frame time: %.3fs" % (self.frame_time),
#                                                      origin, font, text_scale, color)
#        self.draw_text_with_background(frame,
#                                       "FPS: %.1f" % (self.fps),
#                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('Face recognition demo', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream
        check_image = True
        
        if check_image:
            print("Image")
            has_frame, frame = input_stream.read()
            if not has_frame:
                return
    
            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            if allow_grow:
                path = "./media/celebrities/" + new_face_name + ".jpg"
                cv2.imwrite(path, frame)

            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)
            
            # result_name = "./media/media/Result_"
            # result_name += str(datetime.datetime.now().day)
            # result_name += str(datetime.datetime.now().month)
            # result_name += str(datetime.datetime.now().year)
            # result_name += "_"
            # result_name += str(datetime.datetime.now().hour)
            # result_name += str(datetime.datetime.now().minute)
            # result_name += str(datetime.datetime.now().second)
            # result_name += str(datetime.datetime.now().microsecond)
            # result_name += ".jpg"

        else:
            print("Video")
            while input_stream.isOpened():
                has_frame, frame = input_stream.read()
                if not has_frame:
                    break
    
                if self.input_crop is not None:
                    frame = Visualizer.center_crop(frame, self.input_crop)
                detections = self.frame_processor.process(frame)
    
                self.draw_detections(frame, detections)
                self.draw_status(frame, detections)
    
                if output_stream:
                    output_stream.write(frame)
                if self.display:
                    self.display_interactive_window(frame)
                    if self.should_stop_display():
                        break
    
                self.update_fps()
                self.frame_num += 1

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, path):
        input_stream = Visualizer.open_input_stream(path)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if crop_width and crop_height:
            crop_size = (crop_width, crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(output_image, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream

def main(visualizer, s, b, name):

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not verbose else log.DEBUG, stream=sys.stdout)
    
    global allow_grow
    global new_face_name
    
    allow_grow = b
    new_face_name = name
    path = s
    #visualizer = Visualizer()
    visualizer.run(path)
    res = possible_labels.copy()
    possible_labels.clear()
    print("res:",res)
    return res

if __name__ == '__main__':
    main()
