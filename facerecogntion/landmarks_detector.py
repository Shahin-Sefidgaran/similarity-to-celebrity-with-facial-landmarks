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

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

class LandmarksDetector(Module):
    POINTS_NUMBER = 5
    # POINTS_NUMBER = 35

    class Result:
        def __init__(self, outputs):
            self.points = outputs

            p = lambda i: self[i]
            self.left_eye = p(0)
            self.right_eye = p(1)
            self.nose_tip = p(2)
            self.left_lip_corner = p(3)
            self.right_lip_corner = p(4)

            # self.left_eye_right = p(0)
            # self.left_eye_left = p(1)
            # self.right_eye_left = p(2)
            # self.right_eye_right = p(3)
            # self.nose_tip = p(4)
            # self.nose_nasal = p(5)
            # self.nose_left = p(6)
            # self.nose_right = p(7)
            # self.mouth_left = p(8)
            # self.mouth_right = p(9)
            # self.mouth_top = p(10)
            # self.mouth_down = p(11)
            # self.left_eyebrow_left = p(12)
            # self.left_eyebrow_middle = p(13)
            # self.left_eyebrow_right = p(14)
            # self.right_eyebrow_left = p(15)
            # self.right_eyebrow_middle = p(16)
            # self.right_eyebrow_right = p(17)
            
            # # Face Contour
            # self.face_contour_18 = p(18)
            # self.face_contour_19 = p(19)
            # self.face_contour_20 = p(20)
            # self.face_contour_21 = p(21)
            # self.face_contour_22 = p(22)
            # self.face_contour_23 = p(23)
            # self.face_contour_24 = p(24)
            # self.face_contour_25 = p(25)
            # self.face_contour_26 = p(26) # Chin Center
            # self.face_contour_27 = p(27)
            # self.face_contour_28 = p(28)
            # self.face_contour_29 = p(29)
            # self.face_contour_30 = p(30)
            # self.face_contour_31 = p(31)
            # self.face_contour_32 = p(32)
            # self.face_contour_33 = p(33)
            # self.face_contour_34 = p(34)

        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(LandmarksDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1],
                              model.outputs[self.output_blob].shape), \
            "Expected model output shape %s, but got %s" % \
            ([1, self.POINTS_NUMBER * 2, 1, 1],
             model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [LandmarksDetector.Result(out[self.output_blob].reshape((-1, 2))) \
                      for out in outputs]
        return results
