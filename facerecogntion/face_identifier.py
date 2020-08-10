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

import cv2
import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

class FaceIdentifier(Module):
    # Taken from the description of the model:
    # intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner
        # (0.3701297, 0.38294342), # left eye
        # (0.2108722, 0.3740713), # right eye
        # (0.62008095, 0.38390845), # nose tip
        # (0.77708155, 0.37416008), # left lip corner
        # (0.50204325, 0.55569977),
        # (0.5028297, 0.62456286,),
        # (0.3749863, 0.58794475),
        # (0.623469, 0.5888293),
        # (0.33817622, 0.72972566),
        # (0.66225123, 0.7303796),
        # (0.5039418, 0.69415116),
        # (0.50254923, 0.7992252),
        # (0.13421932, 0.2878478),
        # (0.25960004, 0.23185839),
        # (0.40167096, 0.27780104),
        # (0.58400536, 0.27573866),
        # (0.72641057, 0.23037605),
        # (0.8485633, 0.28932196),
        # (0.06008479, 0.37317428),
        # (0.06305936, 0.4730534),
        # (0.07798474, 0.5689985),
        # (0.10198098, 0.663603),
        # (0.14060964, 0.7528989),
        # (0.20309275, 0.83044225),
        # (0.28472105, 0.89283603),
        # (0.38159856, 0.9414395),
        # (0.5045598, 0.9609611),
        # (0.62053126, 0.94369805),
        # (0.7130332, 0.89847785),
        # (0.7907388, 0.83879846),
        # (0.84999853, 0.76312506),
        # (0.8864083, 0.67562014),
        # (0.90868145, 0.5816711),
        # (0.9214608 , 0.48534274),
        # (0.9229456 , 0.38203907)]

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, model, match_threshold=0.5, match_algo='HUNGARIAN'):
        super(FaceIdentifier, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"

        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert len(model.outputs[self.output_blob].shape) == 4 or \
            len(model.outputs[self.output_blob].shape) == 2, \
            "Expected model output shape [1, n, 1, 1] or [1, n], got %s" % \
            (model.outputs[self.output_blob].shape)

        self.faces_database = None

        self.match_threshold = match_threshold
        self.match_algo = match_algo

    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(FaceIdentifier, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_threshold(self):
        return self.match_threshold

    def get_matches(self):
        descriptors = self.get_descriptors()

        matches = []
        matches2 = []
        matches3 = []
        if len(descriptors) != 0:
            matches, matches2, matches3 = self.faces_database.match_faces(descriptors, self.match_algo)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if distance == 1:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)

            results.append(self.Result(id, distance, descriptors[num]))
        
        results2 = []
        for num, match in enumerate(matches2):
            id = match[0]
            distance = match[1]
            if distance == 1:
               id = self.UNKNOWN_ID
               unknowns_list.append(num)

            results2.append(self.Result(id, distance, descriptors[num]))
        
        results3 = []
        for num, match in enumerate(matches3):
            id = match[0]
            distance = match[1]
            if distance == 1:
               id = self.UNKNOWN_ID
               unknowns_list.append(num)

            results3.append(self.Result(id, distance, descriptors[num]))

        return results, unknowns_list, results2, results3

    def get_descriptors(self):
        return [out[self.output_blob].flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            "2d input arrays are expected, got %s" % (src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=(0))
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=(0))

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - \
            np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            "Input lengths differ, got %s and %s" % \
            (len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            assert len(image.shape) == 4, "Face image is expected"
            image = image[0]

            scale = np.array((image.shape[-1], image.shape[-2]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=np.float64) * scale
            landmarks = image_landmarks.get_array() * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            img = image.transpose((1, 2, 0))
            cv2.warpAffine(img, transform, tuple(scale), img,
                           flags=cv2.WARP_INVERSE_MAP)
            image[:] = img.transpose((2, 0, 1))
