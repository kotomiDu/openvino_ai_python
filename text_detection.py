"""
 Copyright (c) 2019 Intel Corporation

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

from common import IEModel
np.set_printoptions(threshold = np.inf)

class Model(IEModel):
    """ Class that allows worknig with person detectpr models. """

    def __init__(self, model_path, device, ie_core, num_requests = 1):
        """Constructor"""

        super().__init__(model_path, device, ie_core, num_requests)
        # text detection configuration
        self.min_area = 300 
        self.min_height = 10
        self.segm_conf_thr = 0.8
        self.link_conf_thr = 0.8
        
        n, _, h, w = self.input_size
        self.input_height = h
        self.input_width = w
        self.batch_size = n
        

    
    def _prepare_frame(self, frame):
        
        in_frame = cv2.resize(frame, (self.input_width, self.input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        
        return in_frame
        
        
    def _process_output(self,result,image_shape):
        def softmax(logits):
            """ Returns softmax given logits. """

            max_logits = np.max(logits, axis=-1, keepdims=True)
            numerator = np.exp(logits - max_logits)
            denominator = np.sum(numerator, axis=-1, keepdims=True)
            return numerator / denominator
            
        res_link = result[0]
        res_seg= result[1]
        segm_logits = []
        link_logits = []
        #print("link",res_link)
        #print("seg",res_seg)
        for i in range(self.batch_size):
            link_logits_i = res_link[i].transpose((1,2,0)).reshape((int(self.input_height/4), int(self.input_width/4), 8, 2))
            segm_logits_i = res_seg[i].transpose((1,2,0))
            link_logits.append(link_logits_i)
            segm_logits.append(segm_logits_i)
        
        segm_scores = softmax(segm_logits)
        link_scores = softmax(link_logits)
       
        bboxes = self.to_boxes([image_shape], segm_scores[:, :, :,1],  link_scores[:, :, :, :,1])
        return bboxes
    

    def __call__(self, frame):
        """Runs model on the specified input"""

        in_frame = self._prepare_frame(frame)
        # two outputs
        # print("inputdata",in_frame)
        result = self.infer(in_frame)
        out = self._process_output(result,frame.shape)

        return out
    
    
    
    def min_area_rect(self,contour):
        """ Returns minimum area rectangle. """

        (center_x, cencter_y), (width, height), theta = cv2.minAreaRect(contour)
        return [center_x, cencter_y, width, height, theta], width * height
    def mask_to_bboxes(self,mask, image_shape):
        """ Converts mask to bounding boxes. """

        image_h, image_w = image_shape[0:2]

        bboxes = []
        max_bbox_idx = mask.max()
        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        for bbox_idx in range(1, max_bbox_idx + 1):
            bbox_mask = (mask == bbox_idx).astype(np.uint8)
            cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(cnts) == 0:
                continue
            cnt = cnts[0]
            rect, rect_area = self.min_area_rect(cnt)

            box_width, box_height = rect[2:-1]
            if min(box_width, box_height) < self.min_height:
                continue

            if rect_area < self.min_area:
                continue

            xys = self.rect_to_xys(rect, image_shape)
            bboxes.append(xys)

        return bboxes
    def rect_to_xys(self,rect, image_shape):
        """ Converts rotated rectangle to points. """

        height, width = image_shape[0:2]

        def get_valid_x(x_coord):
            return np.clip(x_coord, 0, width - 1)

        def get_valid_y(y_coord):
            return np.clip(y_coord, 0, height - 1)

        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        for i_xy, (x_coord, y_coord) in enumerate(points):
            x_coord = get_valid_x(x_coord)
            y_coord = get_valid_y(y_coord)
            points[i_xy, :] = [x_coord, y_coord]
        points = np.reshape(points, -1)
        return points


    def get_neighbours(self,x_coord, y_coord):
        """ Returns 8-point neighbourhood of given point. """
        return [(x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1), \
            (x_coord - 1, y_coord), (x_coord + 1, y_coord), \
            (x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)]
    def is_valid_coord(self,x_coord, y_coord, width, height):
        """ Returns true if given point inside image frame. """

        return 0 <= x_coord < width and 0 <= y_coord < height


    def decode_image(self,segm_scores, link_scores, segm_conf_threshold, link_conf_threshold):
        """ Convert softmax scores to mask. """
        segm_mask = segm_scores >= segm_conf_threshold
        link_mask = link_scores >= link_conf_threshold
        points = list(zip(*np.where(segm_mask)))
        height, width = np.shape(segm_mask)
        group_mask = dict.fromkeys(points, -1)

        def find_parent(point):
            return group_mask[point]

        def set_parent(point, parent):
            group_mask[point] = parent

        def is_root(point):
            return find_parent(point) == -1

        def find_root(point):
            root = point
            update_parent = False
            while not is_root(root):
                root = find_parent(root)
                update_parent = True

            if update_parent:
                set_parent(point, root)

            return root

        def join(point1, point2):
            root1 = find_root(point1)
            root2 = find_root(point2)

            if root1 != root2:
                set_parent(root1, root2)

        def get_all():
            root_map = {}

            def get_index(root):
                if root not in root_map:
                    root_map[root] = len(root_map) + 1
                return root_map[root]

            mask = np.zeros_like(segm_mask, dtype=np.int32)
            for point in points:
                point_root = find_root(point)
                bbox_idx = get_index(point_root)
                mask[point] = bbox_idx
            return mask

        for point in points:
            y_coord, x_coord = point
            neighbours = self.get_neighbours(x_coord, y_coord)
            for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
                if self.is_valid_coord(neighbour_x, neighbour_y, width, height):

                    link_value = link_mask[y_coord, x_coord, n_idx]
                    segm_value = segm_mask[neighbour_y, neighbour_x]
                    if link_value and segm_value:
                        join(point, (neighbour_y, neighbour_x))

        mask = get_all()
        return mask

    def decode_batch(self,segm_scores, link_scores):
        """ Returns boxes mask for each input image in batch."""

        batch_size = segm_scores.shape[0]
        batch_mask = []
        for image_idx in range(batch_size):
            image_pos_pixel_scores = segm_scores[image_idx, :, :]
            image_pos_link_scores = link_scores[image_idx, :, :, :]
            mask = self.decode_image(image_pos_pixel_scores, image_pos_link_scores,
                                self.segm_conf_thr, self.link_conf_thr)
            batch_mask.append(mask)
        return np.asarray(batch_mask, np.int32)

    def to_boxes(self,image_shapes, segm_pos_scores, link_pos_scores):
        """ Returns boxes for each image in batch. """
        bboxes = []
        for image_shape, seg_item, link_item in zip(image_shapes,segm_pos_scores,link_pos_scores):
            seg_item = np.expand_dims(seg_item, axis=0)
            link_item = np.expand_dims(link_item, axis=0)
            mask = self.decode_batch(seg_item, link_item)[0, ...]
            item_box = self.mask_to_bboxes(mask, image_shape)
            bboxes.append(item_box)

        # print(image_data.shape,segm_pos_scores.shape,link_pos_scores.shape)
        # mask = self.decode_batch(segm_pos_scores, link_pos_scores, conf)[0, ...]
        # bboxes = self.mask_to_bboxes(mask, conf, image_data.shape)

        return bboxes
