"""
Copyright 2021 Huawei Technologies Co., Ltd

CREATED:  2020-6-04 20:12:13
MODIFIED: 2021-11-25 12:00:45
"""

# -*- coding:utf-8 -*-
import acl
import time
import cv2
import numpy as np
from functools import cmp_to_key

from utils.acl_util import check_ret
from utils.postprocessing import Process
from utils.box_utils import BBox, compare_alldims, compare_titledims
from utils.imgproc import resize_aspect_ratio, normalizeMeanVariance, \
    cvt2HeatmapImg
from data.constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST


class Model(object):
    
    def __init__(self,
                 device_id,
                 model_path):
        self.device_id = device_id
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context  = None             # pointer
        self.stream  = None
        self.input_data = None
        self.output_data = None
        self.model_desc = None          # pointer when using
        self.input0_dataset_buffer = None
        self.input1_dataset_buffer = None
        self.input1_buffer = None
        self.model_input_width = None
        self.model_input_height = None
        self.model_output_width = None
        self.model_output_height = None
        self.input_dataset = None
        self.init_resource()
        
        
    def __del__(self):
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        if self.input1_buffer:
            ret = acl.rt.free(self.input1_buffer)
            check_ret("acl.rt.free", ret)

        print("[Model] class Model release source success")
        
        if self.stream:
            acl.rt.destroy_stream(self.stream)

        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        
        print("[ACL] class Sample release source success")
        
        
    def init_resource(self):
        if bool(acl.rt.get_device(0)[1]):
            print("[ACL] init resource stage:")
            acl.init()

            ret = acl.rt.set_device(self.device_id)
            check_ret("acl.rt.set_device", ret)

            self.context, ret = acl.rt.create_context(self.device_id)
            check_ret("acl.rt.create_context", ret)

            self.stream, ret = acl.rt.create_stream()
            check_ret("acl.rt.create_stream", ret)
            print("[ACL] init resource stage success")

            print("[MODEL] class Model init resource stage:")
            # context
            acl.rt.set_context(self.context)
        
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_output_dataset(output_size)
        print("=" * 100)
        
        print("model input size", input_size)
        for i in range(input_size):
            print("input ", i)
            print("model input dims", acl.mdl.get_input_dims(self.model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(self.model_desc, i))
            self.model_input_height, self.model_input_width = acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][2:]
        print("=" * 100)
        
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.model_output_height, self.model_output_width = acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'][1:3]
        print("=" * 100)

        print("[MODEL] class Model init resource stage success")
        print("=" * 100)
        
        
    def run(self, img, cropped = False, poly = True, boxes_coord = None, characters = None,  thresholds = None ): # Overloaded function
        process = Process(poly, characters, thresholds)
        print("[PROCESS] init process success")
        
        if  not cropped and not boxes_coord :
            t0 = time.time()
            
            # resize
            mag_ratio = round((self.model_input_width / self.model_input_height), 1)
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, self.model_input_width, cv2.INTER_LINEAR, mag_ratio)
            ratio_h = ratio_w = 1 / target_ratio

            # preprocessing
            x = normalizeMeanVariance(img_resized)
            x = np.transpose(x, (2,0,1))    # [h, w, c] to [c, h, w]               
            image_np_expanded = np.expand_dims(x, axis=0)   # [c, h, w] to [b, c, h, w]
            print("[PreProc] image_np_expanded shape:", image_np_expanded.shape)
            img_resized = np.ascontiguousarray(image_np_expanded)

            img_dev_ptr, img_buf_size = self.transfer_img_to_device(img_resized)
            print("[ACL] img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)

            self._gen_input_dataset(img_dev_ptr, img_buf_size)
            self.forward()

            ret = acl.rt.free(img_dev_ptr)
            check_ret("acl.rt.free", ret)

            y = self.get_model_output_by_index(0).reshape(-1,  self.model_output_height, 
                                                                                           self.model_output_width, 2)
            
            # make score and link map
            score_text = y[0,:,:,0]
            score_link = y[0,:,:,1]

            t0 = time.time() - t0
            t1 = time.time()

            # Post-processing
            bboxes, polys = process.getDetBoxes(score_text, score_link)

            # coordinate adjustment
            bboxes = process.adjustResultCoordinates(bboxes, ratio_w, ratio_h)
            polys = process.adjustResultCoordinates(polys, ratio_w, ratio_h)
            for k in range(len(polys)):
                if polys[k] is None: polys[k] = bboxes[k]

            t1 = time.time() - t1

            print("[RESULT] infer / postproc time : {:.3f} / {:.3f}".format(t0, t1))

            return bboxes, polys
        
        else :
            bboxes_sorted = None
            t0 = time.time()
            
            # pre-processing
            print("[INFO] recognizing texts . . .")

            if cropped is False :
                bboxes_sorted = self._read_bboxes(boxes_coord, img)
                # crop textboxes from image
                cropped_imgs = []
                for box in bboxes_sorted:
                    # crop each rectangle, enlarged by horizontal offset value, from original image
                    cropped_imgs.append(box.get_crop(0, 0))

                for i, cropped_img in enumerate(cropped_imgs):
                    preds = self._preprocessing(cropped_img)
                # Post-processing
                    process.text_boxes(preds, bboxes_sorted, i)    # recognize corresponding text for each text box
            
            else:
                bboxes = []
                bboxes.append(BBox([0, 0, 0, 0, 0, 0, 0, 0], 1, 0, img))
                bboxes_sorted = sorted(bboxes, key=cmp_to_key(compare_alldims))
                preds = self._preprocessing(img)
                
                # Post-processing
                process.text_boxes(preds, bboxes_sorted, 0)    # recognize corresponding text
                
            print("[INFO] recognition done!")

            if len(bboxes_sorted)==0:
                return len(bboxes_sorted)

            # concatenate text boxes that mistakenly detected as two words but actually is one word
            bboxes_sorted = process.get_concat_same() 

            t0 = time.time() - t0
            print("[RESULT] image runtime : {:.3f}".format(t0))

            return bboxes_sorted
        
    
    def _read_bboxes(self, boxes_coord, image):
        # save box coordinates from EAST output which is pixel location of each vertex of rectangle
        bboxes = []
        print("[INFO] loading text boxes . . .")
        for box_coord in boxes_coord:
            bboxes.append(BBox(box_coord, 1, 0, image))
        bboxes_sorted = sorted(bboxes, key=cmp_to_key(compare_alldims))
        
        print('[INFO]  {} text boxes found.'.format(len(bboxes_sorted)))
        
        return bboxes_sorted
    
    
    def _preprocessing(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/127.5 - 1.0
        img_resized = cv2.resize(img_gray,(self.model_input_width, self.model_input_height ), interpolation=cv2.INTER_CUBIC)
        img_np_expanded = np.expand_dims(np.float32(img_resized), (0,1))

        print("[PreProc] image_np_expanded shape:", img_np_expanded.shape)
        img_dev_ptr, img_buf_size = self.transfer_img_to_device( np.ascontiguousarray(
                                                                                                                img_np_expanded))
        print("[ACL] img_dev_ptr, img_buf_size: ", img_dev_ptr, img_buf_size)

        self._gen_input_dataset(img_dev_ptr, img_buf_size)
        self.forward()

        ret = acl.rt.free(img_dev_ptr)
        check_ret("acl.rt.free", ret)
        
        preds = self.get_model_output_by_index(0).reshape(-1,  self.model_output_height, 
                                                                                           self.model_output_width)
        
        return preds
    
    
    def transfer_img_to_device(self, img_resized):
        
        # BGR to RGB
        img_host_ptr = acl.util.numpy_to_ptr(img_resized)
        img_buf_size = img_resized.itemsize * img_resized.size
        print("[ACL] img_host_ptr, img_buf_size: ", img_host_ptr, img_buf_size)
        img_dev_ptr, ret = acl.rt.malloc(img_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_dev_ptr, img_buf_size, img_host_ptr, img_buf_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        
        return img_dev_ptr, img_buf_size
    
    
    def forward(self):
        print('[MODEL] execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_data)
        #check_ret("acl.mdl.execute", ret)

        #free the input dataset
        if self.input0_dataset_buffer:
            ret = acl.destroy_data_buffer(self.input0_dataset_buffer)
            check_ret("acl.destroy_data_buffer", ret)
            self.input0_dataset_buffer = None
        if self.input_dataset:
            ret = acl.mdl.destroy_dataset(self.input_dataset)
            check_ret("acl.destroy_dataset", ret)
            self.input_dataset = None

        print('[MODEL] execute stage success')
        
        
    def _gen_input_dataset(self, dvpp_output_buffer, dvpp_output_size, image_height=None, image_width=None):
        print("[MODEL] create model input dataset:")
        self.input_dataset = acl.mdl.create_dataset()
        self.input0_dataset_buffer = acl.create_data_buffer(dvpp_output_buffer,
                                                      dvpp_output_size)
        _, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset,
            self.input0_dataset_buffer)
        if ret:
            ret = acl.destroy_data_buffer(self.input0_dataset_buffer)
            self.input0_dataset_buffer = None
            check_ret("acl.destroy_data_buffer", ret)

        print("[MODEL] create model input dataset success")
        
        
    def _gen_output_dataset(self, size):
        print("[MODEL] create model output dataset:")
        dataset = acl.mdl.create_dataset()
        for i in range(size):
            temp_buffer_size = acl.mdl.\
                get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            dataset_buffer = acl.create_data_buffer(
                temp_buffer,
                temp_buffer_size)

            _, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
            if ret:
                acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)
        self.output_data = dataset
        print("[MODEL] create model output dataset success")
        
        
    def _release_dataset(self, ):
        for dataset in [self.input_dataset, self.output_data]:
            if not dataset:
                continue
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)

            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)
            
            
    def get_model_output_by_index(self, i):
        temp_output_buf = acl.mdl.get_dataset_buffer(self.output_data, i)

        infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
        infer_output_size = acl.get_data_buffer_size(temp_output_buf)

        output_host, _ = acl.rt.malloc_host(infer_output_size)
        acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
                              infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        
        return acl.util.ptr_to_numpy(output_host, (infer_output_size//4,), 11)