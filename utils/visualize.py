#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]

from utils.boxes import xyxy2cxcywh, xywh2xyxy, rbox2poly


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking1(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh[:4]
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{} {:.2f}'.format(int(obj_id), scores[i]) if scores is not None else '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    try:
        tlwhs = np.array(tlwhs)
        tlwhs[:, 2] = np.array(tlwhs)[:, 2] + np.array(tlwhs)[:, 0]
        tlwhs[:, 3] = np.array(tlwhs)[:, 3] + np.array(tlwhs)[:, 1]
    except:
        pass
    for i, tlwh in enumerate(tlwhs):
        obj_id = int(obj_ids[i])
        id_text = '{} {:.2f}'.format(int(obj_id), scores[i]) if scores is not None else '{}'.format(int(obj_id))
        intbox = tlwh[:2]
        tlwh = rbox2poly(xyxy2cxcywh(np.array([tlwh])))[0]

        polygon_list = np.array([(tlwh[0], tlwh[1]), (tlwh[2], tlwh[3]),
                                 (tlwh[4], tlwh[5]), (tlwh[6], tlwh[7])], np.int32)
        color = get_color(abs(obj_id))
        cv2.drawContours(image=im, contours=[polygon_list], contourIdx=-1, color=color, thickness=line_thickness)

        # cv2.rectangle(im, pt1=(int(intbox[0]),int(intbox[1])), pt2=(int(intbox[0]+5),int(intbox[1]+5)), color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (int(intbox[0]), int(intbox[1])), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

    # for i, tlwh in enumerate(tlwhs):
    #     x1, y1, w, h = tlwh[:4]
    #     intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    #     obj_id = int(obj_ids[i])
    #     id_text = '{} {:.2f}'.format(int(obj_id), scores[i]) if scores is not None else '{}'.format(int(obj_id))
    #     if ids2 is not None:
    #         id_text = id_text + ', {}'.format(int(ids2[i]))
    #     color = get_color(abs(obj_id))
    #     cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    #     cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
    #                 thickness=text_thickness)


def plot_det(image, tlwhs, img_info, args, fps=0., ids2=None):
    # img_h, img_w = img_info['height'], img_info['width']
    # scale = min(args.tsize[0] / float(img_h), args.tsize[1] / float(img_w))
    # tlwhs /= scale

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    line_thickness = 3
    radius = max(5, int(im_w / 140.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        intbox = tuple(map(int, (x1, y1, w, h)))
        color = get_color(abs(0))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    return im


def plot_kf_pred(strack_pool, img_info, args):
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in strack_pool:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)

    def plot_track(image, tlwhs):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]
        line_thickness = 3
        # for i, tlwh in enumerate(tlwhs):
        #     x1, y1, w, h = tlwh[:4]
        #     intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        #     color = [255, 255, 255]
        #     cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        try:
            tlwhs = np.array(tlwhs)
            tlwhs[:, 2] = np.array(tlwhs)[:, 2] + np.array(tlwhs)[:, 0]
            tlwhs[:, 3] = np.array(tlwhs)[:, 3] + np.array(tlwhs)[:, 1]
        except:
            # print(" not plot")
            pass
        for i, tlwh in enumerate(tlwhs):
            tlwh = rbox2poly(xyxy2cxcywh(np.array([tlwh])))[0]

            polygon_list = np.array([(tlwh[0], tlwh[1]), (tlwh[2], tlwh[3]),
                                     (tlwh[4], tlwh[5]), (tlwh[6], tlwh[7])], np.int32)
            color = get_color(abs(5))
            cv2.drawContours(image=im, contours=[polygon_list], contourIdx=-1, color=color, thickness=line_thickness)

        return im

    online_im = plot_track(img_info['raw_img'], online_tlwhs)
    return online_im


def plot_circle(image, tlwhs, img_info, args):
    img_h, img_w = img_info['height'], img_info['width']
    scale = min(args.tsize[0] / float(img_h), args.tsize[1] / float(img_w))
    tlwhs /= scale

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    line_thickness = 3
    radius = max(5, int(im_w / 140.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, r, r = tlwh
        # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        intbox = tuple(map(int, (x1, y1, r, r)))
        color = [114, 114, 0]
        cv2.circle(im, intbox[0:2], intbox[3], color=color, thickness=line_thickness)
    return im


def plot_polygon(image, tlwhs, img_info, args):
    # img_h, img_w = img_info['height'], img_info['width']
    # scale = min(args.tsize[0] / float(img_h), args.tsize[1] / float(img_w))
    # tlwhs /= scale

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    line_thickness = 3
    radius = max(5, int(im_w / 140.))

    for i, tlwh in enumerate(tlwhs):
        polygon_list = np.array([(tlwh[0], tlwh[1]), (tlwh[2], tlwh[3]),
                                 (tlwh[4], tlwh[5]), (tlwh[6], tlwh[7])], np.int32)
        color = get_color(abs(0))
        cv2.drawContours(image=im, contours=[polygon_list], contourIdx=-1, color=color, thickness=line_thickness)

        # x1, y1, w, h = tlwh
        # # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        # intbox = tuple(map(int, (x1, y1, w, h)))
        #
        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
