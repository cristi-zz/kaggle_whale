__author__ = "visoft"

import os.path as path
import numpy as np
import cv2
import json
import skimage.measure as skmeasure
import skimage.filters.rank as skrank
import pickle
import math


def parse_anlt(json_path):
    """
    Parse Anil point list
    See: https://github.com/anlthms/whale-2015

    :param json_path:
    :return:
    """
    whale_dict = {}
    with open(json_path,'r') as f:
        data = json.load(f)
        for element in data:
            fname = element['filename']
            annot = element['annotations']
            assert len(annot) == 1
            annot = annot[0]
            assert annot['class'] == 'point'
            x = annot['x']
            y = annot['y']
            whale_dict[fname] = (x,y)
    return whale_dict


def get_some_edges_ded(img, edge_percent=0.1, low_threshold_ratio=0.8, pre_gauss=None, low_th=10, step=20):
    """
    Gets some edges by specifying the percent of desired strong edges in the image

    :param img:
    :param edge_percent:
    :param low_threshold_ratio:
    :param pre_gauss:
    :return:
    """

    img_size = img.shape[0:2]
    pixel_count = img_size[0] * img_size[1]

    if pre_gauss is not None:
        img = cv2.GaussianBlur(np.copy(img), (0, 0), pre_gauss)

    for th1 in range(255, low_th, -step):
        th2 = th1 * low_threshold_ratio
        edges = cv2.Canny(img, th1, th1, L2gradient=True)
        non_zero_count = float(np.sum(edges != 0))
        non_zero_percent = non_zero_count / pixel_count * 100

        if non_zero_percent >= edge_percent:
            # print "selected th: " + str(th1)
            th2 = th1 * low_threshold_ratio
            edges = cv2.Canny(img, th1, th2, L2gradient=True)
            return edges
    # print "default: " + str(th1)
    th2 = low_th * low_threshold_ratio
    edges = cv2.Canny(img, th1, th2, L2gradient=True)
    return edges


def canny_pyramid(img_bw, sigmas=None, edge_percent=0.1, lowest_th=10):
    """
    Used by edge similarity. Pull out sigmas if needed

    :param img_bw:
    :return:
    """

    if sigmas is None:
        sigmas = [2, 2.5, 3, 4, 5]
    N = len(sigmas)
    results = []
    for k in range(N):
        edge = get_some_edges_ded(img_bw, edge_percent, 0.5, pre_gauss=sigmas[k], low_th=lowest_th)
        results.append(np.copy(edge))

    return results


def get_edge_correlation(img_gray, show):
    """
    Maybe as a feature in a classifier. But not good to separate whales


    :param img_gray:
    :param show:
    :return:
    """
    RW = 70
    RH = 70
    RW_C = img_gray.shape[1] / RW
    RH_C = img_gray.shape[0] / RH
    gray_dim = 500
    gray_fx = float(gray_dim) / img_gray.shape[1]
    gray_res = cv2.resize(img_gray, (gray_dim, int(img_gray.shape[0] * gray_fx)))

    canny_pyr = canny_pyramid(img_gray, [0.5, 2, 2.5, 3, 4, 5, 8, 15], edge_percent=0.1, lowest_th=30)
    N = len(canny_pyr)
    NP = img_gray.shape[1] * img_gray.shape[0]
    NP_small = RW * RH

    orient_pyr = []

    gradX = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
    gradY = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)
    _, orient = cv2.cartToPolar(gradX, gradY, angleInDegrees=True)

    hist_list = []
    for i in range(N):
        img_to_comp = orient[canny_pyr[i] > 0]
        h = cv2.calcHist([img_to_comp], [0], None, [16], [0, 360])
        h = h / NP
        hist_list.append(h)
        # if show:
        #     cv2.imshow("Canny for level " + str(i),cv2.resize(canny_pyr[i],(0,0),fx=0.2,fy=0.2))
    big_hist = np.vstack(hist_list).astype(np.float32)

    correlation = np.zeros((RH_C, RW_C), np.float64)
    for r in range(RH_C):
        for c in range(RW_C):
            sm_hist_list = []
            for i in range(N):
                suborient = orient[r * RW:(r + 1) * RW, c * RH:(c + 1) * RH]
                subcanny = canny_pyr[i][r * RW:(r + 1) * RW, c * RH:(c + 1) * RH]
                subimg = suborient[subcanny > 0]
                h = cv2.calcHist([subimg], [0], None, [16], [0, 360])
                h = h / NP_small
                sm_hist_list.append(h)
            sm_hist = np.vstack(sm_hist_list).astype(np.float32)
            corr = cv2.compareHist(big_hist, sm_hist, cv2.HISTCMP_CHISQR_ALT)
            correlation[r, c] = corr

    correlation = correlation - np.min(correlation)
    correlation /= np.max(correlation)

    corr_sc_1 = cv2.resize(correlation, (int(RW * RW_C * gray_fx), int(RH * RH_C * gray_fx)))
    corr_sc = np.zeros((gray_res.shape[0], gray_res.shape[1]))
    corr_sc[0:corr_sc_1.shape[0], 0:corr_sc_1.shape[1]] = corr_sc_1

    if show:
        print "Edge corr statistics:"
        print np.mean(correlation)
        print np.min(correlation)
        print np.max(correlation)
        correlation = correlation - np.min(correlation)
        correlation /= np.max(correlation)
        # cv2.imshow("edge corr",cv2.resize(correlation,(0,0),fx=5,fy=5))
        cv2.imshow("edge corr", corr_sc)

    return corr_sc


def get_histo_correlation(img, show):
    """
    Pretty good job in separating whales
    The idea is taken from https://github.com/eduardofv/whale_detector
    https://www.kaggle.com/c/noaa-right-whale-recognition/forums/t/17473/finding-the-whale-by-histogram-similarity

    Just histogram is not enough, I also used some edge statistics.

    :param img:
    :param show:
    :return:
    """
    img_hist = np.copy(img).astype(np.float)
    # This is the key to good detection. Standardize all the histograms
    for k in range(3):
        avg = np.mean(img_hist[:, :, k])
        std = np.std(img_hist[:, :, k])
        img_hist[:, :, k] -= avg
        img_hist[:, :, k] /= std
        img_hist[:, :, k] *= 30
        img_hist[:, :, k] += 128
        img_hist[:, :, k] = np.maximum(img_hist[:, :, k], 0)
        img_hist[:, :, k] = np.minimum(img_hist[:, :, k], 255)
    img_hist = img_hist.astype(np.uint8)

    RW = 50
    RH = 50
    RW_C = img.shape[1] / RW
    RH_C = img.shape[0] / RH
    NP = img.shape[1] * img.shape[0]
    NP_small = RW * RH

    gray_dim = 500
    gray_fx = float(gray_dim) / img.shape[1]

    if False:
        prm = ([0, 1, 2], None, [16, 16, 16], [0, 255, 0, 255, 0, 255])
        th_mult = 1.6
    else:
        prm = ([1, 2], None, [16, 16], [0, 255, 0, 255])
        th_mult = 1.4

    gray = img[:, :, 0]
    gray_res = cv2.resize(gray, (gray_dim, int(img.shape[0] * gray_fx)))
    full_hist = cv2.calcHist([img_hist], *prm).astype(np.float32)
    # #create feature vector
    correlation = np.zeros((RH_C, RW_C), np.float64)
    for r in range(RH_C):
        for c in range(RW_C):
            subimg = img_hist[r * RW:(r + 1) * RW, c * RH:(c + 1) * RH, :]
            sub_hist = cv2.calcHist([subimg], *prm).astype(np.float32)
            corr = cv2.compareHist(full_hist, sub_hist, cv2.HISTCMP_CORREL)
            correlation[r, c] = corr
    if show:
        print "Histogram corr stats:"
        print str(np.mean(correlation))
        print np.min(correlation)
        print np.max(correlation)

    # corr_blur = cv2.GaussianBlur(correlation,None,sigmaX=9)
    # if show:
    #     cv2.imshow("correlation blur",corr_blur)
    # correlation = correlation + corr_blur

    corr_sc_1 = cv2.resize(correlation, (int(RW * RW_C * gray_fx), int(RH * RH_C * gray_fx)))
    corr_sc = np.ones((gray_res.shape[0], gray_res.shape[1]))
    corr_sc[0:corr_sc_1.shape[0], 0:corr_sc_1.shape[1]] = corr_sc_1

    return corr_sc, th_mult


def seg_using_histo(img_o, show=False, fname=""):
    """
    Preselect with histo similarity, selection with edge similarity
    https://www.kaggle.com/c/noaa-right-whale-recognition/forums/t/17473/finding-the-whale-by-histogram-similarity

    :param img_o:
    :param show:
    :return:
    """

    img = cv2.cvtColor(img_o, cv2.COLOR_RGB2YCrCb)

    edge_correl = get_edge_correlation(img[:, :, 0], show)
    correlation, th_mult = get_histo_correlation(img, show)

    # process the edge correlation
    edge_correl = (edge_correl * 255).astype(np.uint8)
    e_th_mult = -0.2
    th, _ = cv2.threshold(edge_correl, 0, 1, cv2.THRESH_OTSU)
    avg_th = np.mean(edge_correl[edge_correl > th])
    std_th = max(np.std(edge_correl[edge_correl > th]), 1)
    _, edge_th = cv2.threshold(edge_correl, th + std_th * e_th_mult, 1, cv2.THRESH_BINARY)
    edge_th = cv2.morphologyEx(edge_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                               iterations=1)

    if show:
        print "edge th selected by otsu= " + str(th) + " std of selected pixels= " + str(std_th)
        cv2.imshow("edge corr th", edge_th * 255)

    # #  not good
    # #multiply and store result
    # corr_mult = np.multiply(edge_correl,correlation)
    # corr_mult /= np.max(corr_mult)
    # if show:
    #     cv2.imshow("correlation multiplied with edges",corr_mult)


    # process correlation
    corr_sc = np.maximum(correlation, 0)
    corr_sc = 1 - corr_sc
    corr_sc = corr_sc * 255
    corr_sc = corr_sc.astype(np.uint8)
    th, _ = cv2.threshold(corr_sc, 0, 1, cv2.THRESH_OTSU)
    avg_th = np.mean(corr_sc[corr_sc > th])
    std_th = np.std(corr_sc[corr_sc > th])
    _, corr_sc_th = cv2.threshold(corr_sc, th + std_th * th_mult, 1, cv2.THRESH_BINARY)
    if show:
        print "Hist corr threshold selected by otsu " + str(th)
        cv2.imshow("correlation raw th", corr_sc_th * 255)
    # Removing small features
    corr_sc_th = cv2.morphologyEx(corr_sc_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                  iterations=1)
    if show:
        cv2.imshow("correlation open th", corr_sc_th * 255)

    # Joining close patches
    corr_sc_th = cv2.morphologyEx(corr_sc_th, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                                  iterations=2)

    # selecting by edge correlation

    candidate_comp, num_of_comp = skmeasure.label(corr_sc_th, neighbors=8, background=0, return_num=True)
    candidate_comp_to_take = candidate_comp * edge_th
    list_comp_to_take = np.unique(candidate_comp_to_take)
    list_comp_to_take = list_comp_to_take[1:]  # remove -1

    selected_patches = np.zeros(corr_sc_th.shape, np.uint8)
    for k in list_comp_to_take:
        selected_patches += (candidate_comp == k).astype(np.uint8)

    # fall back procedure. If no patches are selected, return just the correlation pathches

    if np.sum(selected_patches) < 10:
        print "Fallback to histo correlation for " + fname
        selected_patches = corr_sc_th

    result_final = cv2.resize(selected_patches, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    result_final = ((result_final > 0.01) * 255).astype(np.uint8)
    if show:
        overlap_patches = corr_sc_th * 128
        overlap_patches = overlap_patches + edge_th * 127
        cv2.imshow("Gray image", cv2.resize(img_o, (corr_sc_th.shape[1], corr_sc_th.shape[0])))
        cv2.imshow("correlation", corr_sc)
        cv2.imshow("correlation thresholded and dilated", corr_sc_th * 255)
        cv2.imshow("Overlapped patches", overlap_patches)
        cv2.imshow("Selected patches by edge histo (final result)", selected_patches * 255)

    return result_final


def pack_whale_candiates(th_img, color_img, show=False):
    """
    Packs the contours with the bounding box (rectangular, paralel with axis), with the area and with the contour.
    Returns them sorted by the area.

    :param th_img:
    :param color_img:
    :param show:
    :return: (bounding box, area of the shape,the actual contour)
    """
    _, conts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    id = -1  # 11
    bbox_list = []

    for cont in conts:
        counter += 1
        area = cv2.contourArea(cont)
        bbox = cv2.boundingRect(cont)
        bbox = list(bbox)
        bbox[0] = max(bbox[0], bbox[0] - 150)
        bbox[1] = max(bbox[1], bbox[1] - 150)
        bbox[2] = min(color_img.shape[1], bbox[2] + 150)
        bbox[3] = min(color_img.shape[0], bbox[3] + 150)

        bbox_list.append((bbox, area, cont))

    bbox_list_sorted = sorted(bbox_list, key=lambda x: x[1], reverse=True)

    return bbox_list_sorted


def get_patch_and_texture_features(img, contours):
    """
    Features for each patch candidate. Takes some time (~tens of seconds)

    :param img:
    :param contours:
    :return:
    """
    bw_mask = np.zeros(img.shape[0:2], np.int32)
    for c in range(len(contours)):
        cv2.drawContours(bw_mask, contours, c, c + 1, -1)

    img_y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # 1-3
    features = []
    for k in range(3):
        fts = []
        for c in range(len(contours)):
            avg_ch = np.average(img_y[:, :, k], weights=(bw_mask == (c + 1)))
            fts.append(avg_ch)
        features.append(np.array(fts).reshape((-1, 1)))
    # 4 - 15
    # TODO Slightly relevant BUT slooooow. Replace them with something convolutional like Law energy + some pyramidal operations
    f2_tmp = []
    for c in range(len(contours)):
        fts = []
        for k in range(3):
            for m in [3, 7, 15, 25]:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
                ent_img = skrank.entropy(img_y[:, :, k], se, mask=(bw_mask == (c + 1)))
                f = np.average(ent_img, weights=(bw_mask == (c + 1)))
                fts.append(f)
        f2_tmp.append(np.array(fts).reshape((1, -1)))
    features.append(np.vstack(f2_tmp))

    # TODO Standardize the histogram for the whole image first. Most bins are empty but some are pretty relevant
    # TODO !!!  RERUN THE WHOLE EXPERIMENTS AFTERWARDS to see if there is an improvement or not !!!
    # 16 - 111
    for k in range(3):
        fts = []
        for c in range(len(contours)):
            H = cv2.calcHist([img_y], [1], (bw_mask == (c + 1)).astype(np.uint8), [32], [0, 255])
            fts.append(H.flatten())
        features.append(np.vstack(fts))

    # 112 - 118
    # shape features
    fts = []
    for c in range(len(contours)):
        M = cv2.moments(contours[c])
        Hu = cv2.HuMoments(M)
        fts.append(Hu.T)
    features.append(np.vstack(fts))

    # 119
    # TODO Highly relevant, Maybe extract a histogram or sth?
    fts = []
    for c in range(len(contours)):
        hull = cv2.convexHull(contours[c], returnPoints=False)
        defects = cv2.convexityDefects(contours[c], hull)
        def_avg = np.mean(defects[:, 0, 3]) / 256.0
        fts.append(np.array(def_avg))
    features.append(np.vstack(fts))

    features = np.hstack(features)
    # print features.shape
    return features


def cut_from_rotated_rect(img_data, rot_box, expand_ratio, newH, contour, gt_pt_array):
    # extend the bbox width by a percent
    if rot_box[1][0] > rot_box[1][1]:
        rb1 = (rot_box[1][0] * expand_ratio, rot_box[1][1] * 1.1)
    else:
        rb1 = (rot_box[1][0] * 1.1, rot_box[1][1] * expand_ratio)
    rot_box = (rot_box[0], rb1, rot_box[2])
    # Get the 'contour' points
    rot_box_pts = cv2.boxPoints(rot_box).reshape((4, 1, 2))
    # Get the box width and height
    rMinD = min(rot_box[1])
    rMaxD = max(rot_box[1])
    # New width, the height is constant,setted above
    newW = float(newH) / rMinD * rMaxD
    # 2,1,0 so the whale is not upside down
    pt_orig = rot_box_pts[[2, 1, 0], 0, :]

    # find out what is the 2'nd point coordinates
    def get_dist(pt1, pt2):
        return math.sqrt(math.pow((pt1[0] - pt2[0]), 2) + math.pow((pt1[1] - pt2[1]), 2))

    d1 = get_dist(pt_orig[0], pt_orig[1])
    d2 = get_dist(pt_orig[1], pt_orig[2])
    if d1 < d2:
        mid_point = [0, newH]
    else:
        mid_point = [newW, 0]

    # create the destination coordinates
    pt_dest = np.array([[0, 0], mid_point, [newW, newH]]).astype(np.float32)
    inv_transf = cv2.getAffineTransform(pt_dest, pt_orig)
    transf = cv2.getAffineTransform(pt_orig, pt_dest)
    x1, y1 = np.meshgrid(np.arange(newW), np.arange(newH), indexing="xy")
    coord_trans = np.dstack([x1, y1])
    coord_trans2 = cv2.transform(coord_trans, inv_transf).astype(np.float32)
    transf_img = cv2.remap(img_data, coord_trans2, None, interpolation=cv2.INTER_CUBIC)
    # Transform the contour and the 2 GT points
    if contour is not None:
        transf_contour = cv2.transform(contour, transf).astype(np.int32)
    else:
        transf_contour = None

    if gt_pt_array is not None:
        transf_gt_pts = cv2.transform(gt_pt_array, transf).astype(np.int32).reshape((2, 2)).tolist()
    else:
        transf_gt_pts = None

    return transf_img, rot_box_pts, transf_contour, transf_gt_pts



def cut_rotated_img(img_name, bbox_info, selected_idx, wd1, wd2, expand_ratio, show=False):
    """
    Cuts an image as specified by the bounding box info.


    :param img_name: Image name
    :param bbox_info: list of tuples generated by the pack_whale_candiates
    :param selected_idx: The selected bbox that will be cutted out. Established usually by ML
    :param wd1: GT dictionary of head points
    :param wd2: GT dictionary of head points
    :param expand_ratio: How much to expand the bbox in the width direction (the body is usually detected)
    :param show: true to show the resulting image
    :return: (cutted_image, rotated_contour, [pt1,pt2]) where pt1,pt2 are the ground trhuth points for the whale head.
    (pt1,pt2) is None if wd1 is None
    """

    newH = 400  # max height of the image, imagining that the whale is horizontal

    bbox_sel = bbox_info[selected_idx]
    contour = bbox_sel[2]

    img_stem = path.split(img_name)[1]
    img_data = cv2.imread(img_name)
    if wd1 is not None:
        pt1 = list(wd1[img_stem])
        pt2 = list(wd2[img_stem])
        gt_pt_array = np.array([pt1, pt2]).reshape((-1, 1, 2))

    rot_box = cv2.minAreaRect(contour)

    transf_img, rot_box_pts, transf_contour, transf_gt_pts = cut_from_rotated_rect(img_data, rot_box, expand_ratio,
                                                                                   newH, contour, gt_pt_array)

    if show:
        transf_img = np.copy(transf_img)
        cv2.drawContours(img_data, [np.int0(rot_box_pts), contour], -1, 128, 2)
        cv2.drawContours(transf_img, [transf_contour], 0, 200, 1)
        if wd1 is not None:
            cv2.line(transf_img, tuple(transf_gt_pts[0]), tuple(transf_gt_pts[1]), [0,0,255], 2)

        cv2.imshow("Original image with marked contour and whale", cv2.resize(img_data, (0, 0), fx=0.3, fy=0.3))
        cv2.imshow("Cutted out image", transf_img)

    return transf_img, transf_contour, transf_gt_pts



def select_whale_patch_using_ML(img, bbox_info, model, img_name="", show=False):
    """
    Selects a whale patch using ML


    :param img:
    :param bbox_info:
    :param model:
    :param img_name:
    :param show:
    :return: tuple with bbox_info index and a flag
    """

    flag = - 1
    return_idx = 0
    if len(bbox_info) == 1:
        # Nothing to do, there is only one option
        print img_name + " 1 patch, returning."
        flag = 0  # 1 patch, no ML schema runned
        return_idx = 0
        return (return_idx, flag)

    # Get the features for the rest
    cont_list = []
    for bbox in bbox_info:
        cont_list.append(bbox[2])
    print img_name + " Computing features for " + str(len(cont_list)) + " contours"
    features = get_patch_and_texture_features(img, cont_list)

    # predict
    labels = model.predict(features)
    cnt_positive = np.sum(labels == 1)
    pos_idxs = np.where(labels == 1)[0]
    if cnt_positive == 1:
        print img_name + " 1 patch detected at ML, returning. Total of " + str(len(cont_list)) + " contours"
        flag = 1  # multiple patches,  ML schema runned and found 1 patch
        return_idx = pos_idxs[0]
        return (return_idx, flag)
    elif cnt_positive == 0:
        print img_name + " No whale candidate was found by ML schema. Returning the largest one."
        flag = 2  # Multiple patches, no patch selected by ML. Returning the biggest one
        return_idx = 0
        return (return_idx, flag)
    else:
        # TODO Make a smarter schema, like group patches that are closer
        print img_name + " More than one candidate detected. Returning the biggest one"
        flag = 3  # Multiple patches, more than 1 patch selected by ML. Returning the biggest one selected
        return_idx = pos_idxs[0]

    return (return_idx, flag)


def select_whale_patch_using_ML_main(image_name, candidate_features, model_file, expand_ratio, dict1, dict2,show=False):
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    patch_idx, flag = select_whale_patch_using_ML(img, candidate_features, model, image_name)

    cutted_image, rotated_contour, gt_pts = cut_rotated_img(image_name, candidate_features, patch_idx, dict1, dict2,
                                                            expand_ratio,show)

    return cutted_image, rotated_contour, gt_pts


def cut_and_save_whale_given_json_rectangle(image_name,rectangle_json_info,show=False):
    """
    Cuts out a rotated rect out of the image_name.
    The rotated rect info is loaded from rectangle_json_info dictionary.


    :param image_name: The image name with path
    :param rectangle_json_info: the t*_rectangles.json information
    :return:
    """

    stem = path.split(image_name)[1]
    image_info = rectangle_json_info[stem]
    img_data = cv2.imread(image_name,cv2.IMREAD_COLOR)

    rot_box = image_info['rotated_box']
    gt_pts = np.array(image_info['gt_pts']).reshape((2,1,2))

    #Send the ground truth taken directly from Anil Thomas lists if you want other image parameters but you still
    # want the GT transformed too.
    transf_img, _, _, _ = cut_from_rotated_rect(img_data, rot_box, 1.3,400,None, None)

    if show:
        transf_img = np.copy(transf_img)
        # cv2.drawContours(img_data, [np.int0(rot_box_pts), contour], -1, 128, 2)
        # cv2.drawContours(transf_img, [transf_contour], 0, 200, 1)
        cv2.line(transf_img, tuple(gt_pts[0,0,:]), tuple(gt_pts[1,0,:]), [0,0,255], 2)

        cv2.imshow("Original image with marked contour and whale", cv2.resize(img_data, (0, 0), fx=0.3, fy=0.3))
        cv2.imshow("Cutted out image", transf_img)

    return transf_img, gt_pts