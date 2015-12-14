__author__ = "visoft"

from src.detect_whales import *


def test_seg_using_histo():
    """
    First step in whale detection.

    :return:
    """

    img_file = "../../images/w_199.jpg"
    img = cv2.imread(img_file,cv2.IMREAD_COLOR)
    assert img is not None

    #Change it to True and uncomment the waitkey to see the results step by step
    whale_patch_candidates = seg_using_histo(img,False)
    # cv2.waitKey()

    #This candidate info is later used to classify the acutal whale
    candidate_info = pack_whale_candiates(whale_patch_candidates,img)

    assert whale_patch_candidates is not None
    assert candidate_info is not None
    assert len(candidate_info) > 0 # A list of candidate contours

    assert len(candidate_info[0][0]) == 4 # The rectangular bounding box
    assert candidate_info[0][1] > 1 # The contour area
    assert candidate_info[0][2].shape[0] > 1 # The actual contour



def test_predict_candidates_using_ml():
    img_file = "../../images/w_199.jpg"
    model_file = "../../images/point_model.pkl"

    pt1_set = "../../images/points1.json"
    pt2_set = "../../images/points2.json"

    img = cv2.imread(img_file,cv2.IMREAD_COLOR)
    assert img is not None

    whale_patch_candidates = seg_using_histo(img,False)
    candidate_info = pack_whale_candiates(whale_patch_candidates,img)
    assert candidate_info is not None

    cutted_image, rotated_contour,gt_pts = select_whale_patch_using_ML_main(img_file,candidate_info,model_file,1.3,
                                                                            parse_anlt(pt1_set),parse_anlt(pt2_set))

    assert cutted_image is not None
    assert  rotated_contour is not None
    assert gt_pts is not None

if __name__ == "__main__":
    pass