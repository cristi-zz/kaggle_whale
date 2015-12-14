__author__ = "visoft"

from src.detect_whales import *


def demo_segment_and_predict_whale(img_file, model_file,pt1_set,pt2_set):
    img = cv2.imread(img_file,cv2.IMREAD_COLOR)

    whale_patch_candidates = seg_using_histo(img,True)
    candidate_info = pack_whale_candiates(whale_patch_candidates,img)
    # Expand ratio 1.3 is used also in my pipeline.

    cutted_image, rotated_contour,gt_pts = select_whale_patch_using_ML_main(img_file,candidate_info,model_file,1.3,
                                                                            parse_anlt(pt1_set),parse_anlt(pt2_set),True)
    # You can save cutted_image and gt_pts for later reference.


def demo_load_processed_info_and_cut_out_whale(img_file,json_file_name):
    with open(json_file_name,'r') as f:
        json_data = json.load(f)

    transf_img, gt_pts = cut_and_save_whale_given_json_rectangle(img_file,json_data,True)
    # You can save cutted_image and gt_pts for later reference.




if __name__ == "__main__":

    # Source image file

    # img_file = "../../images/w_38.jpg"
    img_file = "../../images/w_199.jpg"

    model_file = "../../images/dec_tree_model.pkl"

    # https://github.com/anlthms/whale-2015
    pt1_set = "../../images/points1.json"
    pt2_set = "../../images/points2.json"

    train_rectangles = "../../images/train_rectangles.json"

    demo_segment_and_predict_whale(img_file,model_file,pt1_set,pt2_set)

    demo_load_processed_info_and_cut_out_whale(img_file,train_rectangles)

    cv2.waitKey()