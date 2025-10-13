import numpy as np
import cv2
from scipy import ndimage
import sys
import os 
from resize_utils import image_as_png
        
        
def crop_val(v, minv, maxv):
    '''This function guarantees that patches are 
    within the image'''
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v


def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

    
def overlap_patch_roi(patch_center, patch_size, roi_mask, 
                      add_val=1000, cutoff=.9):
    
    '''This function returns true if the patch satisfies
    the minimum overlapping area with the region of interest'''
    x1,y1 = (patch_center[0] - int(patch_size/2), 
             patch_center[1] - int(patch_size/2))
    x2,y2 = (patch_center[0] + int(patch_size/2), 
             patch_center[1] + int(patch_size/2))
    x1 = crop_val(x1, 0, roi_mask.shape[1])
    y1 = crop_val(y1, 0, roi_mask.shape[0])
    x2 = crop_val(x2, 0, roi_mask.shape[1])
    y2 = crop_val(y2, 0, roi_mask.shape[0])
    
    roi_area = (roi_mask>0).sum()
    roi_patch_added = roi_mask.copy()
    roi_patch_added = roi_patch_added.astype('uint16')
    roi_patch_added[y1:y2, x1:x2] += add_val
    patch_area = (roi_patch_added>=add_val).sum()
    inter_area = (roi_patch_added>add_val).sum().astype('float32')
   
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)


def sample_patches(img, index, label, roi_mask, folder_s, folder_s10,
                   list_s, list_s10, patch_size=224,
                   pos_cutoff=0.9, neg_cutoff=0.10,
                   nb_bkg=11, nb_abn=10, start_sample_nb=0,
                   verbose=True):
    """
    This function generates the s and the s10 patch image datasets.
      - S: one patch centered on the ROI and one random background patch
      - S10: ten patches around the ROI (high-overlap) plus corresponding background patches
    """

    index = int(index)
    print(index)

    # 1) Verify shapes
    if img.shape != roi_mask.shape:
        raise Exception("mask and img have different shape, check input data")
    print("img and mask shape match")

    # 2) Pad image and mask for safe extraction
    half = patch_size // 2
    img_p = add_img_margins(img, half)
    mask_p = add_img_margins(roi_mask, half).astype("uint8")

    # 3) Threshold to binary
    _, thresh = cv2.threshold(mask_p, 254, 255, cv2.THRESH_BINARY)

    # 4) Find contours
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 5) If no ROI contour, skip ROI-based sampling
    if not contours:
        if verbose:
            print(f"No contours found for index {index}; skipping ROI patches.")
    else:
        # 6) Compute largest contour bounding box and centroid
        cont_areas = [cv2.contourArea(c) for c in contours]
        idx = np.argmax(cont_areas)
        rx, ry, rw, rh = cv2.boundingRect(contours[idx])
        cy, cx = ndimage.measurements.center_of_mass(thresh)
        cx, cy = int(round(cx)), int(round(cy))
        print("ROI centroid=", (cx, cy)); sys.stdout.flush()

        # 7) Extract the S patch (ROI-centered)
        x1, y1 = crop_val(cx - half, 0, mask_p.shape[1]), crop_val(cy - half, 0, mask_p.shape[0])
        x2, y2 = crop_val(cx + half, 0, mask_p.shape[1]), crop_val(cy + half, 0, mask_p.shape[0])
        s_patch = img_p[y1:y2, x1:x2].astype("uint16")
        path_s = os.path.join(folder_s, f"roi_{index}.png")
        image_as_png(s_patch, path_s)
        list_s.append({"path": path_s, "label": label})

        # 8) Sample abnormal patches around ROI
        rng = np.random.RandomState(321)
        sampled_abn = 0
        nb_try = 0
        while sampled_abn < nb_abn:
            x = rng.randint(rx, rx + rw)
            y = rng.randint(ry, ry + rh)
            nb_try += 1

            if overlap_patch_roi((x, y), patch_size, mask_p, cutoff=pos_cutoff):
                patch = img_p[y - half : y + half, x - half : x + half].astype("uint16")
                path_a = os.path.join(folder_s10, f"roi_{index}_{sampled_abn}.png")
                image_as_png(patch, path_a)
                list_s10.append({"path": path_a, "label": label})
                sampled_abn += 1
                nb_try = 0
                if verbose:
                    print(f"sampled an abn patch at center={(x,y)}")

            if nb_try > 1000:
                pos_cutoff -= 0.05
                nb_try = 0
                if pos_cutoff <= 0:
                    raise Exception("Overlap cutoff too low; check ROI mask input.")

    # 9) Sample background patches for both S and S10
    rng = np.random.RandomState(321)
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        x = rng.randint(patch_size, img_p.shape[1] - patch_size)
        y = rng.randint(patch_size, img_p.shape[0] - patch_size)

        if not overlap_patch_roi((x, y), patch_size, mask_p, cutoff=neg_cutoff):
            bkg = img_p[y - half : y + half, x - half : x + half].astype("uint16")
            if sampled_bkg == 0:
                path_b = os.path.join(folder_s, f"bkg_{index}.png")
                list_s.append({"path": path_b, "label": 0})
            else:
                path_b = os.path.join(folder_s10, f"bkg_{index}_{sampled_bkg}.png")
                list_s10.append({"path": path_b, "label": 0})
            image_as_png(bkg, path_b)
            sampled_bkg += 1
            if verbose:
                print(f"sampled a bkg patch at center={(x,y)}")

    return list_s, list_s10 
