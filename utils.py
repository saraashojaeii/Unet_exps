import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize



def apply_otsu_thresholding(tensor, is_mask=True):
    """
    Applies Otsu's thresholding to a single-channel network output tensor.
    
    Parameters:
    - tensor: A 2D or 3D (1 channel) PyTorch tensor of network outputs, with values in [0, 1].
    - is_mask: A boolean indicating if the tensor is a mask. If False, assumes it is a contour.
    
    Returns:
    - A binary tensor with the same shape as the input, where each element is 0 or 1.
    """
    # Ensure the tensor is on CPU and convert it to a numpy array
    tensor_np = tensor.cpu().detach().numpy()
    
    # Handle both single image and batch of images
    if tensor_np.ndim == 3:  # Batch of images
        result = []
        for img in tensor_np:
            # Convert the probabilities to a suitable format for Otsu's thresholding
            img_scaled = (img * 255).astype(np.uint8)
            th, binary_img = cv2.threshold(img_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Convert back to tensor and append to results
            result.append(torch.tensor(binary_img / 255.0).float())
        return torch.stack(result)  # Stack to get a batch tensor back
    else:  # Single image
        tensor_np_scaled = (tensor_np * 255).astype(np.uint8)
        th, binary_np = cv2.threshold(tensor_np_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

def iou(y_pred, y_true):
    
    # y_pred = apply_otsu_thresholding(y_pred, is_mask=True)
    # y_pred = torch.sigmoid(y_pred)
    # y_pred = (y_pred > 0.5).float()
    th = apply_otsu_thresholding(y_pred, is_mask=True)
    # y_pred = y_pred.to(device)
    # y_true = y_true.to(device)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > th).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return intersection / (union + 1e-6)

def f1_score(y_pred, y_true):
    
    th = apply_otsu_thresholding(y_pred, is_mask=True)
    # y_pred = y_pred.to(device)
    # y_true = y_true.to(device)
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > th).float()
    tp = (y_true * y_pred).sum().item()
    precision = tp / (y_pred.sum().item() + 1e-6)
    recall = tp / (y_true.sum().item() + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int32)
    return tp


def relaxed_f1(pred, gt, buffer):
    ''' Usage and Call
    # rp_tp, rr_tp, pred_p, gt_p = relaxed_f1(predicted.cpu().numpy(), labels.cpu().numpy(), buffer = 3)

    # rprecision_tp += rp_tp
    # rrecall_tp += rr_tp
    # pred_positive += pred_p
    # gt_positive += gt_p

    # precision = rprecision_tp/(gt_positive + 1e-12)
    # recall = rrecall_tp/(gt_positive + 1e-12)
    # f1measure = 2*precision*recall/(precision + recall + 1e-12)
    # iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
    '''

    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    # for b in range(pred.shape[0]):
    pred_sk = skeletonize(pred)
    gt_sk = skeletonize(gt)
        # pred_sk = pred[b]
    # gt_sk = gt[b]

    #The correctness represents the percentage of correctly extracted road data, i.e., the percentage
    #of the extracted data which lie within the buffer around the reference network (groudn truth):
    rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)

    #The completeness is the percentage of the reference data which is explained by the extracted
    #data, i.e., the percentage of the reference network which lie within the buffer around the
    #extracted data (prediction):
    rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
    pred_positive += len(np.where(pred_sk == 1)[0])
    gt_positive += len(np.where(gt_sk == 1)[0])

    #Correctness corresponds to relaxed precision
    #Completeness corresponds to relaxed recall 
    #Quality corresponds to intersection-over-union

    comm= rrecall_tp/(gt_positive + 1e-12) #length of matched reference/ length of reference
    corr= rprecision_tp/(pred_positive + 1e-12)   #length of matched extraction/ length of extraction
    qul = (comm*corr )/(comm- (comm*corr) + corr+ 1e-12)
    return comm*100, corr*100, qul*100
