from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# predicted_label = np.load('/home/dell/disk/Jinlong/Time-Series-Library-main/results/test_iTransformer_TSF_ftS_sl255_ll100_pl255_dm64_nh8_el2_dl1_df64_fc1_ebtimeF_dtTrue_destest_isvmdFalse_attnmaskFalse_att_mask_noneTrue_embedding_flagFalse/FFT_predseismicfacies.npy')
# # predicted_label = np.load('/home/dell/disk/Jinlong/faciesdata/train_labels.npy')
# true_label_all = np.load('/home/dell/disk/Jinlong/faciesdata/train_labels.npy')
# gt_labels = true_label_all.reshape(-1, 255)
# lendataset = len(gt_labels) 
# proportion = int(lendataset*0.2)
# true_label_all = gt_labels[:proportion, :]

# pred_label_all_cm = predicted_label.flatten()
# true_label_all_cm = true_label_all.flatten()
# pred_label_all_cm = true_label_all_cm
# classes = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5']

# cm = confusion_matrix(true_label_all_cm, pred_label_all_cm, labels=range(len(classes)))
# class_counts = np.sum(cm, axis=1)
# normalized_cm = cm / class_counts[:, np.newaxis]
# fig, ax = plt.subplots()
# im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(len(classes)),
#         yticks=np.arange(len(classes)),
#         xticklabels=classes,
#         yticklabels=classes,
#         title='Confusion Matrix',
#         ylabel='True Label',
#         xlabel='Predicted Label')

# # Add text annotations to the confusion matrix plot
# thresh = cm.max() / 2.
# for i in range(len(classes)):
#     for j in range(len(classes)):
#         ax.text(j, i, format(cm[i, j], 'd'),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")

# fig.tight_layout()
# ax.figure.colorbar(im, ax=ax)
# lt.psavefig(r"C:\Users\Administrator\Desktop\SegFormer_matrix.png", dpi=500)
# plt.show()


# """calculate the class accuracy matrix"""
# import numpy as np

def calculate_class_accuracy(gt_labels, pred_labels, num_classes):
    """
    Calculate class accuracy between ground truth and predicted labels.
    
    Args:
        gt_labels: Ground truth labels (numpy array).
        pred_labels: Predicted labels (numpy array).
        num_classes: Number of classes.
        
    Returns:
        Class accuracy for each class.
    """
    class_accuracies = np.zeros(num_classes)

    for class_idx in range(num_classes):
        class_indices = (gt_labels == class_idx)
        total_pixels_class = np.sum(class_indices)
        pred_labels_class = pred_labels[class_indices]
        correct_predictions_class = np.sum(pred_labels_class == class_idx)
        if total_pixels_class > 0:
            class_accuracies[class_idx] = correct_predictions_class / total_pixels_class
        else:
            class_accuracies[class_idx] = np.nan  

    return class_accuracies


def calculate_fwiu(gt_labels, pred_labels, num_classes):
    """
    Calculate Frequency Weighted Intersection over Union (FWIU).
    
    Args:
        gt_labels: List of ground truth masks (numpy arrays).
        pred_labels: List of predicted masks (numpy arrays).
        num_classes: Number of classes.
        
    Returns:
        Frequency Weighted Intersection over Union (FWIU).
    """
    class_counts = np.zeros(num_classes)
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for gt_label, pred_label in zip(gt_labels, pred_labels):
        for i in range(num_classes):
            gt_class = gt_label == i
            pred_class = pred_label == i
            intersection[i] += np.logical_and(gt_class, pred_class).sum()
            union[i] += np.logical_or(gt_class, pred_class).sum()
            class_counts[i] += gt_class.sum()

    fwiu = (intersection / union) * (class_counts / class_counts.sum())
    fwiu = np.nan_to_num(fwiu)  # Replace NaN values with 0
    return np.sum(fwiu)


def calculate_iou_per_class(gt_mask, pred_mask, class_label):
    """
    Calculate Intersection over Union (IoU) for a specific class label.
    
    Args:
        gt_mask: Ground truth mask (numpy array).
        pred_mask: Predicted mask (numpy array).
        class_label: Class label for which IoU is calculated.
        
    Returns:
        Intersection over Union (IoU) for the specified class label.
    """
    # Binary masks for the specific class
    gt_class = gt_mask == class_label
    pred_class = pred_mask == class_label
    pred_class = np.squeeze(pred_class)
    # Intersection and union for the specific class
    intersection = np.logical_and(gt_class, pred_class)
    union = np.logical_or(gt_class, pred_class)
    
    # IoU for the specific class
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_mean_iou(gt_masks, pred_masks, num_classes):
    """
    Calculate mean Intersection over Union (mIoU) across multiple samples and classes.
    
    Args:
        gt_masks: List of ground truth masks (numpy arrays).
        pred_masks: List of predicted masks (numpy arrays).
        num_classes: Number of classes.
        
    Returns:
        Mean Intersection over Union (mIoU).
    """
    class_iou = np.zeros(num_classes)
    num_samples = len(gt_masks)
    
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        for class_label in range(num_classes):
            class_iou[class_label] += calculate_iou_per_class(gt_mask, pred_mask, class_label)
    
    mean_iou = np.mean(class_iou / num_samples)
    return mean_iou


# Example usage:
if __name__ == "__main__":
    # Example ground truth and predicted labels (replace with your data)
    # pred_labels = np.load('/home/dell/disk/Jinlong/Time-Series-Library-main/nzf_dformer_trace_predictions.npy')
    pred_labels = np.load('/home/dell/disk/Jinlong/Time-Series-Library-main/results/3_30_NZ_VMD_MAT_train_only_and_test_Train_iTransformer_TSF_ftS_sl255_ll100_pl255_dm64_nh8_el2_dl1_df64_fc1_ebtimeF_dtTrue_destest_isvmdTrue_attnmaskFalse_att_mask_noneTrue_embedding_flagTrue/4_1_MAT_vmd_nz_pred_seismic_facies.npy')

    # pred_labels = np.load('/home/dell/disk/Jinlong/Time-Series-Library-main/results/Train_BiLSTM_TSF_ftS_sl255_ll100_pl255_dm64_nh8_el2_dl1_df64_fc1_ebtimeF_dtTrue_destest_isvmdFalse_attnmaskFalse_att_mask_noneTrue_embedding_flagFalse/FFT_predseismicfacies.npy')
    
    # gt_labels = np.load('/home/dell/disk/Jinlong/faciesdata/train_labels.npy')
    # gt_labels = np.load('/home/dell/disk/Jinlong/faciesdata/labels.npy')
    gt_labels = np.load('/home/dell/disk/Jinlong/faciesdata/330NZ_VMD_facies_randomlu_selected_100_selectedlabel.npy')
    
    
    # gt_labels = np.swapaxes(gt_labels, 1,0)
    # gt_labels = np.swapaxes(gt_labels, -1,1)
    # gt_labels = gt_labels.reshape(-1, 1006)
    lendataset = len(gt_labels) 
    proportion = int(lendataset*0.1)
    gt_labels = gt_labels[:proportion, :]

    num_classes = 6  # Number of classes in your dataset
    # 56080 80
    # Calculate class accuracy
    
    class_accuracies = calculate_class_accuracy(gt_labels.flatten(), pred_labels.flatten(), num_classes)

    class_accuracies_str = " ".join(["{:.3f}".format(acc) for acc in class_accuracies])

    print("Class accuracies:", class_accuracies_str)

    fwiu = calculate_fwiu(gt_labels, pred_labels, num_classes)

    print("FWIU: {:.3f}".format(fwiu))

    # mean iou needs the array to be a list
    mean_iou = calculate_mean_iou([gt_labels],[pred_labels], num_classes)

    print("Mean IoU: {:.3f}".format(mean_iou))

"""calculate the mean intersection over union"""



