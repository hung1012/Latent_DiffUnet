import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  

IMAGE_DIR = '/home/admin_mcn/thaotlp/data/ISIC/image'
MASK_DIR = '/home/admin_mcn/thaotlp/data/ISIC/mask'
OUTPUT_DIR = '/home/admin_mcn/thaotlp/output/test'
LATENT_DIR = '/home/admin_mcn/thaotlp/output/latent_gt'
COMPARE_DIR = '/home/admin_mcn/thaotlp/output/compare'

def visualize(metric, **images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.suptitle("IoU score:" + "{:.4f}".format(metric[0]) + ", Dice score:" + "{:.4f}".format(metric[1]), fontsize=24)
        plt.title(name.replace('_',' ').title(), fontsize=18)
        plt.imshow(image)
    plt.savefig(COMPARE_DIR + '/' + str(fileidx) + '.png')


def calculate_iou(predicted_mask, mask):
    intersection = np.logical_and(mask, predicted_mask)
    union = np.logical_or(mask, predicted_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_dice_score(mask, predicted_mask):
    intersection = np.logical_and(mask, predicted_mask)
    union = np.logical_or(mask, predicted_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(intersection) + np.sum(union))
    return dice_score

mean_iou = 0
mean_dice = 0

for filename in os.listdir(OUTPUT_DIR):
    metrics = []
    fileidx = filename.split('.')[0]

    image = cv2.imread(os.path.join(IMAGE_DIR, fileidx + '.jpg'), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(MASK_DIR, fileidx + '.jpg'), cv2.IMREAD_GRAYSCALE)
    latent = cv2.imread(os.path.join(LATENT_DIR, filename), cv2.IMREAD_GRAYSCALE)
    output = cv2.imread(os.path.join(OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
    
    resized_latent = cv2.resize(latent, (image.shape[1], image.shape[0])) 
    resized_output = cv2.resize(output, (image.shape[1], image.shape[0])) 
        
    iou_score = calculate_iou(resized_output, mask)
    dice_score = calculate_dice_score(resized_output, mask)

    metrics.append(iou_score)
    metrics.append(dice_score)
    mean_iou += iou_score
    mean_dice += dice_score

    visualize(
        metric = metrics,
        original_image = image,
        gt_mask = mask,
        latent_reconstruction = resized_latent,
        predicted_mask = resized_output,
    )

print("The average IoU score:", "{:.5f}".format(mean_iou / len(os.listdir(OUTPUT_DIR))))
print("The average Dice score:", "{:.5f}".format(mean_dice / len(os.listdir(OUTPUT_DIR))))

