import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os

ROOT_DIR = '/data/users/mzy/zyw/code/hrnet/dataset/Potsdam2/train/vis'
Label_Dir = '/data/users/mzy/zyw/code/hrnet/dataset/Potsdam2/train/label/'

if not os.path.isdir(Label_Dir):
        os.makedirs(Label_Dir)

def get_isprs_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (6, 3)
    """
    return np.asarray([[255, 255, 255],  # Impervious surfaces 0
                       [0, 0, 255],  # Building 5
                       [0, 255, 255],  # Low vegetation 4
                       [0, 255, 0],  # Tree 3
                       [255, 255, 0],  # Car 2
                       [255, 0, 0],  # Clutter/background 1
                       ])

def encode_segmap(mask):
    """Encode segmentation label images as classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (6000, 6000, 3), in which the classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_isprs_labels()):
        print(ii, label)
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """


    n_classes = 6
    label_colours = get_isprs_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()



def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou



if __name__=='__main__':

    rgbmasks = os.listdir(ROOT_DIR)
    for rgblabel in rgbmasks:
        print(rgblabel)
        if rgblabel[-3:] == 'tif':
            rgb = cv2.imread(os.path.join(ROOT_DIR, rgblabel))
            img = encode_segmap(rgb)
            print(img.shape)
            cv2.imwrite(Label_Dir+rgblabel[:-4]+'.png', img)
            print(rgblabel, " wirten down")

    # reslut = "/data/users/mzy/zyw/code/hrnet/dataset/Potsdam/train/label"
    # reslutlabel = os.listdir(reslut)
    # for each in reslutlabel:
    #     # if len(each) == 25:
    #
    #     img = cv2.imread(os.path.join(reslut, each))
    #     print(img.shape)
    #     resultz = np.zeros(shape=img.shape)
    #     channel0 = np.where(img[:, :, 0] == 0, 255, 0)  # blue
    #     resultz[:, :, 0] = channel0
    #     resultz[:, :, 1] = channel0
    #     resultz[:, :, 2] = channel0
    #     channel1 = np.where(img[:, :, 0] == 1, 255, 0)  # blue
    #     resultz[:, :, 2] = resultz[:, :, 2] + channel1
    #
    #     channel2 = np.where(img[:, :, 0] == 2, 255, 0)  # blue
    #     resultz[:, :, 1] = resultz[:, :, 1] + channel2
    #     resultz[:, :, 2] = resultz[:, :, 2] + channel2
    #
    #     channel3= np.where(img[:, :, 0] == 3, 255, 0)  # blue
    #     resultz[:, :, 1] = resultz[:, :, 1] + channel3
    #
    #     channel4 = np.where(img[:, :, 0] == 4, 255, 0)  # blue
    #     resultz[:, :, 0] = resultz[:, :, 0]+ channel4
    #     resultz[:, :, 1] = resultz[:, :, 1] + channel4
    #
    #     channel5 = np.where(img[:, :, 0] == 5, 255, 0)  # blue
    #     resultz[:, :, 0] = resultz[:, :, 0]+channel5
    #     # resultz[:, :, 2] = channel3
    #     # resultz[:, :, 2] = channel1
    #     # channel2 = np.where(img[:, :, 0] == 4, 255, 0)  # blue
    #         # resultz[:, :, 1] = channel2
    #         # cv2.imwrite(Label_Dir+each[:-4]+'.png', resultz.astype(np.uint8))
    #
    #         # print("")
    #         # print(img.shape)
    #         # img = decode_segmap(img, True)
    #         # print(img.shape)
    #     cv2.imwrite(Label_Dir+each[:-4]+'.png', resultz)
    #     print(each, " wirten down")
