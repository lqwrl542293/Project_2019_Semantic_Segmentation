import cv2
import random
import os
import numpy as np
import imageio


img_w = 512 #生成图片的宽和高

img_h = 512

ROOT_DIR = '/data/users/mzy/zyw/code/hrnet/dataset/Potsdam/train'  # 标签以及训练文件的根目录
SAVE_FOLDER = '/data/users/mzy/zyw/code/hrnet/dataset/Potsdam/train/splitdata'
TARGET_TYPE = '.png'

src_root = os.path.join(ROOT_DIR, 'src')
label_root = os.path.join(ROOT_DIR, 'label')
vis_root = os.path.join(ROOT_DIR, 'vis')


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask, mode,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if mode == 'color':
            height, width, channel = image.shape
        if mode == 'gray':
            height, width = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def data_augment(xb, yb, mode):  # xb:source;yb:label
    if np.random.random() < 0.25:  # 随机旋转
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)

    if mode != 'gray':
        # hsv空间颜色变化
        xb = randomHueSaturationValue(xb, hue_shift_limit=(-30, 30),
                                      sat_shift_limit=(-5, 5),
                                      val_shift_limit=(-15, 15))

    xb, yb = randomShiftScaleRotate(xb, yb, mode,
                                    shift_limit=(-0.1, 0.1),
                                    scale_limit=(-0.1, 0.1),
                                    aspect_limit=(-0.1, 0.1),
                                    rotate_limit=(-0, 0))  # 留作实验，测试pipeline

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:  # 沿着x轴旋转
        xb = cv2.flip(xb, 0)
        yb = cv2.flip(yb, 0)

    if (np.random.random() < 0.1) & (mode != 'gray'):
        xb = random_gamma_transform(xb, 1.0)

    # if np.random.random() < 0.25:
    #     xb = blur(xb)
    #
    # if np.random.random() < 0.2:
    #     xb = add_noise(xb)

    return xb, yb



'''
前面是数据扩增的代码，要用的时候再加上
'''

def creat_dataset(mode, src_imgae):

    if mode ==True:
        index = 0
    else:
        index = 1
    print('creating dataset...')

    src = cv2.imread(os.path.join(src_root, src_imgae))

    vis_name = src_imgae[:-8] + 'label.tif'
    vis = cv2.imread(os.path.join(vis_root, vis_name))
    label_name = src_imgae[:-8] + 'label.png'
    label = cv2.imread(os.path.join(label_root, label_name))

    src_sets = []
    vis_sets = []
    label_sets = []
    height = src.shape[0]
    width = src.shape[1]
    col = 0
    countsrc = 0
    countlab = 0
    countvis = 0
    while 1:
        row = 0
        if height - col > img_h:
            while 1:
                if width - row > img_w:
                    col_b = col + img_h
                    row_b = row + img_w
                    src_sets.append(src[col:col_b, row:row_b])
                    vis_sets.append(vis[col:col_b, row:row_b])
                    label_sets.append(label[col:col_b, row:row_b])
                else:
                    col_b = col + img_h
                    row_b = width
                    row = row_b - img_w
                    src_sets.append(src[col:col_b, row:row_b])
                    vis_sets.append(vis[col:col_b, row:row_b])
                    label_sets.append(label[col:col_b, row:row_b])
                    break #break 要减少一个缩进
                row += img_w
        else:
            while 1:

                col_b = height
                col = col_b - img_h
                if width - row > img_w:
                    row_b = row + img_h
                    src_sets.append(src[col:col_b, row:row_b])
                    vis_sets.append(vis[col:col_b, row:row_b])
                    label_sets.append(label[col:col_b, row:row_b])
                else:
                    row_b = width
                    row = row_b - img_w
                    src_sets.append(src[col:col_b, row:row_b])
                    vis_sets.append(vis[col:col_b, row:row_b])
                    label_sets.append(label[col:col_b, row:row_b])
                    break
                row = row + img_w
            break
        col += img_h


    for each in src_sets:
        filename = src_imgae[:-4]
        # filename = filename + 'IRRG'  # name of saved pics
        filename = filename + '-' + str(countsrc)
        savesrc = os.path.join(SAVE_FOLDER, 'src')
        if not os.path.isdir(savesrc):
            os.makedirs(savesrc)
        f_name = savesrc + '/'  + filename.lower() + TARGET_TYPE
        b, g, r = cv2.split(each)
        each = cv2.merge([r, g, b])
        imageio.imsave(f_name, each)
        countsrc += 1


    for each in vis_sets:
        filename = src_imgae[:-4]
        # filename = filename + 'IRRG'  # name of saved pics
        filename = filename + '-' + str(countvis)
        savevis = os.path.join(SAVE_FOLDER, 'vis')
        if not os.path.isdir(savevis):
            os.makedirs(savevis)
        f_name = savevis + '/'  + filename.lower() + TARGET_TYPE
        b, g, r = cv2.split(each)
        each = cv2.merge([r, g, b])
        imageio.imsave(f_name, each)
        countvis += 1

    # count = 0
    for each in label_sets:
        filename = src_imgae[:-4]
        filename = filename + '-' + str(countlab)
        savelabel = os.path.join(SAVE_FOLDER, 'label')
        if not os.path.isdir(savelabel):
            os.makedirs(savelabel)
        f_name = savelabel + '/' + filename.lower() + TARGET_TYPE
        b, g, r = cv2.split(each)
        each = cv2.merge([r, g, b])
        imageio.imsave(f_name, each)
        countlab += 1
    print(countlab, countsrc, countvis)

def cutImg(self, source, target, patch_size=1024):  # source->src target->cut
    test_sets = os.listdir(source)
    original_shape = {}
    for i, image_name in enumerate(test_sets):  # randomly choose
        cut_image_folder = target + image_name[:-4] + '/'  # test/cut/.../
        if not os.path.exists(cut_image_folder):
            os.mkdir(cut_image_folder)
        img = cv2.imread(source + image_name)
        height, width, channel = img.shape  # attention!
        original_shape[image_name[:-4]] = img.shape  # should take control of the order
        image = np.zeros(shape=(height, width, channel))
        image[0:height, 0:width, :] = img
        start_width = 0
        end_width = start_width + patch_size
        start_height = 0
        end_height = start_height + patch_size

        count = 0
        while end_width <= width:
            while end_height <= height:
                patch = image[start_height:end_height, start_width:end_width, :]
                # print(count)
                cv2.imwrite(cut_image_folder + str(count) + '.png', img=patch.astype(np.uint8))
                # print(start_height, end_height, start_width, end_width)
                count += 1
                if (end_height == height):
                    break

                start_height += patch_size
                end_height = start_height + patch_size

                if end_height > image.shape[0]:
                    start_height = height - patch_size
                    end_height = height
            if(end_width ==  width):
                break

            start_width += patch_size
            end_width = end_width + patch_size
            if end_width > width:
                end_width = width
                start_width = width - patch_size
            # print(start_width, end_width)

            start_height = 0
            end_height = start_height + patch_size
        print('patches count:', count)
    print('original_shape_dict:', original_shape)
    return original_shape

if __name__=='__main__':
    # print(src_root)
    src = os.listdir(src_root)
    for src_imgae in src:
        print(src_imgae)
        # creat_dataset(False, src_imgae)
        cutImg(src_root, )

    # creat_dataset(mode=True)