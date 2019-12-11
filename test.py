import torch
from torch.autograd import Variable as V
from skimage.morphology import remove_small_objects
import torch.nn.functional as F
import cv2
import os
import numpy as np
from time import time
import argparse
from models.seg_hrnet import HighResolutionNet
from models.ccnet import ResNet
from config import config
from config import update_config
from sklearn import metrics
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args



class TTAFrame():
    def __init__(self, net,  model, label_list, config):
        if model == 'ccnet':
            self.net = eval('models.' + config.MODEL.NAME + '.Seg_Model')(
            num_classes=6).cuda()
        else:
            self.net = net(config).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.label_list = label_list
        self.model = model
    def test_one_img_from_path(self, path,  evalmode=True):
        if evalmode:
            self.net.eval()
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        print("test_one_img_from_path", img.shape)
        # if self.model == 'HRNet':
        #     img = img[:, :, 0:1]
        # img = img[:, :, 0:1]
        # 8次TTA 水平垂直对称加旋转
        img90 = np.array(np.rot90(img))  # np.rot90 是旋转前两个维度
        print('img90', img90.shape)
        img1 = np.concatenate([img[None], img90[None]])
        print('img1', img1.shape)
        img2 = np.array(img1)[:, ::-1]
        print('img2', img2.shape)
        img3 = np.concatenate([img1, img2])
        print('img3', img3.shape)
        img4 = np.array(img3)[:, :, ::-1]
        print('img4', img4.shape)
        img5 = img3.transpose(0, 3, 1, 2)
        print('img5', img5.shape)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        print('img5', img5.shape)
        img6 = V(torch.Tensor(img6).cuda())
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        print('maska', maska.shape)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        print('maskb', maskb.shape)
        mask1 = maska + maskb[:, :, :, ::-1]
        print('mask1', mask1.shape)
        mask2 = mask1[:2] + mask1[2:, :, ::-1]
        print('mask2', mask2.shape)
        mask3 = mask2[0] + np.rot90(mask2[1].transpose(1, 2, 0)).transpose(2, 0, 1)[:, ::-1, ::-1]
        print('mask3', mask3.shape)
        # softmax probability output [num_channels, 1024, 1024]
        if self.model == 'HRNet':
            # interpolate
            mask3 = torch.from_numpy(mask3).unsqueeze(0)
            print('mask3', mask3.shape)
            mask3 = torch.nn.functional.upsample(mask3, scale_factor=4, mode='bilinear').squeeze(0).numpy()
            print('mask3', mask3.shape)
        else:
            mask3 = torch.from_numpy(mask3)
            mask3 = F.softmax(mask3, dim=0)
            mask3 = mask3.numpy()
        return mask3

    def load(self, path):
        print(path)
        self.net.load_state_dict(torch.load(path, map_location='cuda:0'))
        # if the network is trained in multi-gpu, you should add map_location when doing test.
        #self.net.load_state_dict(torch.load(path))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def computeIoU(self, label_list=[], path1='', path2=''):
        union_sum = 0
        intersection_sum = 0
        IoU_list_class = []
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        for i in range(len(label_list)):
            i = label_list[i]
            temp1 = np.where(img1 == i, 1, 2)
            temp2 = np.where(img2 == i, 1, 3)
            intersection = np.sum(np.where(temp1 == temp2, 1, 0))
            intersection_sum += intersection  # 交集
            temp1 = np.where(img1 == i, 1, 0)
            temp2 = np.where(img2 == i, 1, 0)
            union = temp1 + temp2
            union[union > 1] = 1
            union = np.sum(union)
            union_sum += union  # 并集
            IoU_list_class.append(intersection / union)
            print('class ' + str(i) + ' IoU:', intersection / union)
        IoU = intersection_sum / union_sum
        return IoU, IoU_list_class

    def testcutImg(self, source, target, patch_size=config.TEST.IMAGE_SIZE):  # source->src target->cut
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

    def stitch_patches(self, source, target,  original_shape, patch_size=config.TEST.IMAGE_SIZE):
        '''
        :param source: 切割好的图片的文件夹
        :param target: 合成好的学出来的图片
        :param original_shape:
        :param patch_size:
        :return:
        '''
        # print(source)
        datasets = os.listdir(source)
        result = np.zeros(original_shape)
        for i, image in enumerate(datasets):
            # print(image)
            patch = cv2.imread(source + image)
            num = int(image[:-4])
            # print(num)
            (height, width, channel) = original_shape

            margin = int(height / patch_size) + 1  # height/patch_size
            # print(margin)
            start_width = (num // margin ) * patch_size  # //取整
            # print(start_width)
            end_width = start_width + patch_size
            # print(end_width)
            if end_width > width:
                end_width = width
                start_width = end_width - patch_size

            start_height = (num % margin) * patch_size
            # print(start_height)
            end_height = start_height + patch_size
            # print(end_height)
            if end_height > height:
                end_height = height
                start_height = end_height - patch_size

            # print(start_height,end_height, start_width,end_width)
            result[start_height:end_height, start_width:end_width, :] = patch
        result = result[0:original_shape[0], 0:original_shape[1], :]
        cv2.imwrite(target, img=result.astype(np.uint8))
        print("Stitch over!!!")

    def visualise(self, source1, source2, target, label):
        img1 = cv2.imread(source1)
        img2 = cv2.imread(source2)
        result = np.zeros(shape=img1.shape)
        channel1 = np.where(img1[:, :, 0] == label, 255, 0)  # blue
        channel2 = np.where(img2[:, :, 1] == label, 255, 0)  # green
        result[:, :, 0] = channel1
        result[:, :, 1] = channel2
        cv2.imwrite(target, result.astype(np.uint8))
        print('visualise over!')

    def test_one_image(self, path):
        img = cv2.imread(path)
        img = V(torch.Tensor(img).cuda())
        img = img.unsqueeze(0)

        img = img.cpu().data.numpy().transpose(0, 3, 1, 2)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        # print(img.shape)
        # print(self.net.forward(img)[0].shape)
        # print(self.net.forward(img)[1].shape)
        if isinstance(self.net.forward(img), list):

            mask = self.net.forward(img)[0].squeeze().cpu().data.numpy()
            # print(mask.shape)
        else:
            mask = self.net.forward(img).squeeze().cpu().data.numpy()
            # print(mask.shape)
        mask = torch.from_numpy(mask).unsqueeze(0)

        mask = torch.nn.functional.upsample(input=mask, size=(1024, 1024), mode='bilinear').squeeze(0).numpy()
        # exit()
        return mask

    def metrics(self, predictions, gts, label_list):
        """ Compute the metrics from the RGB-encoded predictions and ground truthes
        Args:
            predictions (array list): list of RGB-encoded predictions (2D maps)
            gts (array list): list of RGB-encoded ground truthes (2D maps, same dims)
        """
        prediction_labels = np.concatenate([predictions.flatten()])
        gt_labels = np.concatenate([gts.flatten()])

        cm = metrics.confusion_matrix(
            gt_labels,
            prediction_labels,
            range(len(label_list)))

        # print("Confusion matrix :")
        # print(cm)
        # print("---")
        # Compute global accuracy
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        total = sum(sum(cm))
        oa = accuracy * 100 / float(total)
        # print("{} pixels processed".format(total))
        # print("Total accuracy : {}%".format(accuracy * 100 / float(total)))
        # print("---")
        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe)
        # print("Kappa: " + str(kappa))
        return kappa, oa

    def visall(self, path, name):
        '''result -> result path'''
        # reslut = "/data/users/mzy/zyw/code/hrnet/dataset/Potsdam/train/label"
        # reslutlabel = os.listdir(result)
        # for each in reslutlabel:
            # if len(each) == 25:

        img = cv2.imread(path)
        print(img.shape)
        resultz = np.zeros(shape=img.shape)
        channel0 = np.where(img[:, :, 0] == 0, 255, 0)
        resultz[:, :, 0] = channel0
        resultz[:, :, 1] = channel0
        resultz[:, :, 2] = channel0
        channel1 = np.where(img[:, :, 0] == 1, 255, 0)
        resultz[:, :, 2] = resultz[:, :, 2] + channel1

        channel2 = np.where(img[:, :, 0] == 2, 255, 0)
        resultz[:, :, 1] = resultz[:, :, 1] + channel2
        resultz[:, :, 2] = resultz[:, :, 2] + channel2

        channel3 = np.where(img[:, :, 0] == 3, 255, 0)
        resultz[:, :, 1] = resultz[:, :, 1] + channel3

        channel4 = np.where(img[:, :, 0] == 4, 255, 0)
        resultz[:, :, 0] = resultz[:, :, 0] + channel4
        resultz[:, :, 1] = resultz[:, :, 1] + channel4

        channel5 = np.where(img[:, :, 0] == 5, 255, 0)
        resultz[:, :, 0] = resultz[:, :, 0] + channel5

        print(cv2.imwrite(source + 'multilabel0/' + name + '.png', resultz))

        print(name,"wirten down")

    def test(self, source,  label_list, split):
        # test_images cut
        path_src = source + 'src/'
        path_cut = source + 'cut/'

        if not os.path.exists(path_cut):
            os.mkdir(path_cut)


        if split:
            original_shape = (6000, 6000, 3)
        else:
            original_shape = self.testcutImg(path_src, path_cut)

        # original_shape, complete_shape = self.testimg(path_src, path_cut)
        # print(original_shape)
        # print(complete_shape)
        # test cut_images
        path_results = source + 'results/'
        if not os.path.exists(path_results):
            os.mkdir(path_results)

        # if os.path.exists(txt_path):
        #     os.remove(txt_path)

        path_results_stitch = path_results + 'stitch/'
        if not os.path.exists(path_results_stitch):
            os.mkdir(path_results_stitch)
        list_cut = os.listdir(path_cut)  # 原图片剪切好的文件夹集合
        tic = time()
        kappa = []
        miou = []
        oa = []
        # pred_label_list = []
        # gt_label_list = []
        text = ''
        if config.train:
            for i, folder in enumerate(list_cut):  # folder是原图片的名称，去了后标的

                target_folder = path_results
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                target_folder = target_folder + folder + '/'  # 每张图片建立了一个result文件夹，放测试出来的图片
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                source_folder = path_cut + folder + '/'  # test/cut/top_Potsdam_... 原图片剪切好的文件夹
                source_list = os.listdir(source_folder)  # 剪切好的每一张图片
                # print('source_list: ',source_list)
                for i, name in enumerate(source_list):
                    # print(name)
                    if i % 10 == 0:
                        print(source_folder, i, 'th', '%.2f' % (time() - tic), 's')
                    mask = self.test_one_image(source_folder + name)  # 测试每张剪切好的图片得到他的mask
                    mask = np.argmax(mask, axis=0)  # 获取每个类可能性最大的即其标签
                    # 1024*1024 -> 768*768
                    # mask = mask[128:896, 128:896]
                    # mask = mask[64:448, 64:448]
                    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)  # 三个通道
                    cv2.imwrite(target_folder + name, mask.astype(np.uint8))
                source_cut_results = path_results + folder + '/'
                target_name = path_results_stitch + folder + '.png'  # 合成好的学出来的图片
                if split:
                    self.stitch_patches(source_cut_results, target_name, original_shape=original_shape)
                else:
                    self.stitch_patches(source_cut_results, target_name, original_shape=original_shape[folder])
                path1 = target_name
                path2 = source + 'label/' + folder[:-4] + 'label.png'
                y_true = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
                y_pred = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

                # self.visall(y_pred)

                kappa_value, overall_accuracy = self.metrics(y_pred, y_true, label_list)
                kappa.append(kappa_value)
                oa.append(overall_accuracy)
                text += folder + '\r\n'
                text = text + 'kappa_value:' + str(kappa_value) + '\r\n'
                text = text + 'overall_accuracy:' + str(overall_accuracy) + '\r\n'
                print(text)
            return np.mean(kappa), np.mean(oa)

        for i, folder in enumerate(list_cut): # folder是原图片的名称，去了后标的

            target_folder = path_results
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            target_folder = target_folder + folder + '/'  # 每张图片建立了一个result文件夹，放测试出来的图片
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            source_folder = path_cut + folder + '/' # test/cut/top_Potsdam_... 原图片剪切好的文件夹
            source_list = os.listdir(source_folder) #剪切好的每一张图片
            # print('source_list: ',source_list)
            for i, name in enumerate(source_list):
                # print(name)
                if i % 10 == 0:
                    print(source_folder, i, 'th', '%.2f' % (time() - tic), 's')
                mask = self.test_one_image(source_folder + name) #测试每张剪切好的图片得到他的mask
                mask = np.argmax(mask, axis=0) #获取每个类可能性最大的即其标签
                # 1024*1024 -> 768*768
                # mask = mask[128:896, 128:896]
                # mask = mask[64:448, 64:448]
                mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)  # 三个通道
                cv2.imwrite(target_folder + name, mask.astype(np.uint8))
            source_cut_results = path_results + folder + '/'
            target_name = path_results_stitch + folder + '.png' # 合成好的学出来的图片
            self.stitch_patches(source_cut_results, target_name, original_shape=original_shape)
            path1 = target_name
            path2 = source + 'label/' + folder[:-4] + 'label.png'

            IoU, IoU_list = self.computeIoU(label_list, path1, path2)
            y_true = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
            y_pred = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)


            kappa_value, overall_accuracy = self.metrics(y_pred, y_true, label_list)

            self.visall(path1, folder[:-4])
            kappa.append(kappa_value)
            oa.append(overall_accuracy)
            miou.append(IoU)
            text = text + folder[:-4] + '\r\n' + 'IoU:' + str(IoU) + '\r\n'
            text = text + 'kappa_value:' + str(kappa_value) + '\r\n'
            text = text + 'overall_accuracy:' + str(overall_accuracy) + '\r\n'
            print(text)
        return text, np.mean(kappa), np.mean(miou), np.mean(oa)
            # with open(txt_path, 'a') as f:
            #     f.write(text)
            # for num, iou in enumerate(IoU_list):
            #     text = 'class ' + str(label_list[num]) + ' IoU:' + str(num) + '\r\n'
            #     with open(txt_path, 'a') as f:
            #         f.write(text)
            # visualise
            # for j in range(0, 6):
            #
            #     target = path_results_stitch + folder[:-4] + str(j) + '_visualise.png'
            #
            #     self.Visualise(path1, path2, target, j)
            #     j += 1
if __name__ == '__main__':

    args = parse_args()
    BATCHSIZE_PER_CARD = config.TEST.BATCH_SIZE_PER_GPU
    label_list = config.TEST.LABEL_LIST
    source = config.TEST.ROOT
    if config.MODEL.NAME == 'seg_hrnet':
        solver = TTAFrame(HighResolutionNet, 'HRNet', label_list, config = config)
    else:
        # seg_model = eval('models.' + config.MODEL.NAME + '.Seg_Model')(
        #     num_classes=6
        # )
        solver = TTAFrame(ResNet, 'ccnet', label_list, config=config)
    solver.load(config.TEST.WEIGTH)
    # a, b = solver.test(source=source, label_list=label_list, split=True)
    text, kappa, miou, oa = solver.test(source=source, label_list=label_list,split=True)
    txt_path = source + 'results/' + config.EXPNAME + '_result.txt'
    with open(txt_path, 'a') as f:
        f.write(text)
        f.write("kappa: "+ str(kappa) + '\r\n')
        f.write("miou: " + str(miou) + '\r\n')
        f.write("oa: " + str(oa) + '\r\n')

