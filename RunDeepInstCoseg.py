import os
import random
from model import *
from torchvision import transforms
from PIL import Image
import math
import numpy as np
import h5py
from torch.optim import Adam
from loss import Loss
import cv2
from evaluation import EvalCoSegAP
#import matplotlib.pyplot as plt
#import pylab
import random



def instance_nms(instance_list, threshold=0.3):
    selected_instances = []
    while len(instance_list) > 0:
        instance = instance_list.pop(0)
        selected_instances.append(instance)
        src_mask = instance[1].astype(bool)

        def iou_filter(x):
            dst_mask = x[1].astype(bool)
            # IoU
            intersection = np.logical_and(src_mask, dst_mask).sum()
            union = np.logical_or(src_mask, dst_mask).sum()
            iou = intersection / (union + 1e-10)
            if iou < threshold:
                return x
            else:
                return None

        instance_list = list(filter(iou_filter, instance_list))
    return selected_instances

def noise_filter(all_class_instance_list, threshold=20):
    all_scores = []
    for i in range(len(all_class_instance_list)):
        for j in range(len(all_class_instance_list[i])):
            all_scores.append(all_class_instance_list[i][j][0])
    thscore = np.percentile(all_scores, threshold)
    new_class_instance_list = []
    for i in range(len(all_class_instance_list)):
        tmp = []
        for j in range(len(all_class_instance_list[i])):
            if all_class_instance_list[i][j][0] > thscore:
                tmp.append((all_class_instance_list[i][j][0], all_class_instance_list[i][j][1]))
        new_class_instance_list.append(tmp)

    return new_class_instance_list

class DataStream:
    def __init__(self, dataset_name, epoches=100):
        self.Dataset_path = './Dataset/' + dataset_name + '/'
        self.Image_path = self.Dataset_path + 'Image'
        self.Image_class = sorted(os.listdir(self.Image_path))
        self.Mask_path = self.Dataset_path + 'Mask'
        self.MCG_path = self.Dataset_path + 'MCG_fast'
        self.SVF_path = self.Dataset_path + 'SalRes_ZhangICCV17'
        self.training_list = []
        self.epoches = epoches

        self.pre_process()



    def pre_process(self):
        for i in range(len(self.Image_class)):
            sub_images_number = len(os.listdir(os.path.join(self.Image_path, self.Image_class[i])))
            temp = []

            for j in range(self.epoches):
                t = list(range(sub_images_number))
                random.shuffle(t)
                if sub_images_number % 2 == 1:
                    t.append(random.randint(0, sub_images_number-1))
                temp.append(t)

            self.training_list.append(temp)


    def gen_image(self, index, img_size=448):
        img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.449], std=[0.226])])

        sub_images_path = os.listdir(os.path.join(self.Image_path, self.Image_class[index]))
        image_pool = torch.zeros(len(sub_images_path), 3, img_size, img_size)
        svf_pool = torch.zeros(len(sub_images_path), 1, img_size, img_size)
        for i in range(len(sub_images_path)):
            path = os.path.join(self.Image_path, self.Image_class[index], sub_images_path[i])
            img = Image.open(path)
            if img.mode == 'RGB':
                img = img_transform(img)
            else:
                img = img_transform_gray(img)
            image_pool[i] = img

            svf_path = os.path.join(self.SVF_path, self.Image_class[index], sub_images_path[i].split('.')[0] + '.mat')
            tmp = h5py.File(svf_path, 'r')
            svf = np.transpose(tmp['SalMap'])
            svf = Image.fromarray(svf)
            svf = gt_transform(svf)
            svf_pool[i] = svf

        return image_pool, svf_pool





if __name__ == '__main__':

    random.seed(0)
    torch.manual_seed(0) # cpu
    torch.cuda.manual_seed(0) #gpu
    np.random.seed(0) #numpy
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device_ids = [0, 1]
    dataset = ['VOC12', 'SOC', 'COCO_VOC', 'COCO_NONVOC']
    # dataset = ['COCO_VOC', 'COCO_NONVOC']
    # data = DataStream('VOC12')
    net = build_model()
    vgg_path = './vgg16_feat.pth'
    # device = torch.device('cuda:0')
    batch_size = 6
    img_size = 448
    lr = 1e-7
    proposal_count = 100
    AP_Threshold = [0.25, 0.5, 0.75]
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    for d in dataset:
        data = DataStream(d)
        AP25, AP50, AP75 = [], [], []
        for i in range(len(data.Image_class)):
            ins_res = os.path.join('./results', d,  data.Image_class[i]) + '.npz'
            if not os.path.exists(ins_res):
                model_save_path = os.path.join('./models', d, data.Image_class[i]) + '.pth'
                if os.path.exists(model_save_path):
                    net.load_state_dict(torch.load(model_save_path))
                    net = net.cuda()
                else:

                    net.apply(weights_init)
                    net.base.load_state_dict(torch.load(vgg_path))

                    if torch.cuda.device_count() > 1:
                        net = torch.nn.DataParallel(net.cuda(), device_ids=device_ids)
                    elif torch.cuda.is_available():
                        net = net.cuda()
                    # net = net.to(device)
                    optimizer = Adam(net.parameters(), lr, weight_decay=5e-4)
                    net.train()
                    net.zero_grad()
                    loss = Loss().cuda()

                    image_pool, svf_pool = data.gen_image(i)
                    for j in range(data.epoches):
                        image_len = len(data.training_list[i][j])
                        ave_loss_sa = 0
                        ave_loss_af = 0
                        ave_loss_co = 0
                        for k in range(math.ceil(image_len / batch_size)):
                            index = data.training_list[i][j][k*batch_size : min((k+1)*batch_size, image_len)]
                            img = image_pool[index].cuda()
                            svf = svf_pool[index].cuda()
                            prediction_salmap, pixel_affinity, sal_affinity, sal_diff, co_peak_value = net(img)

                            loss_sa, loss_af, loss_co = loss(prediction_salmap, svf, pixel_affinity, sal_affinity, sal_diff, co_peak_value)
                            all_loss = loss_sa + loss_af + loss_co
                            ave_loss_sa += loss_sa.item()
                            ave_loss_af += loss_af.item()
                            ave_loss_co += loss_co.item()
                            
                            all_loss.backward()

                            optimizer.step()
                        print('epoch: [%d/%d], loss_saliency: [%.4f], loss_affinity: [%.4f], loss_co_peak: [%.4f]' % (j, data.epoches, ave_loss_sa/image_len, ave_loss_af/image_len, ave_loss_co/image_len))
                    # model_save_path = os.path.join'./models/' + data.Image_class[i] + '.pth'
                    if not os.path.exists('./models/' + d):
                        os.makedirs('./models/' + d)
                    if torch.cuda.device_count() > 1:
                        torch.save(net.module.state_dict(), model_save_path)
                    else:
                        torch.save(net.state_dict(), model_save_path)

                # peak back propagation
                print('peak_back_propagation')

                net._patch()
                all_eval_img = os.listdir(os.path.join(data.Image_path, data.Image_class[i]))
                all_class_instance_list = []
                for k in range(len(all_eval_img)):
                    path = os.path.join(data.Image_path, data.Image_class[i], all_eval_img[k])
                    img = Image.open(path)
                    w, h = img.size
                    img_area = w*h

                    MCG_path = os.path.join(data.MCG_path, data.Image_class[i], all_eval_img[k].split('.')[0] + '.mat')
                    temp = h5py.File(MCG_path, 'r')
                    mag_proposals = np.transpose(temp['masks'])
                    mag_scores = np.transpose(temp['scores'])

                    sorted_mag_scores_index = np.argsort(-mag_scores[:, 0])[: min(proposal_count, mag_proposals.shape[-1])]
                    mag_scores = mag_scores[sorted_mag_scores_index]
                    mag_proposals = mag_proposals[:, :, sorted_mag_scores_index]

                    proposals_num = mag_proposals.shape[-1]
                    if img.mode == 'RGB':
                        img = img_transform(img)
                    else:
                        img = img_transform_gray(img)
                    image_pool = torch.zeros(1, 3, img_size, img_size).cuda()
                    image_pool[0] = img
                    image_pool.requires_grad_()
                    sal_map, plane_peak, peak_list, small_salmap = net(image_pool, visual=True)
                    grad_output = small_salmap.new_empty(small_salmap.size())
                    instance_list = []
                    for idx in range(peak_list.size(0)):
                        grad_output.zero_()
                        grad_output[0, 0, peak_list[idx, 0], peak_list[idx, 1]] = 1
                        if image_pool.grad is not None:
                            image_pool.grad.zero_()
                        small_salmap.backward(grad_output, retain_graph=True)
                        prm = image_pool.grad.detach().sum(1).clone().clamp(min=0)
                        prm = prm[0] / prm[0].sum()
                        prm = prm * sal_map[0, 0]
                        prm = prm.detach().cpu().numpy()
                        prm = Image.fromarray(prm)
                        prm = np.array(prm.resize((w, h), Image.BILINEAR))

                        bg_map = 1 - sal_map[0, 0]
                        bg_map = bg_map.detach().cpu().numpy()
                        bg_map = Image.fromarray(bg_map)
                        bg_map = np.array(bg_map.resize((w, h), Image.BILINEAR))

                        max_val = -np.inf
                        instance_mask = None

                        for p in range(proposals_num):
                            raw_mask = Image.fromarray(mag_proposals[:, :, p])
                            raw_mask = np.array(raw_mask.resize((w, h), Image.NEAREST))
                            contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
                            mask = raw_mask.astype(bool)
                            mask_area = mask.sum()

                            if (mask_area >= 0.85 * img_area) or (mask_area < 0.00002 * img_area):
                                continue
                            else:
                                val = 0.8 * prm[mask].sum() + prm[contour_mask].sum() - 1e-5 * bg_map[mask].sum()
                                if val > max_val:
                                    max_val = val
                                    instance_mask = mask

                        if instance_mask is not None:
                            instance_list.append((max_val, instance_mask))

                    instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
                    # NMS
                    instance_list = instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), 0.4)

                    # for ss in range(len(instance_list)):
                    #     plt.text(0, 0, '%.6f'%instance_list[ss][0], ha='center', va='bottom', fontsize=20)
                    #     plt.imshow(instance_list[ss][1])
                    #     pylab.show()

                    all_class_instance_list.append(instance_list)

                # NoiseFilter
                all_class_instance_list = noise_filter(all_class_instance_list)

                GTInstMasks = []
                ins_gt_list = os.listdir(os.path.join(data.Mask_path, data.Image_class[i]))
                for s in range(len(all_eval_img)):
                    ins_gt_name = all_eval_img[s].split('.')[0] + '_InstID'
                    ins_gt = list(filter(lambda x: ins_gt_name in x, ins_gt_list))
                    GTInstMask = []
                    for t in range(len(ins_gt)):
                        cur_ins_gt_path = os.path.join(data.Mask_path, data.Image_class[i], ins_gt[t])
                        GTInstMask.append(np.array(Image.open(cur_ins_gt_path)))
                    GTInstMasks.append(GTInstMask)
                SelectProposals, SelectScores = [], []
                for s in range(len(all_class_instance_list)):
                    cur_instance_list = all_class_instance_list[s]
                    SelectProposal = list(map(lambda x: x[1], cur_instance_list))
                    SelectScore = list(map(lambda x: x[0], cur_instance_list))
                    SelectProposals.append(SelectProposal)
                    SelectScores.append(SelectScore)
                if not os.path.exists('./results/' + d):
                    os.makedirs('./results/' + d)
                np.savez(ins_res, SelectProposals=SelectProposals, SelectScores=SelectScores, GTInstMasks=GTInstMasks, all_eval_img=all_eval_img)
                net._recover()
                net.train()
            else:
                nzfile = np.load(ins_res, allow_pickle=True)
                SelectProposals, SelectScores, GTInstMasks, all_eval_img = nzfile['SelectProposals'], nzfile['SelectScores'], nzfile['GTInstMasks'], nzfile['all_eval_img']
            AP_All = EvalCoSegAP(SelectProposals, SelectScores, GTInstMasks, AP_Threshold)
            AP25.append(AP_All[0])
            AP50.append(AP_All[1])
            AP75.append(AP_All[2])
            print("%s : ap25: %.4f, ap50: %.4f, ap75: %.4f" % (data.Image_class[i], AP_All[0], AP_All[1], AP_All[2]))
            # print('done')
        print("%s : ap25: %.4f, ap50: %.4f, ap75: %.4f" % (d, np.mean(AP25), np.mean(AP50), np.mean(AP75)))

    # ins_gt_list = os.listdir(os.path.join(data.Mask_path, data.Image_class[i]))
    # for s in range(len(all_class_instance_list)):
    #     cur_instance_list = all_class_instance_list[s]
    #     SelectProposals = list(map(lambda x: x[1], cur_instance_list))
    #     SelectScores = list(map(lambda x: x[0], cur_instance_list))
    #     ins_gt_name = all_eval_img[s].split('.')[0] + '_InstID'
    #     ins_gt = list(filter(lambda x: ins_gt_name in x, ins_gt_list))
    #     GTInstMasks = []
    #     for t in range(len(ins_gt)):
    #         cur_ins_gt_path = os.path.join(data.Mask_path, data.Image_class[i], ins_gt[t])
    #         GTInstMasks.append(np.array(Image.open(cur_ins_gt_path)))
    #     AP_All = EvalCoSegAP(SelectProposals, SelectScores, GTInstMasks, AP_Threshold)








        # for s in range(len(all_class_instance_list)):
        #     cur_instance_list = all_class_instance_list[s]
        #     for q in range(len(cur_instance_list)):
        #         cur_instance_map = Image.fromarray(cur_instance_list[q][1].astype(np.int32) * 255)
        #
        #         save_root_path = os.path.join('./results/VOC12', data.Image_class[i])
        #         if not os.path.exists(save_root_path):
        #             os.mkdir(save_root_path)
        #         save_path = os.path.join(save_root_path, all_eval_img[s].split('.')[0] + '_ins_%d.jpg' % (q))
        #         cur_instance_map.convert('L').save(save_path)

    #     print('done')
    #     net._recover()
    #     net.train()
    #
    #
    #
    #
    #
    #
    # print('all done')