import os
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import glob

import dataset
from config_validator import Config
from misc.infer_wsi_utils import *
from loss.ceo_loss import count_pred


def compute_acc(pred_, ano_):
    pred, ano = pred_.copy(), ano_.copy()
    pred = pred[ano > 0]
    ano = ano[ano > 0]
    acc = np.mean(pred == ano)
    return np.round(acc, 4)


class Inferer(Config):
    def __init__(self, _args=None):
        super(Inferer, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
        self.run_info = self.run_info
        self.net_name = self.run_info
        self.net_dir = self.net_dir
        self.in_img_path = self.in_img_path
        self.in_ano_path = self.in_ano_path
        self.in_patch = self.in_patch
        self.out_img_path = self.out_img_path
        self.net_name = self.net_name
        self.infer_batch_size = 256
        self.nr_procs_valid = 31
        self.patch_size = 1144
        self.patch_stride = 1144 // 2
        self.nr_classes = 4

    def resize_save(self, svs_code, save_name, img, scale=1.0):
        ano = img.copy()
        cmap = plt.get_cmap('jet')
        path = f'{self.out_img_path}/{svs_code}/'
        img = (cmap(img / scale)[..., :3] * 255).astype('uint8')
        img[ano == 0] = [10, 10, 10]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{path}/{save_name}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return 0

    def infer_step_m(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_class, _ = net(imgs)  # forward
            prob = nn.functional.softmax(logit_class, dim=1)
            # prob = prob.permute(0, 2, 3, 1)  # to NHWC
            return prob.cpu().numpy()

    def infer_step_c(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            logit_class = net(imgs)  # forward
            prob = nn.functional.softmax(logit_class, dim=1)
            # prob = prob.permute(0, 2, 3, 1)  # to NHWC
            return prob.cpu().numpy()

    def infer_step_r(self, net, batch, net_name):
        net.eval()  # infer mode

        imgs = batch  # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        with torch.no_grad():  # dont compute gradient
            if "rank_ordinal" in net_name:
                logits, probas = net(imgs)
                predict_levels = probas > 0.5
                pred = torch.sum(predict_levels, dim=1)
                return pred.cpu().numpy()
            elif "rank_dorn" in net_name:
                pred, softmax = net(imgs)
                return pred.cpu().numpy()
            elif "soft_label" in net_name:
                logit_regres = net(imgs)  # forward
                label = torch.tensor([0., 1. / 3., 2. / 3., 1.]).repeat(len(logit_regres), 1).permute(1, 0).cuda()
                idx = torch.argmin(torch.abs(logit_regres - label), 0)
                return idx.cpu().numpy()
            elif "FocalOrdinal" in net_name:
                logit_regress = net(imgs)
                pred = count_pred(logit_regress)
                return pred.cpu().numpy()
            else:
                logit_regres = net(imgs)  # forward
                label = torch.tensor([0., 1., 2., 3.]).repeat(len(logit_regres), 1).permute(1, 0).cuda()
                idx = torch.argmin(torch.abs(logit_regres - label), 0)
                return idx.cpu().numpy()

    def predict_one_model(self, net, svs_code, net_name="Multi_512_mse"):
        infer_step = Inferer.__getattribute__(self, f'infer_step_{net_name[0].lower()}')
        ano = np.float32(np.load(f'{self.in_ano_path}/{svs_code}.npy'))  # [h, w]
        inf_output_dir = f'{self.out_img_path}/{svs_code}/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir)

        path_pairs = glob.glob(f'{self.in_patch}/{svs_code}/*/*.png')
        infer_dataset = dataset.DatasetSerialWSI(path_pairs)
        dataloader = data.DataLoader(infer_dataset,
                                     num_workers=self.nr_procs_valid,
                                     batch_size=256,
                                     shuffle=False,
                                     drop_last=False)

        out_prob = np.zeros([self.nr_classes, ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]
        out_prob_count = np.zeros([ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]

        for batch_data in dataloader:
            imgs_input, imgs_path = batch_data
            imgs_path = np.array(imgs_path).transpose(1, 0)
            output_prob = infer_step(net, imgs_input, net_name)
            for idx, patch_loc in enumerate(imgs_path):
                patch_loc = patch_loc.astype(int) // 16
                patch_loc = [patch_loc[0], patch_loc[1]]
                out_prob_count[patch_loc[0]:patch_loc[0] + self.patch_size // 16,
                patch_loc[1]:patch_loc[1] + self.patch_size // 16] += 1
                for grade in range(self.nr_classes):
                    out_prob[grade][patch_loc[0]:patch_loc[0] + self.patch_size // 16,
                    patch_loc[1]:patch_loc[1] + self.patch_size // 16] += output_prob[idx][grade]

        out_prob_count[out_prob_count == 0.] = 1.
        out_prob /= out_prob_count
        predict = np.argmax(out_prob, axis=0) + 1

        for c in range(self.nr_classes):
            out_prob[c][ano == 0] = 0
        predict[ano == 0] = 0

        acc = compute_acc(predict, ano)
        print(acc)

        self.resize_save(svs_code, f'predict_{net_name}_{acc}', predict, scale=4.0)
        self.resize_save(svs_code, 'ano', ano, scale=4.0)
        np.save(f'{self.out_img_path}/{svs_code}/predict_{net_name}', predict)
        np.save(f'{self.out_img_path}/{svs_code}/ano', ano)
        print('done')
        return 0

    def predict_one_model_regress(self, net, svs_code, net_name="Multi_512_mse"):
        infer_step = Inferer.__getattribute__(self, f'infer_step_{net_name[0].lower()}')
        ano = np.float32(np.load(f'{self.in_ano_path}/{svs_code}.npy'))  # [h, w]
        inf_output_dir = f'{self.out_img_path}/{svs_code}/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir)

        path_pairs = glob.glob(f'{self.in_patch}/{svs_code}/*/*.png')
        infer_dataset = dataset.DatasetSerialWSI(path_pairs)
        dataloader = data.DataLoader(infer_dataset,
                                     num_workers=self.nr_procs_valid,
                                     batch_size=128,
                                     shuffle=False,
                                     drop_last=False)
        out_prob = np.zeros([self.nr_classes, ano.shape[0], ano.shape[1]], dtype=np.float32)  # [h, w]

        for batch_data in dataloader:
            imgs_input, imgs_path = batch_data
            imgs_path = np.array(imgs_path).transpose(1, 0)
            output_prob = infer_step(net, imgs_input, net_name)
            for idx, patch_loc in enumerate(imgs_path):
                patch_loc = patch_loc.astype(int) // 16
                patch_loc = [patch_loc[0], patch_loc[1]]
                for grade in range(self.nr_classes):
                    if grade == output_prob[idx]:
                        out_prob[grade][patch_loc[0]:patch_loc[0] + self.patch_size // 16,
                        patch_loc[1]:patch_loc[1] + self.patch_size // 16] += 1
        predict = np.argmax(out_prob, axis=0) + 1

        for c in range(self.nr_classes):
            out_prob[c][ano == 0] = 0
        predict[ano == 0] = 0

        acc = compute_acc(predict, ano)
        plt.imshow(predict)
        plt.show()
        print(acc)
        self.resize_save(svs_code, f'predict_{net_name}_{acc}', predict, scale=4.0)
        self.resize_save(svs_code, 'ano', ano, scale=4.0)
        np.save(f'{self.out_img_path}/{svs_code}/predict_{net_name}', predict)
        np.save(f'{self.out_img_path}/{svs_code}/ano', ano)
        print('done')
        return 0

    def run_wsi(self):
        device = 'cuda'

        self.task_type = self.net_name.split('_')[0]

        if "rank_dorn" in self.net_name:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model_rank_ordinal')  # dynamic import
            net = net_def.jl_efficientnet(task_mode='regress_rank_dorn', pretrained=True)

        elif "FocalOrdinalLoss" in self.net_name:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model')  # dynamic import
            net = net_def.jl_efficientnet(task_mode='class', pretrained=True, num_classes=3)
        else:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model')  # dynamic import
            net = net_def.jl_efficientnet(task_mode=self.task_type.lower(), pretrained=True)

        net = torch.nn.DataParallel(net).to(device)
        inf_model_path = os.path.join(self.net_dir, self.net_name, f'trained_net.pth')
        saved_state = torch.load(inf_model_path)
        net.load_state_dict(saved_state)

        name_wsi_list = findExtension(self.in_ano_path, '.npy')

        for name in name_wsi_list:
            svs_code = name[:-4]
            print(svs_code)
            acc_wsi = []
            if 'REGRESS' in self.net_name:
                acc_one_model = self.predict_one_model_regress(net, svs_code, net_name=self.net_name)
            else:
                acc_one_model = self.predict_one_model(net, svs_code, net_name=self.net_name)
            acc_wsi.append(acc_one_model)


####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--run_info', type=str, default='REGRESS_rank_dorn',
                        help='CLASS, REGRESS, MULTI + loss, '
                             'loss ex: Class_ce, MULTI_mtmr, REGRESS_rank_ordinal, REGRESS_rank_dorn'
                             'REGRESS_FocalOrdinalLoss, REGRESS_soft_ordinal')
    parser.add_argument('--net_dir', type=str,
                        default='/media/trinh/Data0/submit_paper_data/JL_pred/model/JL_model/JL_colon_model/',
                        help='path to checkpoint model')
    parser.add_argument('--in_img_path', type=str,
                        default='/media/data1/trinh/data/workspace_data/colon_wsi/ColonWSI/',
                        help='path to wsi image')
    parser.add_argument('--in_ano_path', type=str,
                        default='/media/data1/trinh/data/workspace_data/colon_wsi/Colon_WSI_annotation_npy/',
                        help='path to wsi npy annotation')
    parser.add_argument('--in_patch', type=str,
                        default='/media/data1/trinh/data/workspace_data/colon_wsi/patches_colon/colon_45WSIs_1144_01_step05_visualize_resize512/',
                        help='path to patch image')
    parser.add_argument('--out_img_path', type=str,
                        default='/media/data1/trinh/data/workspace_data/colon_wsi/JointLearning_wsi_pred/',
                        help='path to patch image')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    inferer = Inferer(_args=args)
    inferer.run_wsi()
