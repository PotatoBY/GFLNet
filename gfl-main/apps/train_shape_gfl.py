import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path as osp

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.model.SkinWeightModel import SkinWeightNet
from lib.model.ImageReconstructModel import *
# from lib.geometry import index

import pdb  # pdb.set_trace()
from torch import nn

from collections import OrderedDict
# get options
opt = BaseOptions().parse()

def save_obj(ps, tris, name):
    with open(name, 'w') as fp:
        for v in ps:
            fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        if tris is not None:
            for f in tris:  # Faces are 1-based, not 0-based in obj files
                fp.write('f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))

def save_batch_objs(bps, face_index, gtypes, names):
    if gtypes is not None:
        garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
        garmentfnums = [8348, 8352, 10516, 7284, 10660, 5468]
        index = gtypes.nonzero(as_tuple=False)[:, 1]
        # index = gtypes.nonzero(as_tuple=False)
        batch_vertices = bps
        batch_faces = face_index
        for ind in range(gtypes.shape[0]):
            up_vertices = batch_vertices[:garmentvnums[index[0 + ind * 2]]]
            bottom_vertices = batch_vertices[garmentvnums[index[0 + ind * 2]]:garmentvnums[index[0 + ind * 2]] + garmentvnums[index[1 + ind * 2]]]
            up_faces = batch_faces[:garmentfnums[index[0 + ind * 2]]]
            bottom_faces = batch_faces[garmentfnums[index[0 + ind * 2]]:garmentfnums[index[0 + ind * 2]] + garmentfnums[index[1 + ind * 2]]]
            ps = [up_vertices, bottom_vertices]
            tris = [up_faces, bottom_faces]
            for i in range(2):
                save_obj(ps[i], tris[i], names[i + ind * 2])
            batch_vertices = batch_vertices[garmentvnums[index[0 + ind * 2]] + garmentvnums[index[1 + ind * 2]]:]
            batch_faces = batch_faces[garmentfnums[index[0 + ind * 2]] + garmentfnums[index[1 + ind * 2]]:]
    else:
        for n in range(bps.shape[0]):
            save_obj(bps[n], face_index, names[n])


def train(opt, visualCheck_0=False):
    # ----- init. -----

    # set GPU idx
    if len(opt.gpu_ids) > 1: os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    cuda = torch.device('cuda') if len(opt.gpu_ids) > 1 else torch.device('cuda:%d' % opt.gpu_id)
    if len(opt.gpu_ids) > 1: assert (torch.cuda.device_count() > 1)

    # make dir to save weights
    os.makedirs(opt.checkpoints_path, exist_ok=True)  # exist_ok=True: will NOT make a new dir if already exist
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)

    # make dir to save visualizations
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    # save args.
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile: outfile.write(json.dumps(vars(opt), indent=2))

    # ----- create train/test dataloaders -----

    train_dataset = TrainDatasetGFL(opt, phase='train')
    test_dataset = TrainDatasetGFL(opt, phase='test')

    # train dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data sizes: ', len(train_dataset))  # 360, (number-of-training-meshes * 360-degree-renderings) := namely, the-number-of-training-views
    print('train data iters for each epoch: ', len(train_data_loader))  # ceil[train-data-sizes / batch_size]

    # test dataloader: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data sizes: ', len(test_dataset))  # 360, (number-of-test-meshes * 360-degree-renderings) := namely, the-number-of-training-views
    # print('test data iters for each epoch: ', len(test_data_loader)) # ceil[test-data-sizes / 1]

    # ----- build networks -----
    # config_vit = CONFIGS_ViT_seg[opt.vit_name]
    # config_vit.n_skip = opt.n_skip
    # if opt.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(opt.img_size / opt.vit_patches_size), int(opt.img_size / opt.vit_patches_size))

    # {create, deploy} networks to the specified GPU
    skinWsNet = SkinWeightNet(4, True)
    netG = ImageReconstructModel(opt, skinWsNet, True)
    # netG = TransPIFuNet(opt, config_vit, projection_mode, img_size=opt.img_size)
    print('Using Network: ', netG.name)
    if len(opt.gpu_ids) > 1: netG = nn.DataParallel(netG)
    netG.to(cuda)

    # define the optimizer
    # conv = list(map(id, netG.segimgEncoder.parameters()))
    # base_params = filter(lambda p: id(p) not in conv, netG.parameters())
    # optimizerG = torch.optim.Adam([{'params': base_params},{'params': netG.segimgEncoder.parameters(), 'lr': opt / 10}], lr=opt.learning_rate,betas=(0.9, 0.999))
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate

    # ----- load pre-trained weights if provided -----

    # load well-trained weights
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        assert (os.path.exists(opt.load_netG_checkpoint_path))
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    # load mid-training weights
    opt.continue_train = True
    if opt.continue_train:
        # opt.checkpoints_path='/mnt/data/Project/GFLNet-master/gfl-main/checkpoints'
        # opt.resume_name='GFLNet'
        # opt.resume_epoch=0
        # opt.resume_iter=2406
        # model_path = '%s/%s/netG_epoch_%d_%d' % (opt.checkpoints_path, opt.resume_name, opt.resume_epoch, opt.resume_iter)
        model_path = '/mnt/data/Project/GFLNet-master/gfl-main/checkpoints/BCNet/garNet.pth'

        new_state_dict = OrderedDict()
        state_dict = torch.load(model_path, map_location=cuda)
        for k, v in state_dict.items():
            name = k
            if 'skinWsNet' in k:
                continue
            new_state_dict[name] = v
        netG.load_state_dict(new_state_dict)

        # print('Resuming from ', model_path)
        # assert (os.path.exists(model_path))
        # netG.load_state_dict(new_state_dict)

        # change lr
        for epoch in range(0, opt.resume_epoch + 1):
            lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

    # ----- enter the training loop -----
    print("entering the training loop...")
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch + 1, 0)  # usually: 0
    for epoch in range(start_epoch, opt.num_epoch):
        netG.train()  # set to training mode (e.g. enable dropout, BN update)

        # start an epoch of training
        with tqdm(train_data_loader, unit="batch") as tepoch:
            train_idx = 0
            for train_data in tepoch:  # 14496 iters for each epoch
                tepoch.set_description(f"Epoch {epoch}")
                iter_start_time = time.time()

                # get a training batch
                image_tensor = train_data['img'].to(device=cuda)  # (B==2, C, W, H) RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
                image_path = train_data['img_paths']
                seg_tensor = train_data['seg'].to(device=cuda)   # (B==2, C, W, H) RGB, 3x512x512 images, float -1. ~ 1., bg is all ZEROS not -1.
                label_tensor = train_data['labels']
                labels_lines_tensor = train_data['label_lines']
                alpha_labels_tensor = train_data['alpha_label']
                weight_labels_tensor = train_data['weight_label']
                img_gtypes = train_data['img_gtypes']

                # network forward pass
                res, error = netG.forward(image_tensor[0], seg_tensor[0], label_tensor, labels_lines_tensor, alpha_labels_tensor, weight_labels_tensor, gtypes=img_gtypes)  # get output for every net

                # compute gradients and update weights
                optimizerG.zero_grad()
                if len(opt.gpu_ids) > 1: error = error.mean()
                # error.backward()
                optimizerG.step()


                # save weights for every opt.freq_save iters, 50 iters
                if (train_idx == len(train_data_loader) - 1) or (train_idx % opt.freq_save == 0 and train_idx != 0):
                    # torch.save(netG.state_dict(), '%s/%s/netG_latest'   % (opt.checkpoints_path, opt.name))
                    torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d_%d' % (opt.checkpoints_path, opt.name, epoch, train_idx))

                # # save query points into .ply (red-inside, green-outside) for every opt.freq_save_ply iters, 100 iters
                # if (train_idx == len(train_data_loader) - 1) or (train_idx % opt.freq_save_ply) == 0:
                #     # .ply (rotated to align with the view)
                #     save_path = '%s/%s/pred_%d_%d.ply' % (opt.results_path, opt.name, epoch, train_idx)
                #     res_one_frame = res[0].cpu()  # (1, 5000)
                #     # points = sample_tensor[0].transpose(0, 1).cpu() # (n_in + n_out, 3)
                #     # save_samples_truncted_prob(save_path, points.detach().numpy(), res_one_frame.detach().numpy())
                #     rot = extrinsic_tensor[0, 0, :3, :3]  # (3, 3)
                #     trans = extrinsic_tensor[0, 0, :3, 3:4]  # (3, 1)
                #     samples_roted = torch.addmm(trans, rot, sample_tensor[0].cpu())  # (3, N)
                #     save_samples_truncted_prob(save_path, samples_roted.T.detach().numpy(), res_one_frame.T.detach().numpy())
                #
                #     # .png (with augmentation)
                #     save_path = '%s/%s/pred_%d_%d.png' % (opt.results_path, opt.name, epoch, train_idx)
                #     image_tensor_reshaped = image_tensor.view(label_tensor.shape[0], -1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1])  # (B==2, num_views, C, W, H)
                #     img_BGR = ((np.transpose(image_tensor_reshaped[0, 0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.).astype(np.uint8)[:, :, ::-1]  # RGB to BGR, (512,512,3), [0, 255]
                #     cv2.imwrite(save_path, img_BGR)  # cv2 save BGR-array into proper-color.png

                # for recording dataloading time
                # iter_data_time = time.time()

                if res['body_ps'] != None:
                    names = []
                    names_body = []
                    names_label = []
                    label_displacements = [[0, 0, 0]]
                    batch_id = 0
                    for ind, file in enumerate(image_path[0]):
                        basename = osp.splitext(osp.basename(file))[0]
                        names.append(osp.join(opt.save_root, file.split('/')[-3] + '_' + basename + '_pca_up.obj'))
                        names.append(osp.join(opt.save_root, file.split('/')[-3] + '_' + basename + '_pca_bottom.obj'))
                        names_body.append(
                            osp.join(opt.save_root, file.split('/')[-3] + '_' + basename + '_pca_smpl.obj'))
                        names_label.append(
                            osp.join(opt.save_root, file.split('/')[-3] + '_' + basename + '_pca_up_label.obj'))
                        names_label.append(
                            osp.join(opt.save_root, file.split('/')[-3] + '_' + basename + '_pca_bottom_label.obj'))
                        label_displacements = np.concatenate((label_displacements, label_tensor[0]['up'][ind]), axis=0)
                        label_displacements = np.concatenate((label_displacements, label_tensor[0]['bottom'][ind]),
                                                             axis=0)
                        label_displacements = np.delete(label_displacements, 0, axis=0)
                        save_batch_objs(res['preds'].cpu().detach().numpy(), res['face_index'], res['gtypes'], names)
                        save_batch_objs(res['body_ps'].cpu().detach().numpy(), netG.smpl.faces, None, names_body)
                        save_batch_objs(label_displacements, res['face_index'], res['gtypes'], names_label)
                        batch_id += 1
                train_idx += 1

        # update epoch idx of the training dataset
        train_dataset.epochIdx = (train_dataset.epochIdx + 1) % opt.epoch_offline_len

        # (lr * opt.gamma) at epoch indices defined in opt.schedule
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        # evaluate the model after each training epoch
        # with torch.no_grad():
        #     netG.eval()  # set to test mode (e.g. disable dropout, BN does't update)
        #
        #     # quantitative eval. for {MSE, IOU, prec, recall} metrics
        #     if not opt.no_num_eval:
        #         test_losses = {}
        #
        #         # compute metrics for 100 test frames
        #         print('calc error (test) ...')
        #         test_errors = calc_error_gfl(opt, netG, cuda, test_data_loader, num_tests=50)  # avg. {error, IoU, precision, recall} computed among 100 frames, each frame has e.g. 5000 query points for evaluation.
        #         text_show_0 = 'Epoch-{} | eval  test CD: {:06f} PSD: {:06f} line_loss: {:06f}'.format(epoch,*test_errors)
        #         print(text_show_0)
        #
        #         # compute metrics for 100 train frames
        #         print('calc error (train) ...')
        #         train_dataset.allow_aug = False  # switch-off training data aug.
        #         train_errors = calc_error_gfl(opt, netG, cuda, test_data_loader, num_tests=50)
        #         train_dataset.allow_aug = True  # switch-on  training data aug.
        #         text_show_1 = 'Epoch-{} | eval train CD: {:06f} PSD: {:06f} line_loss: {:06f}'.format(epoch,*train_errors)
        #         print(text_show_1)



if __name__ == '__main__':
    train(opt)
