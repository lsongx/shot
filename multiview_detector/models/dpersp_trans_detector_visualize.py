import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


# visualize
class DPerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', depth_scales=4):
        super().__init__()
        self.depth_scales = depth_scales
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat,
                                                                           dataset.base.depth_margin)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        proj_mats = {}
        for i in range(self.depth_scales):
            proj_mats[i] = [
                torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[(cam, i)] @ img_zoom_mat)
                for cam in range(self.num_cam)]

        batch_size = 1
        # [N*B,C,H,W]
        for i in range(self.depth_scales):
            proj = torch.stack(proj_mats[i]).float()[None].repeat([batch_size,1,1,1]) # B
            all_proj = 0
            proj_mats[i] = nn.Parameter(proj.view([-1,3,3]), requires_grad=False).to('cuda:0')
        # self.proj_mats = nn.ParameterDict(proj_mats) # no need to register
        self.proj_mats = proj_mats

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            norm_layer = None
            # norm_layer = lambda x: nn.GroupNorm(16, x)
            base = nn.Sequential(*list(resnet18(
                # pretrained=True,
                replace_stride_with_dilation=[False, True, True],
                norm_layer=norm_layer).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:1')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.feat_down = nn.Sequential(
            nn.Conv2d(out_channel, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2)).to('cuda:0')
        out_channel = 128

        self.feat_before_merge = nn.ModuleDict({
            f'{i}': nn.Conv2d(out_channel, out_channel, 3, padding=1)
            for i in range(self.depth_scales)
        }).to('cuda:0')

        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel*self.num_cam+2, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4,  dilation=4, bias=False)).to('cuda:0')

        self.depth_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                              nn.Conv2d(64, self.depth_scales, 1, bias=False)).to('cuda:0')

        self.coord_map = nn.Parameter(self.coord_map, False).to('cuda:0')
        self.trans_img = nn.Sequential(nn.AdaptiveAvgPool2d(self.upsample_shape), nn.Conv2d(2,2,21,padding=10,bias=False))
        for p in self.trans_img.parameters():
            p.requires_grad = False
        self.trans_img[1].weight.data = dataset.img_kernel.float()
        self.trans_img.to('cuda:0')
        self.trans_map = nn.Sequential(nn.AdaptiveAvgPool2d(self.reducedgrid_shape), nn.Conv2d(2,2,41,padding=20,bias=False))
        for p in self.trans_map.parameters():
            p.requires_grad = False
        self.trans_map[1].weight.data = dataset.map_kernel.float()
        self.trans_map.to('cuda:0')
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(reduction='none')


    def warp_perspective(self, img_feature_all):
        warped_feat = 0
        warped_feat_no_conv = 0
        img_feature_all = self.feat_down(img_feature_all)
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1) # [b*n,d,h,w]
        uncertainty = self.get_uncertainty_score(depth_select) # [b*n,1,h,w]

        import matplotlib.pyplot as plt
        # feat = img_feature_all * depth_select[:,4][:,None]
        # for j in range(7):
        for j in range(6):
            feat = torch.norm(img_feature_all[j].detach(), dim=0)
            # feat = torch.norm(feat[j].detach(), dim=0)
            plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/feat-f{j}.png'), feat.cpu().numpy(), cmap='Oranges')
            for i in range(5):
                plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/depth-f{j}-p{i}.png'), depth_select[j,i].detach().cpu().numpy(), cmap='Blues')
                tt = kornia.warp_perspective(feat[None,None], self.proj_mats[i][j][None], self.reducedgrid_shape)[0,0]
                plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/feat-f{j}-p{i}.png'), tt.cpu().numpy(), cmap='Oranges')

        for i in range(self.depth_scales):
            in_feat = img_feature_all * depth_select[:,i][:,None]
            out_feat = kornia.warp_perspective(in_feat, self.proj_mats[i], self.reducedgrid_shape)
            warped_feat_no_conv += out_feat
            warped_feat += self.feat_before_merge[f'{i}'](out_feat) # [b*n,c,h,w]

        # for j in range(7):
        for j in range(6):
            t = torch.norm(warped_feat[j,...].detach(), dim=0)
            plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/warped-feat-f{j}.png'), t.cpu().numpy())
            t = torch.norm(warped_feat_no_conv[j,...].detach(), dim=0)
            plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/warped-feat-no-conv-f{j}.png'), t.cpu().numpy())
        t = torch.norm(warped_feat.view([-1,*self.reducedgrid_shape]).detach(), dim=0)
        plt.imsave(os.path.expanduser(f'~/code/jupyter/dua/data/da/warped-feat-all.png'), t.cpu().numpy())
        import pdb; pdb.set_trace()
        return warped_feat

    def forward(self, imgs, imgs_gt=None, map_gt=None, alpha=0, visualize=False):
        # np.save('imgs_gt.npy',imgs_gt.cpu().numpy())
        # np.save('map_gt.npy',map_gt.cpu().numpy())

        from PIL import Image
        from multiview_detector.utils.image_utils import img_color_denormalize
        denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = denormalize(imgs[0])*255
        for i in range(imgs.shape[1]):
            t = Image.fromarray(img[i].cpu().numpy().astype('uint8').transpose([1,2,0]))
            t = t.resize((480,270))
            t.save(os.path.expanduser(f'~/code/jupyter/dua/data/da/image-f{i}.png'))

        t = torch.tensor(np.asarray(t).transpose([2,0,1]))[None].float().cuda()
        for i in range(5):
            tt = kornia.warp_perspective(t, self.proj_mats[i][0][None], self.reducedgrid_shape)
            tt = tt[0].cpu().numpy().transpose([1,2,0]).astype('uint8')
            tt = Image.fromarray(tt)
            tt.save(os.path.expanduser(f'~/code/jupyter/dua/data/da/image{i}.png'))

        import matplotlib.pyplot as plt
        plt.imsave(os.path.expanduser('~/code/jupyter/dua/data/da/map.png'), self.trans_map(map_gt)[0,0].cpu().numpy())
        plt.imsave(os.path.expanduser('~/code/jupyter/dua/data/da/imgs_gt_0.png'), self.trans_img(imgs_gt)[0,0].cpu().numpy())
        # import ipdb; ipdb.set_trace()

        # implemented assuming B=1
        B, N, C, H, W = imgs.shape
        img_feature_all = self.base_pt1(imgs.view([-1,C,H,W]).to('cuda:1'))
        img_feature_all = self.base_pt2(img_feature_all.to('cuda:1'))
        img_feature_all = F.interpolate(img_feature_all, self.upsample_shape, mode='bilinear').to('cuda:0')
        imgs_result = self.img_classifier(img_feature_all)
        world_features = self.warp_perspective(img_feature_all)
        world_features = torch.cat([world_features.view([B,-1,*self.reducedgrid_shape]), self.coord_map.repeat([B, 1, 1, 1])], dim=1)
        map_result = self.map_classifier(world_features)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        if not self.training:
            return map_result, [i[None] for i in imgs_result]
        with torch.no_grad():
            imgs_gt = self.trans_img(imgs_gt).to('cuda:0')
            map_gt = self.trans_map(map_gt)
        loss = self.mse(imgs_result, imgs_gt)*alpha+self.mse(map_result, map_gt)
        return loss, map_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat, depth_margin):

        # np.save('intrinsic_matrices.npy',intrinsic_matrices)
        # np.save('extrinsic_matrices.npy',extrinsic_matrices)
        # np.save('worldgrid2worldcoord_mat.npy',worldgrid2worldcoord_mat)

        projection_matrices = {}
        for cam in range(self.num_cam):
            for i in range(self.depth_scales):
                # intrinsic_matrices [4,4] @ extrinsic_matrices [4,4] @ worldgrid2worldcoord_mat [4,4]
                # inv: worldgrid2worldcoord_mat-1 @ extrinsic_matrices-1 @ intrinsic_matrices-1
                extrinsic_matrices[cam][:,3] = extrinsic_matrices[cam][:,3]+extrinsic_matrices[cam][:,2]*i*depth_margin
                worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
                worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
                imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
                # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
                # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
                permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) # x->y, y->x
                projection_matrices[(cam,i)] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)


if __name__ == '__main__':
    test()
