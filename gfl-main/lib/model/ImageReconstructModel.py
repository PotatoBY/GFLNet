import torch
import torch_scatter
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from .GCNs import ResidualAdd, MultiPerceptro, SpiralConv
from smpl_pytorch.SMPL import SMPL
from smpl_pytorch.util import batch_rodrigues, batch_global_rigid_transformation
from torchvision.models import ResNet, resnet34
import numpy as np
import os.path as osp
from torchvision.ops import roi_align
from ..utils import compute_fnorms,compute_vnorms
import openmesh as om
from .GARMENT import GARMENT
import cv2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class PatchEncoder(Module):

	def __init__(self, layers=[2, 2, 2, 2]):
		super(PatchEncoder, self).__init__()

		self.inplanes = 32
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=True, dilation=1)
		self.bn1 = nn.BatchNorm2d(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(32, layers[0])
		self.layer2 = self._make_layer(64, layers[1], stride=2)
		self.layer3 = self._make_layer(128, layers[2], stride=2)
		self.layer4 = self._make_layer(256, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes, stride),
				nn.BatchNorm2d(planes),
			)

		layers = []
		layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(BasicBlock(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.reshape(x.size(0), -1)
		return x


class ImageEncoder(ResNet):
	def __init__(self, size=[512, 512], resSet=[2, 2, 2, 2], gar_latent_size=256):
		super(ImageEncoder, self).__init__(BasicBlock, resSet)
		self.size = size
		self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
		self.fc = nn.Linear(512 * 64, 2048)
		# origine
		# self.dropout=nn.Dropout(p=0.5)
		# ft_overfit
		self.dropout = nn.Dropout(p=0.9)
		self.shape_fc = nn.Linear(2048, 10)
		self.pose_fc = nn.Linear(2048, 24 * 9)  # output matrix formation
		self.tran_fc = nn.Linear(2048, 3)
		self.tran_dp = nn.Dropout(p=0.3)
		self.gar_latent_size = gar_latent_size
		self.gar_Hierarchifs_size = 64 + 128 + 256 + 512
		self.gar_fc = nn.Linear(2048, self.gar_latent_size)
		# this mean value is from train set.
		self.register_buffer('tran_mean',
							 torch.from_numpy(np.array([-1.0962e-02, 2.8778e-01, 1.2973e+01]).astype(np.float32)))

	def forward(self, x):
		assert (x.shape[-2] == self.size[0])
		assert (x.shape[-1] == self.size[1])
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		fs1 = x
		x = self.layer2(x)
		fs2 = x
		x = self.layer3(x)
		fs3 = x
		x = self.layer4(x)
		fs4 = x
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.dropout(x)
		x = self.fc(x)
		x = self.relu(x)
		shapes = self.shape_fc(x)
		poses = self.pose_fc(x)
		trans = self.tran_fc(self.tran_dp(x)) + self.tran_mean.view(1, 3)
		gars = self.gar_fc(x)
		return shapes, poses, trans, gars, (fs1, fs2, fs3, fs4)

class SegImageEncoder(ResNet):
	def  __init__(self, size=[512, 512], resSet=[3, 8, 36, 3], gar_latent_size=256):
		super(SegImageEncoder, self).__init__(BasicBlock, resSet)
		self.size = size
		self.seg_avgpool = nn.AdaptiveAvgPool2d((8, 8))
		self.seg_fc = nn.Linear(512 * 64, 2048)
		self.seg_dropout = nn.Dropout(p=0.9)
		self.gar_latent_size = gar_latent_size
		self.seg_gar_fc = nn.Linear(2048, self.gar_latent_size)
		self.line_shirts_fc = nn.Linear(self.gar_latent_size, 6)
		self.line_pants_fc = nn.Linear(self.gar_latent_size, 4)
		self.line_skirts_fc = nn.Linear(self.gar_latent_size, 3)

	def forward(self, x, input, type):
		assert(x.shape[-2]==self.size[0])
		assert(x.shape[-1]==self.size[1])
		# x = torch.cat([x1,x2],dim=1)
		# self.seg_conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(x.device)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.seg_avgpool(x)
		x = torch.flatten(x,1)
		x = self.seg_dropout(x)
		x = self.seg_fc(x)
		x = self.relu(x)
		gars=self.seg_gar_fc(x)
		lines= torch.Tensor().to(gars.device)
		for ind in range(gars.shape[0]):
			# line_up = self.line_shirts_fc(torch.cat([gars[ind], input[ind]], dim=0))
			line_up = self.line_shirts_fc(gars[ind])
			line1 = torch.clip(line_up[0:1] * line_up[0:1], 0, 2)
			line2 = torch.clip(line_up[1:6] * line_up[1:6], 1, 10)
			if (type[ind]==0):
				line_bottom = self.line_pants_fc(gars[ind])
			else :
				line_bottom = self.line_skirts_fc(gars[ind])
			line3 = torch.clip(line_bottom * line_bottom, 1, 10)
			if line3.shape[0]==3:
				line3 = torch.cat([line3, torch.Tensor([100]).to(gars.device)], dim=0)
			line = torch.cat([line1, line2, line3], dim=0)
			lines = torch.cat([lines, line.unsqueeze(0)], dim=0)

		# trans=self.tran_fc(self.tran_dp(x))+self.tran_mean.view(1,3)
		return lines


class SkinDeformNet(Module):
	def __init__(self,smpl):
		super(SkinDeformNet,self).__init__()
		self.smpl=smpl
	def skeleton(self,shapes,require_body=False):
		return self.smpl.skeleton(shapes,require_body)
	def forward(self,ps,JsorShapes,ws,poses,batch,check_rotation=True,is_Rotation=False):
		batch_num=poses.shape[0]
		assert(batch_num==JsorShapes.shape[0])
		if JsorShapes.shape.numel()==batch_num*10:	#is shapes
			Js=self.smpl.skeleton(JsorShapes)
		else:
			Js=JsorShapes

		# Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24, 3, 3)
		if poses.numel()==batch_num*24*3:
			Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24, 3, 3)
			Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
		elif poses.numel()==batch_num*24*9:
			#input poses are general matrix
			if not is_Rotation:
				ms=poses.reshape(-1,3,3)
				# use gram schmit regularization
				b1=F.normalize(ms[:,:,0],dim=1)
				dot_prod = torch.sum(b1 * ms[:, :, 1], dim=1, keepdim=True)
				b2 = F.normalize(ms[:, :, 1] - dot_prod * b1, dim=-1)
				b3 = torch.cross(b1,b2,dim=1)
				Rs=torch.stack([b1,b2,b3],dim=-1).reshape(batch_num,24,3,3)
			else:
				Rs=poses.reshape(batch_num,24,3,3)
			Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
		elif poses.numel()==batch_num*24*16:
			A=poses.reshape(batch_num,24,4,4)
			Js_transformed=None
			Rs=None

		# Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
		splitl=torch_scatter.scatter(batch.new_ones(batch.numel(),1),batch,dim=0).cpu().numpy().reshape(-1).astype(np.int32).tolist()
		ws=ws.split(splitl,0)
		T=torch.cat([weight.matmul(a.reshape(24,16)) for weight,a in zip(ws,A)],dim=0)
		T=T.reshape(-1,4,4)
		ps=torch.cat((ps,ps.new_ones(ps.shape[0],1)),dim=-1).unsqueeze(-1)
		ps=torch.matmul(T,ps).squeeze(-1)
		return ps[:,0:3],T,Rs,Js_transformed

class GarmentPcaDecodeLayer(Module):
	def __init__(self, pca_npz):
		super(GarmentPcaDecodeLayer, self).__init__()
		datas = np.load(pca_npz)
		self.register_buffer('mean', torch.from_numpy(datas['mean'].astype(np.float32)))
		self.register_buffer('components', torch.from_numpy(datas['components'].astype(np.float32)))
		self.register_buffer('std', torch.from_numpy(datas['singular_values'].astype(np.float32)))
		self.type = osp.splitext(osp.basename(pca_npz))[0]
		mesh = om.read_trimesh(osp.join(osp.dirname(pca_npz), 'garment_tmp.obj'))
		self.register_buffer('edge_index', torch.from_numpy(mesh.hv_indices().transpose()).to(torch.long))
		self.register_buffer('face_index', torch.from_numpy(mesh.face_vertex_indices()).to(torch.long))

		vf_fid = torch.zeros(0, dtype=torch.long)
		vf_vid = torch.zeros(0, dtype=torch.long)
		for vid, fids in enumerate(mesh.vertex_face_indices()):
			fids = torch.from_numpy(fids[fids >= 0]).to(torch.long)
			vf_fid = torch.cat((vf_fid, fids), dim=0)
			vf_vid = torch.cat((vf_vid, fids.new_ones(fids.shape) * vid), dim=0)
		self.register_buffer('vf_findex', vf_fid)
		self.register_buffer('vf_vindex', vf_vid)
		self.vnum = mesh.n_vertices()
		self.fnum = mesh.n_faces()

	def unregular_pcas(self, pcas):
		return pcas * self.std.view(1, -1)

	def forward(self, pcas):
		return torch.matmul(pcas, self.components) + self.mean.view(1, -1)


def imgBatchFromGarBatch(batch, gtypes):
	garnums_perimg = (gtypes != 0).sum(1)  # per img has one up and one bottom, this is trainset situation
	if (garnums_perimg != 2).sum() == 0:
		imgbatch = batch // 2
	else:  # general situation
		imgbatch = batch.detach().clone()
		e_ids = garnums_perimg.cumsum(dim=0)
		e_ids = e_ids.detach().cpu().numpy()
		s_id = 0
		for ind, e_id in enumerate(e_ids):
			if ind - 1 >= 0:
				s_id = e_ids[ind - 1]
			else:
				s_id = 0
			imgbatch[(batch >= s_id) * (batch < e_id)] = ind
	return imgbatch


# datas is (all_vnums,feature_num) or batch_num,-1 datas  'shirts','short_shirts','pants','short_pants','skirts','short_skirts'
def order_data_follow_gartypes(datas, batch_num, gar_batch, gtypes, garmentvnums=[4248, 4258, 5327, 3721, 5404, 2818], garments=['shirts', 'short_shirts', 'pants', 'short_pants', 'skirts', 'short_skirts']):
	indexs = gtypes.nonzero(as_tuple=False)
	gar_type_ids_per_gar = indexs[:, 1]  # type
	img_batch_ids_per_gar = indexs[:, 0]  # batch
	gar_batch_ids_per_gar = torch.arange(indexs.shape[0], device=indexs.device, dtype=torch.long)
	ordered_datas = []
	ordered_gtypes = []
	ordered_select_img_bach_ids = []
	for ind, garvnum in enumerate(garmentvnums):

		select_mask = (gar_type_ids_per_gar == ind)
		select_img_batch_ids = img_batch_ids_per_gar[select_mask]
		select_gar_batch_ids = gar_batch_ids_per_gar[select_mask]
		select_gars_num = select_gar_batch_ids.numel()
		if select_gars_num > 0:
			if gar_batch is not None:
				select_rows = ((gar_batch.view(-1, 1) - select_gar_batch_ids.view(1, -1)) == 0).nonzero(as_tuple=False)[
							  :, 0]
			tmp = []
			for data in datas:
				if data.shape[0] == batch_num:
					tmp.append(data[select_img_batch_ids].reshape(select_gars_num, data.shape[-1]))
				else:
					if gar_batch is None:
						assert (False)
					tmp.append(data[select_rows, :].reshape(select_gars_num, garvnum, data.shape[-1]))
			# print(garvnum)
			ordered_datas.append(tmp)
			ordered_gtypes.append(ind)
			ordered_select_img_bach_ids.append(select_img_batch_ids)
	return ordered_datas, ordered_select_img_bach_ids, ordered_gtypes


def unorder_data_follow_imgbatch(ordered_datas, ordered_imgbids, ordered_gtypes, batch_num, garlayers=None, require_gar_batch=False, require_edge_index=False, require_face_index=False, require_vffindex=False, require_vfvindex=False):
	record_offset = False
	if require_gar_batch or require_edge_index or require_face_index or require_vffindex or require_vfvindex:
		assert (garlayers is not None)
		assert (6 == len(garlayers))
		record_offset = True

	datas = []
	for ordered_data in ordered_datas[0]:
		datas.append(ordered_data.new_zeros(0, ordered_data.shape[-1]))
	if require_gar_batch:
		batch = torch.zeros(0, dtype=torch.long, device=ordered_imgbids[0].device)
	if require_edge_index:
		edge_index = torch.zeros(2, 0, dtype=torch.long, device=ordered_imgbids[0].device)
	if require_face_index:
		face_index = torch.zeros(0, 3, dtype=torch.long, device=ordered_imgbids[0].device)
	if require_vffindex:
		vf_findex = torch.zeros(0, dtype=torch.long, device=ordered_imgbids[0].device)
	if require_vfvindex:
		vf_vindex = torch.zeros(0, dtype=torch.long, device=ordered_imgbids[0].device)

	gid = 0
	voffset = 0
	foffset = 0
	for bid in range(batch_num):
		for ind, img_bids, tmp_datas in zip(ordered_gtypes, ordered_imgbids, ordered_datas):
			select_mask = img_bids == bid
			if select_mask.sum() == 0:
				continue
			if record_offset:
				garlayer = garlayers[ind]
			for tid, (data, ordered_data) in enumerate(zip(datas, tmp_datas)):
				if ordered_data.dim() == 2:
					data = torch.cat((data, ordered_data[select_mask]), dim=0)
				else:
					data = torch.cat((data, ordered_data[select_mask].reshape(-1, ordered_data.shape[-1])), dim=0)
				datas[tid] = data
			if require_gar_batch:
				batch = torch.cat((batch, batch.new_ones(garlayer.vnum) * gid), dim=0)
			if require_edge_index:
				edge_index = torch.cat((edge_index, garlayer.edge_index + voffset), dim=-1)
			if require_face_index:
				face_index = torch.cat((face_index, garlayer.face_index), dim=0)
			if require_vffindex:
				vf_findex = torch.cat((vf_findex, garlayer.vf_findex + foffset), dim=0)
			if require_vfvindex:
				vf_vindex = torch.cat((vf_vindex, garlayer.vf_vindex + voffset), dim=0)
			gid += 1
			if record_offset:
				voffset += garlayer.vnum
				foffset += garlayer.fnum
	other_out = {}
	if require_gar_batch:
		other_out['gar_batch'] = batch
	if require_edge_index:
		other_out['edge_index'] = edge_index
	if require_face_index:
		other_out['face_index'] = face_index
	if require_vffindex:
		other_out['vf_findex'] = vf_findex
	if require_vfvindex:
		other_out['vf_vindex'] = vf_vindex

	return datas, other_out


class GarmentPcaLayer(Module):
	def __init__(self, gtype, latent_size):
		super(GarmentPcaLayer, self).__init__()
		self.gtype = gtype
		if type(latent_size) == list:
			self.decoder = MultiPerceptro(latent_size)
		else:
			self.decoder = MultiPerceptro([latent_size, 128, 64])

	def forward(self, xs):
		return self.decoder(xs)


class GarmentDisplacementNet(Module):
	def __init__(self, opt, imgf_size, gar_latent_size, gartype, step_size=2):
		super(GarmentDisplacementNet, self).__init__()
		spiral_np = np.load(osp.join(osp.dirname(__file__),
									 opt.datasetDir+'/body_garment_dataset/tmps/%s/spiral_indices_%d.npy' % (
									 gartype, step_size)))
		# self.pointMLP=MultiPerceptro([3+3+9+3+imgf_size+gar_latent_size,512,256],False)
		# #final 1:
		# infeature_size=3+3+9+3+imgf_size+gar_latent_size
		infeature_size = 3 + 3 + 3 + 9 + 3 + 10 + 64 + imgf_size + gar_latent_size
		self.pointMLP = nn.Sequential(nn.Linear(infeature_size, 256, False), nn.ReLU())
		# self.pointMLP=SpiralConv(3+3+9+3+imgf_size+gar_latent_size,256,spiral_np)
		self.res1 = ResidualAdd(SpiralConv(256, 256, spiral_np), SpiralConv(256, 256, spiral_np), nn.ReLU(), nn.ReLU())
		self.midDown = nn.Linear(256, 128, False)
		self.ress = nn.ModuleList(
			[ResidualAdd(SpiralConv(128, 128, spiral_np), SpiralConv(128, 128, spiral_np), nn.ReLU(), nn.ReLU()) for i
			 in range(3)])
		# self.outConv=SpiralConv(256+256,3,spiral_np[:,:,:7])
		# #final 1:
		# self.outMLP=MultiPerceptro([128+128,128,3])
		self.outMLP = MultiPerceptro([256 + 128 + 128, 256, 128, 3])

	def forward(self, x):
		# pfs=self.pointMLP(x)
		assert (x.dim() == 3)
		x = self.pointMLP(x)
		batch_num, vnum, in_size = x.shape
		pfs = torch.cat((x, x.new_zeros(batch_num, 1, in_size)), dim=1)
		vnum += 1
		zero_padding = pfs.new_ones(1, vnum, 1)
		zero_padding[0, -1, 0] = 0.0
		fs = self.midDown(self.res1(pfs, zero_padding=zero_padding)) * zero_padding
		gfs, _ = torch.max(fs[:, :-1, :], 1, keepdim=True)
		for res in self.ress:
			fs = res(fs, zero_padding=zero_padding)
		# out=self.outConv(torch.cat((fs,gfs.expand(batch_num,fs.shape[1],gfs.shape[-1]))*zero_padding,dim=-1),zero_padding=zero_padding)
		out = self.outMLP(
			torch.cat((pfs, fs, gfs.expand(batch_num, fs.shape[1], gfs.shape[-1]) * zero_padding), dim=-1))
		return out[:, :-1, :]


def get_patchs_from_imgs(pros, imgs, imgbatch, box_len=32):
	x1 = pros[:, 0] - box_len / 2.
	x2 = pros[:, 0] + box_len / 2.
	y1 = pros[:, 1] - box_len / 2.
	y2 = pros[:, 1] + box_len / 2.
	boxes = torch.stack((imgbatch.to(torch.float), x1, y1, x2, y2), dim=-1)
	return roi_align(imgs, boxes, (box_len, box_len))


class ImageReconstructModel(Module):
	def __init__(self, opt, SkinWeightNet, with_classification=False):
		super(ImageReconstructModel, self).__init__()
		self.name = 'GFLNet'
		self.batch_num = opt.batch_size
		self.imgEncoder = ImageEncoder()
		# self.segimgEncoder = SegImageEncoder()
		# self.patchEncoder=PatchEncoder()
		self.garments = ['shirts', 'short_shirts', 'pants', 'short_pants', 'skirts', 'short_skirts']
		self.garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
		self.garmentfnums = [8348, 8352, 10516, 7284, 10660, 5468]
		self.garPcaparamLayers = nn.ModuleList(
			[GarmentPcaLayer(gtype, 10 + self.imgEncoder.gar_latent_size) for gtype in self.garments])
		self.garPcapsLayers = nn.ModuleList([GarmentPcaDecodeLayer(
			osp.join(osp.dirname(__file__), opt.datasetDir+'/body_garment_dataset/tmps/%s/pca_data.npz' % gtype)) for gtype in self.garments])
		self.garDisplacementLayers = nn.ModuleList(
			[GarmentDisplacementNet(opt, 256, self.imgEncoder.gar_latent_size, gtype) for gtype in self.garments])
		self.patchEncoder = MultiPerceptro([3 * 32 * 32, 1024, 524, 256])
		# self.skinWsNet = SkinWeightNet
		# use pretrained model, fix weights
		# for param in self.skinWsNet.parameters():
		# 	param.requires_grad = False
		self.smpl = SMPL(osp.join(osp.dirname(__file__), opt.datasetDir+'/smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt'), obj_saveable=True)
		self.garment = GARMENT(opt)
		self.skinDeformNet = SkinDeformNet(self.smpl)
		self.gar_classification = with_classification
		if with_classification:
			self.up_classifier = nn.Linear(self.imgEncoder.gar_latent_size, 2)
			self.up_dropout = nn.Dropout(p=0.2)
			self.bottom_classifier = nn.Linear(self.imgEncoder.gar_latent_size, 4)
			self.bottom_dropout = nn.Dropout(p=0.2)

	def pro_ps(self, ps, cam_k):
		proPs = ps.matmul(cam_k.transpose(0, 1))
		depth = proPs[:, 2].reshape(-1, 1)
		select = (depth >= -1.0e-4) * (depth <= 1.0e-4)
		signs = depth.sign()
		signs[(signs >= -0.01) * (signs <= 0.01)] = 1.0
		depth[select] = signs[select] * 1.0e-4
		proPs = torch.cat((proPs[:, 0:2] / depth, proPs[:, 2].reshape(-1, 1)), dim=-1)
		return proPs

	# all imgs have same cam_k
	def extract_tris_infos(self, tris_infos):
		edge_index = tris_infos['edge_index']
		garbatch = tris_infos['gar_batch']
		face_index = tris_infos['face_index']
		vf_vindex = tris_infos['vf_vindex']
		vf_findex = tris_infos['vf_findex']
		self.edge_index = edge_index
		self.face_index = face_index
		self.vf_vindex = vf_vindex
		self.vf_findex = vf_findex
		self.garbatch = garbatch
		return garbatch, edge_index, face_index, vf_vindex, vf_findex

	def filter(self, imgs, segimg, lines, alpha, input_imgbatch=None, **kwargs):
		gtypes = self.gtypes
		batch_num = self.batch_num
		shapes, poses, trans, garlatents, _ = self.imgEncoder(imgs)
		if 'garl' in kwargs:
			garlatents = kwargs['garl']
		self.shapes = shapes
		self.poses = poses
		self.trans = trans
		self.garl = garlatents
		self.device = imgs.device

		if gtypes is None:
			assert (self.gar_classification)
		if self.gar_classification:
			tmpfs = F.relu(garlatents)
			up_gar_prob = self.up_classifier(self.up_dropout(tmpfs))
			bottom_gar_prob = self.bottom_classifier(self.bottom_dropout(tmpfs))
			self.up_gar_prob = up_gar_prob
			self.bottom_gar_prob = bottom_gar_prob
			up_index = up_gar_prob.max(1)[1]
			up_gtypes = up_gar_prob.new_zeros(up_gar_prob.shape).scatter(1, up_index.repeat(2, 1).transpose(0, 1),up_gar_prob.new_ones(up_gar_prob.shape))
			bottom_index = bottom_gar_prob.max(1)[1]
			bottom_gtypes = bottom_gar_prob.new_zeros(bottom_gar_prob.shape).scatter(1, bottom_index.repeat(2, 1).transpose(0, 1), bottom_gar_prob.new_ones(bottom_gar_prob.shape))
			if gtypes is not None:
				if type(gtypes) is not torch.Tensor:
					gtypes = torch.tensor(gtypes, dtype=torch.long, device=self.device)
					rows, cols = torch.nonzero(gtypes >= 0, as_tuple=False).transpose(0, 1)
			tgtypes = torch.cat((up_gtypes, bottom_gtypes), dim=-1)
			for r, c in zip(rows, cols):
				c = gtypes[r, c].item()
				if c < 2:
					tgtypes[r, 0:2] = 0
					tgtypes[r, c] = 1
				elif c < 6:
					tgtypes[r, 2:] = 0
					tgtypes[r, c] = 1
			gtypes = tgtypes

		self.gtypes = gtypes
		ordered_datas, ordered_imgbids, ordered_gtypes = order_data_follow_gartypes([shapes, poses, garlatents],batch_num, None, gtypes,self.garmentvnums, self.garments)
		self.ordered_imgbids = ordered_imgbids
		self.ordered_datas = ordered_datas#####
		self.ordered_gtypes = ordered_gtypes####
		pca_datas = []
		for ind, (shapes_gtype, _, latents_gtype) in zip(ordered_gtypes, ordered_datas):
			pca_params = self.garPcaparamLayers[ind](torch.cat((shapes_gtype, latents_gtype), dim=-1))
			# print(pca_params.shape)
			# pca_params = torch.tensor(alpha[0][1]).to(imgs.device).to(torch.float32)
			# print(pca_params.shape)
			pca_ps = self.garPcapsLayers[ind](pca_params).reshape(shapes_gtype.shape[0], self.garmentvnums[ind], 3)
			pca_datas.append([pca_params, pca_ps])
		[pcas_perg, gps_pca], tris_infos = unorder_data_follow_imgbatch(pca_datas, ordered_imgbids, ordered_gtypes,batch_num, self.garPcapsLayers, True, True,True, True, True)
		self.pca_params = pcas_perg
		self.tris_infos = tris_infos
		self.pca_datas = pca_datas######

		garbatch, edge_index, face_index, vf_vindex, vf_findex = self.extract_tris_infos(tris_infos)
		self.face_index = face_index
		imgbatch = imgBatchFromGarBatch(garbatch, gtypes)
		if input_imgbatch is not None:
			assert ((imgbatch - input_imgbatch).sum() == 0)
		self.imgbatch = imgbatch
		# self.imgbatch = imgBatchFromGarBatch(garbatch, gtypes)
		# Js, body_ns = self.skinDeformNet.skeleton(shapes, True)
		# pose_Rs, Js_transformed = self.skinDeformNet(gps_pca, Js, poses)
		# Js_transformed = Js_transformed + trans.unsqueeze(1)
		# self.Js_3d = Js_transformed

		vtypes = []
		for batch in range(batch_num):
			batch_gtypes = gtypes[batch].unsqueeze(0)
			index = batch_gtypes.nonzero(as_tuple=False)[:, 1]
			vtype = 0
			if index[1] == 4 or index[1] == 5:
				vtype = 1
			vtypes.append(vtype)

		# self.gps_pca = gps_pca
		# self.lines = None
		lines = self.segimgEncoder(segimg, garlatents, vtypes)
		# self.lines = torch.Tensor(lines).to(imgs.device)
		if lines is not None:
			self.gps_pca = self.garment(self.lines, gps_pca, face_index, gtypes)

	def repose(self, imgs, weight, label, cam_k=None, **kwargs):
		# gtypes = self.gtypes
		# batch_num = self.batch_num
		# gps_pca = self.gps_pca
		# gps_pcas = torch.Tensor().to(gps_pca.device)
		# # gps_pcas2 = torch.Tensor().to(gps_pca.device)
		# body_ps = torch.Tensor().to(gps_pca.device)
		# start_vindex = 0
		# for batch in range(batch_num):
		# 	batch_gtypes = gtypes[batch].unsqueeze(0)
		# 	index = batch_gtypes.nonzero(as_tuple=False)[:, 1]
		# 	vert_num = self.garmentvnums[index[0]] + self.garmentvnums[index[1]]
		# 	batch_gps_pca = gps_pca[start_vindex:vert_num + start_vindex]
		#
		# 	##Repose
		# 	batch_body_ps, batch_gps_pca2, _, _ = self.smpl(shapes[batch].unsqueeze(0), pose_Rs[batch].unsqueeze(0),batch_gps_pca, batch_gtypes, True, False)
		#
		# 	batch_body_ps = batch_body_ps + trans[batch].unsqueeze(0).unsqueeze(1)
		# 	start_vindex += vert_num
		#
		# 	body_ps = torch.cat([body_ps, batch_body_ps], dim=0)
		# 	gps_pcas = torch.cat([gps_pcas, batch_gps_pca], dim=0)
		# 	# gps_pcas2 = torch.cat([gps_pcas2, batch_gps_pca2[0]], dim=0)
		#
		# # gps_pcas2 = gps_pcas2.unsqueeze(0)
		# # gps_pcas2 = gps_pcas2 + trans[imgbatch, :]
		# gps_pcas = gps_pcas.unsqueeze(0)
		# self.preds = gps_pcas + trans[self.imgbatch, :]
		gps_pca = self.gps_pca
		tris_infos = self.tris_infos
		# shapes = label['shape'].to(self.device).to(torch.float32)
		# poses = label['pose'].to(self.device).to(torch.float32)
		# trans = label['tran'].to(self.device).to(torch.float32)
		shapes = self.shapes
		poses = self.poses
		trans = self.trans
		gtypes = self.gtypes
		batch_num = self.batch_num
		ordered_imgbids = self.ordered_imgbids
		ordered_datas = self.ordered_datas#####
		ordered_gtypes = self.ordered_gtypes  #####
		pca_datas = self.pca_datas#####

		garbatch, edge_index, face_index, vf_vindex, vf_findex = self.extract_tris_infos(tris_infos)
		imgbatch = imgBatchFromGarBatch(garbatch, gtypes)
		Js, body_ns = self.skinDeformNet.skeleton(shapes, True)
		diss = (gps_pca.unsqueeze(1) - Js[imgbatch, :]).norm(dim=-1)
		# if self.skinWsNet.use_normal:
		# 	vnorms = compute_vnorms(gps_pca, face_index, vf_vindex, vf_findex)
		# 	ws = self.skinWsNet(torch.cat((gps_pca, vnorms, diss), dim=-1), edge_index, garbatch)
		# else:
		# 	ws = self.skinWsNet(torch.cat((gps_pca, diss), dim=-1), edge_index, garbatch)
		# weight = []
		# weight.append([np.load("/mnt/data/Project/GFLNet-master/data/motion_datas/neutral_datas/short_shirts/SPRING0088/1/garment_pca_weights.npy")])
		result = torch.cat(weight[0], 1).squeeze(0)
		if result.shape[0]==gps_pca.shape[0]:
			result = result.to(self.device)
			ws = result
			if cam_k is None:
				cam_k = torch.Tensor([[3.0375e+03, 0.0000e+00, 2.7000e+02], [0.0000e+00, 3.0375e+03, 2.7000e+02], [0.0000e+00, 0.0000e+00, 1.0000e+00]])
				cam_k = cam_k.to(self.device)
			deform_rec, transforms, pose_Rs, Js_transformed = self.skinDeformNet(gps_pca, Js, ws, poses, imgbatch)
			self.transforms = transforms
			deform_norms = compute_vnorms(deform_rec, face_index, vf_vindex, vf_findex)
			if 'pro_fs' in kwargs:
				pro_features = kwargs['pro_fs']
			else:
				deform_rect = deform_rec + trans[imgbatch, :]
				deform_rect_pros = self.pro_ps(deform_rect, cam_k)[:, :2]

				pro_patches = get_patchs_from_imgs(deform_rect_pros, imgs, imgbatch)
				pro_features = self.patchEncoder(pro_patches.reshape(-1, 3 * 32 * 32))
			self.pro_fs = pro_features
			ordered_datas2, _, ordered_gtypes2 = order_data_follow_gartypes([deform_rec, deform_norms, pro_features, transforms[:, :3, :3].reshape(-1, 9), transforms[:, :3, 3]],batch_num, garbatch, gtypes)
			assert (ordered_gtypes2 == ordered_gtypes)

			displacement_datas = []
			for ind, (pca_params, pca_ps), (shapes_gtype, _, latents_gtype), (deform_ps, deform_ns, pro_fs, Rs_gtype, Ts_gtype) in zip(ordered_gtypes2, pca_datas, ordered_datas, ordered_datas2):
				garvnum = self.garmentvnums[ind]
				size = pca_ps.shape[0]
				displacement_gtype = self.garDisplacementLayers[ind](torch.cat((pca_ps, deform_ps, deform_ns, Rs_gtype, Ts_gtype, shapes_gtype[:, None, :].expand(size, garvnum, 10), pca_params[:, None, :].expand(size, garvnum, 64), pro_fs, latents_gtype[:, None, :].expand(size, garvnum, -1)), dim=-1))
				displacement_datas.append([displacement_gtype])
			[displacements], _ = unorder_data_follow_imgbatch(displacement_datas, ordered_imgbids, ordered_gtypes2, batch_num)

			gps_diss = gps_pca + displacements

			deform_rec = gps_diss

			tmps = torch.cat((deform_rec, deform_rec.new_ones(deform_rec.shape[0], 1)), dim=-1).unsqueeze(-1)
			gps_rec = torch.matmul(transforms, tmps).squeeze(-1)[:, :3]
			gps_rec = gps_rec + trans[imgbatch, :]
			Js_transformed = Js_transformed + trans.unsqueeze(1)
			body_ps, _, _ = self.smpl(shapes, pose_Rs, True, False)
			body_ps = body_ps + trans.unsqueeze(1)

			self.preds = gps_rec
			self.body_ps = body_ps
		else:
			self.preds = None
			self.body_ps = None

	def get_res(self):

		return {
			'preds': self.preds,
			'body_ps':self.body_ps,
			'face_index':self.face_index,
			'gtypes':self.gtypes}

	def get_error(self, labels, labels_lines, alpha_labels):
		'''
        Hourglass has its own intermediate supervision scheme
        '''
		loss_func = torch.nn.MSELoss()
		loss_softmax = torch.nn.CrossEntropyLoss()
		# accumulate errors from all the latent feature layers
		loss = 0
		num_pre = 0
		batch_num = self.batch_num
		gtypes = self.gtypes

		for i in range(batch_num):
			label_displacement = [[0, 0, 0]]
			up_prob = self.up_gar_prob[i].unsqueeze(0)
			bottom_prob = self.bottom_gar_prob[i].unsqueeze(0)
			u_index = self.garmentvnums.index(labels['up'][i].shape[0])
			d_index = self.garmentvnums.index(labels['bottom'][i].shape[0])
			right_type_num = 0
			up_err_type = 0
			bottom_err_type = 0
			index = np.where(gtypes[i].cpu().numpy() == 1)[0]
			if gtypes[i][u_index] == 1:
				right_type_num += 1
				if u_index == 0:
					up_err_type = 1
				elif u_index == 1:
					up_err_type = 2
			if gtypes[i][d_index] == 1:
				right_type_num += 1
				if d_index == 2:
					bottom_err_type = 3
				elif d_index == 3:
					bottom_err_type = 4
				elif d_index == 4:
					bottom_err_type = 5
				else:
					bottom_err_type = 6
			up_type = [u_index]
			bottom_type = [d_index - 2]
			loss_soft = loss_softmax(up_prob, torch.Tensor(up_type).long().to(self.device)) + loss_softmax(bottom_prob, torch.Tensor(bottom_type).long().to(self.device))
			loss = loss + loss_func(self.shapes[i].to(torch.float32), labels['shape'][i].to(self.device).to(torch.float32)) + loss_func(self.trans[i].to(torch.float32), labels['tran'][i].to(self.device).to(torch.float32)) + loss_func(self.poses[i].to(torch.float32), labels['pose'][i].to(self.device).to(torch.float32)) + loss_soft  # compute loss for every net
			label_displacement = np.concatenate((label_displacement, labels['up'][i]), axis=0)
			label_displacement = np.concatenate((label_displacement, labels['bottom'][i]), axis=0)

			label_displacement = np.delete(label_displacement, 0, axis=0)
			num = self.garmentvnums[index[0]] + self.garmentvnums[index[1]]
			gps_pca_ = self.preds[num_pre:num_pre + num, :]
			num_pre = num_pre + num
			line = None
			if self.lines is not None:
				line = self.lines[i]
				if line[-1] == 100:
					line = line[:-1]
				labels_line = torch.Tensor(labels_lines[i]).to(self.device)
			pca_param = torch.cat([self.pca_params[i * 2 + 0].unsqueeze(0), self.pca_params[i * 2 + 1].unsqueeze(0)], dim=0)
			alpha_label = torch.squeeze(torch.tensor([item.cpu().detach().numpy() for item in alpha_labels[i]]).cuda().to(self.device))
			loss_alpha = loss_func(alpha_label.to(torch.float32), pca_param.to(torch.float32))
			loss += loss_alpha
			if right_type_num != 0:
				if up_err_type == 1 or up_err_type == 2:
					if line is not None:
						loss_line = loss_func(line[:6], labels_line[:6] * 10)
						loss += loss_line
					loss += loss_func(gps_pca_[:self.garmentvnums[up_err_type - 1], :].to(torch.float32), torch.from_numpy(label_displacement).to(self.device)[:self.garmentvnums[up_err_type - 1],:].to(torch.float32))
				if bottom_err_type == 3 or bottom_err_type == 4:
					if line is not None:
						loss_line = loss_func(line[6:], labels_line[6:] * 10)
						loss += loss_line
					loss += loss_func(gps_pca_[-1 * self.garmentvnums[bottom_err_type - 1]:, :].to(torch.float32), torch.from_numpy(label_displacement).to(self.device)[-1 * self.garmentvnums[bottom_err_type - 1]:, :].to(torch.float32))
				elif bottom_err_type == 5 or bottom_err_type == 6:
					if line is not None:
						loss_line = loss_func(line[6:], labels_line[6:] * 10)
						loss += loss_line
					loss += loss_func(gps_pca_[-1 * self.garmentvnums[bottom_err_type - 1]:, :].to(torch.float32), torch.from_numpy(label_displacement).to(self.device)[-1 * self.garmentvnums[bottom_err_type - 1]:, :].to(torch.float32))
		return loss

	def forward(self, image, segImage, label, labels_lines, alpha_labels, weight_labels, gtypes=None):
		self.gtypes = gtypes
		# Phase 1: get image feature
		self.filter(image, segImage, labels_lines, alpha_labels)

		# Phase 2: repose
		self.repose(image, weight_labels, label[0])

		# Phase 3: get the prediction
		res = self.get_res()  # return self.preds, (B, 1, n_in+n_out)

		# Phase 4: get the error, default is nn.MSELoss()
		# error = self.get_error(label[0], labels_lines, alpha_labels)  # R
		error=None
		# gtypes = label_types

		# lines = None
		return res, error

