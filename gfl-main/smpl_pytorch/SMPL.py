import torch
import json
import sys
import numpy as np
from smpl_pytorch.util import batch_global_rigid_transformation, batch_rodrigues, reflect_pose
import torch.nn as nn
import os
import pickle
import math
import torch.nn.functional as F

def dist3d(s, d):
    return math.sqrt((s[0]-d[0])**2+(s[1]-d[1])**2+(s[2]-d[2])**2)

class SMPL(nn.Module):
	def __init__(self, model_path, joint_type='cocoplus', obj_saveable=False):
		super(SMPL, self).__init__()

		if joint_type not in ['cocoplus', 'lsp']:
			msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
			sys.exit(msg)

		self.model_path = model_path
		self.joint_type = joint_type
		with open(model_path, 'r') as reader:
			model = json.load(reader)

		if obj_saveable:
			self.faces = model['f']
		else:
			self.faces = None

		np_v_template = np.array(model['v_template'], dtype=np.float)
		self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
		self.size = [np_v_template.shape[0], 3]

		np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
		self.num_betas = np_shapedirs.shape[-1]
		np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
		self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

		np_J_regressor = np.array(model['J_regressor'], dtype=np.float)
		self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

		np_posedirs = np.array(model['posedirs'], dtype=np.float)
		num_pose_basis = np_posedirs.shape[-1]
		np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
		self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

		self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

		np_joint_regressor = np.array(model['cocoplus_regressor'], dtype=np.float)
		if joint_type == 'lsp':
			self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
		else:
			self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

		np_weights = np.array(model['weights'], dtype=np.float)

		vertex_count = np_weights.shape[0]
		vertex_component = np_weights.shape[1]

		# batch_size = 10
		# np_weights = np.tile(np_weights, (batch_size, 1))
		self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

		self.register_buffer('e3', torch.eye(3).float())

		self.cur_device = None

	def save_obj(self, verts, obj_mesh_name):
		if not self.faces:
			msg = 'obj not saveable!'
			sys.exit(msg)

		with open(obj_mesh_name, 'w') as fp:
			for v in verts:
				fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))

			for f in self.faces:  # Faces are 1-based, not 0-based in obj files
				fp.write('f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))

	# output joints is not smpl self skeleton, is regressor for h3.6m Mscoco,see HMR paper
	# smpl self skeleton should be self.J_transformed
	def forward(self, beta, theta, v_cloth, gtypes=None, get_skin = False, theta_in_rodrigues=True):
		if not self.cur_device:
			device = beta.device
			self.cur_device = torch.device(device.type, device.index)

		num_batch = beta.shape[0]

		# v_cloths = self.v_template
		# for i in v_cloths:
		# 	i[1] -= 0.04
		# self.v_template = v_cloths

		v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
		Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
		Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
		Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
		J = torch.stack([Jx, Jy, Jz], dim=2)
		if theta_in_rodrigues:
			Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
		else:  # theta is already rotations
			Rs = theta.view(-1, 24, 3, 3)

		pose_feature = (Rs[:, 1:, :, :]).sub(self.e3, alpha=1.0).view(-1, 207)
		pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
		v_posed = pose_params + v_shaped
		# garment deformation
		# if v_cloth is not None :#and garment_class is not None:
		# 	garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
		# 	index = gtypes.nonzero(as_tuple=False)[:, 1]
		# 	up_vertices = v_cloth[:garmentvnums[index[0]]]
		# 	bottom_vertices = v_cloth[garmentvnums[index[0]]:]
		# 	vertices = [up_vertices, bottom_vertices]
		# 	verts_indices = []
		# 	v_deformed = self.v_template.clone()
		# 	v_deformed = v_deformed.unsqueeze(0)
		# 	for n in range(2):
		# 		if not os.path.exists(str(garmentvnums[index[n]]) + ".txt"):
		# 			with open(str(garmentvnums[index[n]]) + ".txt", "w") as f:
		# 				tmp_indices = []
		# 				for i in range(vertices[n].shape[0]):
		# 					min_index = 0
		# 					min_value = dist3d(vertices[n][i], v_deformed[0][0])
		# 					for j in range(v_deformed.shape[1]):
		# 						tmp_value = dist3d(vertices[n][i], v_deformed[0][j])
		# 						if min_value > tmp_value:
		# 							min_value = tmp_value
		# 							min_index = j
		# 					f.write(str(min_index) + ',')
		# 					tmp_indices.append(min_index)
		# 				verts_indices.append(np.array(tmp_indices))
		# 		else:
		# 			with open(str(garmentvnums[index[n]]) + ".txt", "r") as f:
		# 				for line in f.readlines():
		# 					line = line.split(',')
		# 					line = list(filter(None, line))
		# 					line = [int(i) for i in line]
		# 					verts_indices.append(line)
		# 	verts_indices = torch.Tensor(sum(verts_indices, [])).clone().type(torch.int64)
		# 	v_deformed[:, verts_indices] += garment_d

		self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=False)



		# Calculate closest SMPL vertex for each vertex of the cloth mesh
		# with torch.no_grad():
		# 	dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2)) ** 2).sum(-1)
		# 	correspondance = torch.argmin(dists, 2)

		garmentvnums = [4248, 4258, 5327, 3721, 5404, 2818]
		index = gtypes.nonzero(as_tuple=False)[:, 1]
		correspondance = []
		for n in range(2):
			with open(str(garmentvnums[index[n]]) + ".txt", "r") as f:
				for line in f.readlines():
					line = line.split(',')
					line = list(filter(None, line))
					line = [int(i) for i in line]
					correspondance.append(line)
		correspondance = torch.Tensor(sum(correspondance, [])).clone().type(torch.int64)

		v_shaped_cloth = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0, correspondance] + v_cloth
		v_posed_cloth = pose_params[0, correspondance] + v_shaped_cloth

		# weight = self.weight[:num_batch]
		# W = weight.view(num_batch, -1, 24)
		W = self.weight.expand(num_batch, *self.weight.shape[1:])
		T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

		v_posed_cloth = v_posed_cloth.unsqueeze(0)
		v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
		v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device=self.cur_device)], dim=2)

		v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
		v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))

		verts = v_homo[:, :, :3, 0]
		verts_cloth = v_homo_cloth[:, :, :3, 0]

		joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
		joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
		joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

		joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

		# if trans is not None:
		# 	verts + trans.unsqueeze(1)
		# if garment_class is not None:
		# 	v_posed_homo = torch.cat([
		# 		v_deformed,
		# 		torch.ones(num_batch, v_deformed.shape[1], 1, device=self.cur_device)], dim=2)
		# 	v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
		# 	v_garment = v_homo[:, :, :3, 0]
		# 	if trans is not None:
		# 		v_garment + trans.unsqueeze(1)
		# 	if get_skin:
		# 		return verts, joints, Rs, v_garment[:, verts_indices]
		# 	else:
		# 		return joints, v_garment[:, verts_indices]
		# else:
		# 	if get_skin:
		# 		return verts, joints, Rs
		# 	else:
		# 		return joints
		if get_skin:
			return verts, verts_cloth, joints, Rs
		else:
			return joints

	def skeleton(self, beta, require_body=False):
		num_batch = beta.shape[0]
		v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
		Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
		Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
		Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
		J = torch.stack([Jx, Jy, Jz], dim=2)
		if require_body:
			return J, v_shaped
		else:
			return J

	def deform_clothed_smpl(self, theta, J, v_smpl, v_cloth, is_Rotation=False):
		assert len(theta) == 1, 'currently we only support batchsize=1'
		num_batch = 1

		device = theta.device
		self.cur_device = torch.device(device.type, device.index)

		if theta.numel() == num_batch * 24 * 3:
			Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24, 3, 3)
		elif theta.numel() == num_batch * 24 * 9:
			# input poses are general matrix
			if not is_Rotation:
				ms = theta.reshape(-1, 3, 3)
				# use gram schmit regularization
				b1 = F.normalize(ms[:, :, 0], dim=1)
				dot_prod = torch.sum(b1 * ms[:, :, 1], dim=1, keepdim=True)
				b2 = F.normalize(ms[:, :, 1] - dot_prod * b1, dim=-1)
				b3 = torch.cross(b1, b2, dim=1)
				Rs = torch.stack([b1, b2, b3], dim=-1).reshape(num_batch, 24, 3, 3)
			else:
				Rs = theta.reshape(batch_num, 24, 3, 3)
		elif theta.numel() == num_batch * 24 * 16:
			A = theta.reshape(num_batch, 24, 4, 4)
			Js_transformed = None
			Rs = None
		if Rs is not None:
			pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

		pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
		v_posed_smpl = pose_params + v_smpl

		# Calculate closest SMPL vertex for each vertex of the cloth mesh
		with torch.no_grad():
			dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2)) ** 2).sum(-1)
			correspondance = torch.argmin(dists, 2)

		v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth

		if Rs is not None:
			self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=False)

		W = self.weight.expand(num_batch, *self.weight.shape[1:])
		T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

		v_posed_homo_smpl = torch.cat(
			[v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device=self.cur_device)], dim=2)
		v_posed_homo_cloth = torch.cat(
			[v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device=self.cur_device)], dim=2)
		v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
		v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))
		verts_smpl = v_homo_smpl[:, :, :3, 0]
		verts_cloth = v_homo_cloth[:, :, :3, 0]

		return verts_smpl, verts_cloth

def getSMPL():
	return SMPL(os.path.normpath(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt')), obj_saveable = True)
def getTmpFile():
	return os.path.join(os.path.dirname(__file__),'hello_smpl.obj')

if __name__ == '__main__':
	device = torch.device('cuda', 1)
	# smpl = SMPL('/home/jby/pytorch_ext/smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt',obj_saveable=True)
	smpl = SMPL(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt'), obj_saveable = True).to(device)
	pose= np.array([
			1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
			-1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
			-1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
			1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
			2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
			7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
			-5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
			-5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
			-7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
			9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
			-9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
			-1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
			-1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
			-8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
			-4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
			3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
			-3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
			6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
			-6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
			4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
			2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
			-1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
			-3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
			3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)
		
	beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
			0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])

	vbeta = torch.tensor(np.array([beta])).float().to(device)
	vpose = torch.tensor(np.array([pose])).float().to(device)

	verts, j, r = smpl(vbeta, vpose, get_skin = True)

	smpl.save_obj(verts[0].cpu().numpy(), './mesh.obj')

	rpose = reflect_pose(pose)
	vpose = torch.tensor(np.array([rpose])).float().to(device)
	
	verts, j, r = smpl(vbeta, vpose, get_skin = True)
	smpl.save_obj(verts[0].cpu().numpy(), './rmesh.obj')

