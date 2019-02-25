import argparse

class optionsTrain(object):
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="PS-FCN args")
		self.initialize()
	def initialize(self):
		## Primary Args
		self.parser.add_argument('--cuda', default=True, action='store_false')
		self.parser.add_argument('--seed', default=123, type=int)
		self.parser.add_argument('--workers', default = 8, type=int)
		self.parser.add_argument('--in_light', default=True, action='store_false')
		self.parser.add_argument('--use_BN', default=False, action='store_true')
		self.parser.add_argument('--in_img_num', default=32, type=int)
		self.parser.add_argument('--epochs', default=30, type=int)
		self.parser.add_argument('--dilation', default=1, type=int)
		self.parser.add_argument('--masked_loss', default=False, action='store_true')        
		
		## Secondary Args
		self.parser.add_argument('--data_dir', default='./data/datasets/PS_Blobby_Dataset')
		self.parser.add_argument('--rescale', default=True, action='store_false')
		self.parser.add_argument('--crop', default=True, action='store_false')
		self.parser.add_argument('--crop_h', default=32, type=int)
		self.parser.add_argument('--crop_w', default=32, type=int)
		self.parser.add_argument('--noise_aug', default=True, action='store_false')
		self.parser.add_argument('--noise', default=0.05, type=float)
		self.parser.add_argument('--color_aug', default=True, action='store_false') 

		self.parser.add_argument('--model', default='PS_FCN')
		self.parser.add_argument('--milestones', default=[5, 10, 15, 20, 25], nargs='+', type=int)
		self.parser.add_argument('--init_lr', default=1e-3, type=float)
		self.parser.add_argument('--lr_decay', default=0.5, type=float)
		self.parser.add_argument('--beta_1', default=0.9, type=float, help='adam')
		self.parser.add_argument('--beta_2', default=0.999, type=float, help='adam')
		self.parser.add_argument('--batch', default=32, type=int)

	
	def parse(self):
		self.args = self.parser.parse_args()
		return self.args

class optionsTest(object):
	
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="PS-FCN args")
		self.initialize()
	def initialize(self):
		## Primary Args
		self.parser.add_argument('--cuda', default=True, action='store_false')
		self.parser.add_argument('--seed', default=123, type=int)
		self.parser.add_argument('--workers', default = 8, type=int)
		self.parser.add_argument('--save_root', default='data/Training/')
		self.parser.add_argument('--retrain', default=None)		
		self.parser.add_argument('--model_path', default='./TrainedModels/model.pth.tar')

		self.parser.add_argument('--in_light', default=True, action='store_false')
		self.parser.add_argument('--use_BN', default=False, action='store_true')
		self.parser.add_argument('--dilation', default=1, type=int)
		self.parser.add_argument('--in_img_num', default=32, type=int)
		self.parser.add_argument('--epochs', default=30, type=int)
		
		## Testing Args
		self.parser.add_argument('--run_model', default=True, action='store_false')
		self.parser.add_argument('--bm_dir', default='./data/DiLiGenT/pmsData')
		self.parser.add_argument('--model', default='PS_FCN_run')
		self.parser.add_argument('--test_batch', default=1, type=int)

	
	def parse(self):
		self.args = self.parser.parse_args()
		return self.args
