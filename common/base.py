import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from main.config import cfg
from main.model import get_model

# dynamic dataset import
exec('from ' + cfg.trainset + ' import ' + cfg.trainset)
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self, model_cfg):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
        self.cfg = model_cfg

    def get_optimizer(self, model):
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=self.cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr * (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr * (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        train_dataset = eval(self.cfg.trainset)(transforms.ToTensor(), "train")

        self.itr_per_epoch = math.ceil(len(train_dataset) / self.cfg.num_gpus / self.cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=train_dataset, batch_size=self.cfg.num_gpus*self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model()

        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)


        if self.cfg.pretrain:
            model.load_state_dict(torch.load(self.cfg.pretrain_cpt)['network'])
            self.logger.info('Load pretrain model from {}'.format(self.cfg.pretrain_cpt))
        if self.cfg.stage_seg:
            model.load_state_dict(torch.load(self.cfg.stage_seg_cpt)['network'], strict=False)
            self.logger.info('Load stage_seg model from {}'.format(self.cfg.stage_seg_cpt))

        if self.cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self, dataset=None):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        if dataset is None:
            self.test_dataset = eval(cfg.testset)(transforms.ToTensor(), "test")
        else:
            self.test_dataset = dataset
        self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
       
    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model()
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, test_epoch):
        self.test_dataset.print_eval_result(test_epoch)
