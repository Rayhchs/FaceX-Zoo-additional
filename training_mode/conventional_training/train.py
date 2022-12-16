"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
"""
Some modification by: Ray Huang
"""
import os
import sys
import shutil
import argparse
import logging as logger
import random, time 


import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
seed = 1000
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from test_protocol.eval_lfw import evaluation


logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class LinearEmbedding(nn.Module):
    def __init__(
        self, base, feature_size=512, embedding_size=128, l2norm_on_train=True
    ):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)
        self.l2norm_on_train = l2norm_on_train

    def forward(self, x):
        feat, _ = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)

        if self.training and (not self.l2norm_on_train):
            return embedding

        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding


class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory, conf, pretrain_state=None):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()
        self.head_type = conf.head_type
        self.pretrain_state = pretrain_state
        if self.head_type.lower() == 'broadface':
            self.backbone.load_state_dict(self.pretrain_state)
            self.model = LinearEmbedding(
                self.backbone,
                feature_size=512,
                embedding_size=128,
                l2norm_on_train=False)

    def forward(self, data, label):
        if self.head_type.lower() == 'adaface':
            feat, norm = self.backbone.forward(data)
            pred = self.head.forward(feat, norm, label)

        elif self.head_type.lower() == 'elasticcosface' or self.head_type.lower() == 'elasticarcface':
            output, _ = self.backbone.forward(data)
            feat = F.normalize(output)
            pred = self.head.forward(feat, label)
        elif self.head_type.lower() == 'broadface':
            feat, _ = self.model.forward(data)
            pred = self.head.forward(feat, label)
        else:
            feat, _ = self.backbone.forward(data)
            pred = self.head.forward(feat, label)
        return pred

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf, eval_value):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        elif conf.head_type == 'MagFace':
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            loss = criterion(outputs, labels) + loss_g
        elif conf.head_type == 'BroadFace':
            loss = model.forward(images, labels)
            loss = torch.unsqueeze(loss, 0)
            loss = torch.sum(loss) / 4 ### For 4 GPUs
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if (batch_idx + 1) % conf.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
            try:
                eval_output = evaluation(test_set='CPLFW', data_conf_file='../../test_protocol/data_conf.yaml', 
                                         backbone_type=conf.backbone_type, backbone_conf_file=conf.backbone_conf_file,
                                         model_path=os.path.join(conf.out_dir, saved_name), batch_size=512, head_type=conf.head_type)
                os.remove(os.path.join(conf.out_dir, saved_name))
                if eval_output >= eval_value:
                    torch.save(state, os.path.join(conf.out_dir, 'best.pt'))
                    logger.info('Save checkpoint %s to disk.' % 'best.pt')
                    eval_value = eval_output
                logger.info(f'CPLFW accuracy: {eval_output}')
            except:
                logger.info('best.pt not saved')
            logger.info(f'CPLFW accuracy: {eval_output}')

    saved_name = 'Epoch_%d.pt' % cur_epoch
    state = {'state_dict': model.module.state_dict(), 
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)
    try:
        eval_output = evaluation(test_set='CPLFW', data_conf_file='../../test_protocol/data_conf.yaml', 
                                    backbone_type=conf.backbone_type, backbone_conf_file=conf.backbone_conf_file,
                                    model_path=os.path.join(conf.out_dir, saved_name), batch_size=512, head_type=conf.head_type)
        if eval_output >= eval_value:
            torch.save(state, os.path.join(conf.out_dir, 'best.pt'))
            logger.info('Save checkpoint %s to disk.' % 'best.pt')
            eval_value = eval_output
        logger.info(f'CPLFW accuracy: {eval_output}')
    except:
        logger.info('best.pt not saved')
    return eval_value

def train(conf):
    """Total training procedure.
    """
    data_loader = DataLoader(ImageDataset(conf.data_root, conf.train_file), 
                             conf.batch_size, True, num_workers = 4)
    conf.device = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file, conf.head_type)    
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    if conf.head_type == 'BroadFace' and not conf.resume:
        sys.exit("BroadFace can only be used for finetune")
    if not conf.head_type == 'BroadFace':
        model = FaceModel(backbone_factory, head_factory, conf)

    ori_epoch = 0
    if conf.resume:
        ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
        state_dict = torch.load(args.pretrain_model)['state_dict']
        if not conf.head_type == 'BroadFace':
            model.load_state_dict(state_dict)
        else:
            backbone = backbone_factory.get_backbone()
            model_dict = backbone.state_dict()
            new_pretrained_dict = {}
            for k in model_dict:
                new_pretrained_dict[k] = state_dict['backbone.'+k]
            model_dict.update(new_pretrained_dict)
            model = FaceModel(backbone_factory, head_factory, conf, model_dict)

    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)
    loss_meter = AverageMeter()
    model.train()
    eval_value = 0
    for epoch in range(ori_epoch, conf.epoches):
        eval_value = train_one_epoch(data_loader, model, optimizer, 
                        criterion, epoch, loss_meter, conf, eval_value)
        lr_schedule.step()                        

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str, 
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, 
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')

    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
