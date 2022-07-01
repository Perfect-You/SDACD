import logging
import torch
import torch.utils.data
from torch.utils.data import distributed, DataLoader
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_path_loader_for_txt, full_test_loader, full_test_loader_for_txt, CDDloader, CDDloader_for_txt)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss
from models.Models import Siam_NestedUNet_Conc, SNUNet_ECAM
from models.models_gan import GeneratorResNet, Discriminator,Discriminator_result, weights_init_normal
import random
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO)

def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'cd_miou':[],
        'learning_rate': [],
    }

    return metrics


def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['cd_miou'].append(cd_report[3])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['cd_miou'].append(cd_report[3])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader_for_txt(opt.train_txt_path, opt.val_txt_path)
    # train_full_load, val_full_load = full_path_loader_for_txt(opt.dataset_dir)


    train_dataset = CDDloader_for_txt(train_full_load, opt, aug=opt.augmentation)
    val_dataset = CDDloader_for_txt(val_full_load, opt, aug=False)

    logging.info('STARTING Dataloading')

    if opt.distributed:
        train_sampler = distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, pin_memory=True, shuffle=(train_sampler is None), batch_size=opt.batch_size, sampler=train_sampler, num_workers=opt.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader,train_sampler, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader_for_txt(opt.test_txt_path)

    test_dataset = CDDloader_for_txt(test_full_load, opt, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))




def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_model(opt, device):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    model = SNUNet_ECAM(opt.num_channel, 2, opt.sync_bn)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)
    # device_ids = list(range(opt.num_gpus))
    # model = SNUNet_ECAM(opt.num_channel, 2).to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)

    return model

def load_model_test(opt, device):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    device_ids = list(range(opt.num_gpus))
    model = SNUNet_ECAM(opt.num_channel, 2).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def load_gan_generator(opt, device):
    input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    model = GeneratorResNet(input_shape, opt.n_residual_blocks)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)
    model.apply(weights_init_normal)

    # input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    # model = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
    # device_ids = list(range(opt.num_gpus))
    # model = nn.DataParallel(model, device_ids=device_ids)


    return model

def load_gan_generator_test(opt, device):
    input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    model = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
    device_ids = list(range(opt.num_gpus))
    model = nn.DataParallel(model, device_ids=device_ids)


    return model


def load_gan_discrimitor(opt, dev):
    input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    model = Discriminator(input_shape)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)
    model.apply(weights_init_normal)

    # input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    # model = Discriminator(input_shape).to(dev)
    # device_ids = list(range(opt.num_gpus))
    # model = nn.DataParallel(model, device_ids=device_ids)

    return model

def load_gan_discrimitor_result(opt, dev):
    input_shape = (opt.num_class, opt.load_size, opt.load_size)
    model = Discriminator_result(input_shape)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)
    model.apply(weights_init_normal)

    # input_shape = (opt.num_channel, opt.load_size, opt.load_size)
    # model = Discriminator(input_shape).to(dev)
    # device_ids = list(range(opt.num_gpus))
    # model = nn.DataParallel(model, device_ids=device_ids)

    return model
