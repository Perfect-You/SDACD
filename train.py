import datetime
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion, load_gan_generator, load_gan_discrimitor,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics, LambdaLR, ReplayBuffer, load_gan_discrimitor_result)
from utils.metrics import Evaluator
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import time
import itertools
from torch.autograd import Variable
import torchvision.transforms as transforms


if __name__ == '__main__':

    """
    Initialize Parser and define arguments
    """
    print("start")
    parser, metadata = get_parser_with_args()
    opt = parser.parse_args()

    """
    Initialize experiments log
    """
    logging.basicConfig(level=logging.INFO)
    os.makedirs(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/', exist_ok=True)
    path = opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    writer = SummaryWriter(path)

    """
    Set up environment: define paths, download data, and set device
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        opt.cuda = True
    else:
        dev = torch.device('cpu')
        opt.cuda = False
    # dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

    # ##############################################################################################
    if opt.cuda:
        try:
            opt.gpu_ids = [int(s) for s in opt.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    num_gpus = len(opt.gpu_ids)
    opt.distributed = num_gpus>1

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        device_ids = opt.gpu_ids
        ngpus_per_node=len(device_ids)
        opt.batch_size = int(opt.batch_size/ngpus_per_node)

    if opt.sync_bn is None:
        if opt.cuda and len(opt.gpu_ids) > 1:
            opt.sync_bn = True
        else:
            opt.sync_bn = False
    # ################################################################################################


    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(seed=777)

    train_loader,train_sampler, val_loader = get_loaders(opt)

    """
    Load Model then define other aspects of the model
    """
    logging.info('LOADING Model')
    model = load_model(opt, dev)
    G_AB = load_gan_generator(opt, dev)
    G_BA = load_gan_generator(opt, dev)
    D_A = load_gan_discrimitor(opt, dev)
    D_B = load_gan_discrimitor(opt, dev)
    D_C = load_gan_discrimitor_result(opt, dev)
    opt.start_epoch = 0

    criterion = get_criterion(opt)
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    if opt.cuda:
        # criterion.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.resume_cd is not None:
        if not os.path.isfile(opt.resume_cd):
            raise RuntimeError("=> no checkpoint found at '{}'".format(opt.resume_cd))
        checkpoint_cd = torch.load(opt.resume_cd, map_location='cpu')
        checkpoint_g_ab = torch.load(opt.resume_g_ab, map_location='cpu')
        checkpoint_g_ba = torch.load(opt.resume_g_ba, map_location='cpu')
        checkpoint_d_a = torch.load(opt.resume_d_a, map_location='cpu')
        checkpoint_d_b = torch.load(opt.resume_d_b, map_location='cpu')
        opt.start_epoch = int(opt.resume_cd.split('.')[0].split('/')[-1].split('_')[-1]) + 1
        if opt.cuda:
            model.load_state_dict(checkpoint_cd)
            G_AB.load_state_dict(checkpoint_g_ab)
            G_BA.load_state_dict(checkpoint_g_ba)
            D_A.load_state_dict(checkpoint_d_a)
            D_B.load_state_dict(checkpoint_d_b)
        else:
            model.load_state_dict(checkpoint_cd)
        print("=> loaded checkpoint '{}' (epoch {})" .format(opt.resume_cd, opt.start_epoch))

    # if you pre-train the GAN or CDNet, use these code to load the pre-trained model
    # checkpoint_g_ab = torch.load(opt.resume_g_ab,map_location='cpu')
    # checkpoint_g_ba = torch.load(opt.resume_g_ba,map_location='cpu')
    # checkpoint_d_a = torch.load(opt.resume_d_a,map_location='cpu')
    # checkpoint_d_b = torch.load(opt.resume_d_b,map_location='cpu')
    # G_AB.load_state_dict(checkpoint_g_ab)
    # G_BA.load_state_dict(checkpoint_g_ba)
    # D_A.load_state_dict(checkpoint_d_a)
    # D_B.load_state_dict(checkpoint_d_b)
    #
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(opt.pretrain_cd,map_location='cpu')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    
    # # G_AB.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_g_ab.items()})
    # # G_BA.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_g_ba.items()})
    # # D_A.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_d_a.items()})
    # # D_B.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_d_b.items()})
    # print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume_g_ab, opt.start_epoch))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)  # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70,80,90,100,105,110], 0.5)
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr_gan, betas=(opt.gan_b1, opt.gan_b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr_gan, betas=(opt.gan_b1, opt.gan_b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr_gan, betas=(opt.gan_b1, opt.gan_b2))
    optimizer_D_C = torch.optim.Adam(D_C.parameters(), lr=opt.lr_gan, betas=(opt.gan_b1, opt.gan_b2))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_C = torch.optim.lr_scheduler.LambdaLR(optimizer_D_C, lr_lambda=LambdaLR(opt.epochs, opt.start_epoch, opt.decay_epoch).step)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

    fake_A_buffer = ReplayBuffer()
    recov_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    recov_B_buffer = ReplayBuffer()
    fake_C_buffer_1 = ReplayBuffer()
    fake_C_buffer_2 = ReplayBuffer()
    fake_C_buffer_3 = ReplayBuffer()


    def unnormalize(tensor):
        tensor = tensor.clone()  # avoid modifying tensor in-place

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t):
            norm_ip(t, float(t.min()), float(t.max()))

        norm_range(tensor)

        return tensor

    transform1 = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])


    """
     Set starting values
    """
    best_metrics = {'cd_f1scores_fusion': -1, 'cd_recalls_fusion': -1, 'cd_precisions_fusion': -1}
    logging.info('STARTING training')
    total_step = -1

    for epoch in range(opt.start_epoch, opt.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()
        evaluator_1 = Evaluator(opt.num_class)
        evaluator_2 = Evaluator(opt.num_class)
        evaluator_3 = Evaluator(opt.num_class)
        evaluator_feature_fusion = Evaluator(opt.num_class)

        """
        Begin Training
        """
        # model.train()

        logging.info('SET model mode to train!')
        confusion_matrix = torch.zeros(opt.num_class, opt.num_class)
        batch_iter = 0
        train_loss = 0.0
        tbar = tqdm(train_loader)
        loss_print = []


        loss_G_print = []
        loss_GAN_print = []
        loss_cycle_print = []
        loss_identity_print = []
        loss_D_print = []

        loss_D_C_print = []

        for i, [batch_img1, batch_img2, labels] in enumerate(tbar):
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            total_step += 1
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            model.train()
            optimizer.zero_grad()

            real_A = Variable(batch_img1.type(Tensor))
            real_B = Variable(batch_img2.type(Tensor))

            valid = Variable(torch.full([real_A.size(0), *D_C.module.output_shape],1/3),requires_grad=False).to(dev)

            fake_B = G_AB(real_A).detach()
            fake_A = G_BA(real_B).detach()
            
            real_A_norm2 = unnormalize(real_A)
            real_A_norm2=transform1(real_A_norm2)
            real_B_norm2 = unnormalize(real_B)
            real_B_norm2 = transform1(real_B_norm2)
            fake_A_norm2 = unnormalize(fake_A)
            fake_A_norm2 = transform1(fake_A_norm2)
            fake_B_norm2 = unnormalize(fake_B)
            fake_B_norm2 = transform1(fake_B_norm2)

            [cd_preds_1, cd_preds_2, cd_preds_3, cd_preds] = model(real_A_norm2, real_B_norm2, fake_B_norm2, fake_A_norm2)

            cd_loss = criterion(cd_preds_1, labels) + criterion(cd_preds_2, labels) + criterion(cd_preds_3, labels) + criterion(cd_preds, labels)
            loss_CD_GAN_1 = criterion_GAN(D_C(cd_preds_1[-1]), valid)
            loss_CD_GAN_2 = criterion_GAN(D_C(cd_preds_2[-1]), valid)
            loss_CD_GAN_3 = criterion_GAN(D_C(cd_preds_3[-1]), valid)
            cd_loss += ((loss_CD_GAN_1 + loss_CD_GAN_2 + loss_CD_GAN_3) / 3) * 0.1

            loss_print.append(cd_loss.data.cpu().numpy())

            cd_loss.backward()
            optimizer.step()

            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            D_test = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            # loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_GAN_cycle_A = criterion_GAN(D_A(recov_A),valid)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_GAN_cycle_B = criterion_GAN(D_B(recov_B),valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA + loss_GAN_cycle_A + loss_GAN_cycle_B) / 4

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            loss_G_print.append(loss_G.data.cpu().numpy())
            loss_GAN_print.append(loss_GAN.data.cpu().numpy())
            loss_cycle_print.append(loss_cycle.data.cpu().numpy())
            loss_identity_print.append(loss_identity.data.cpu().numpy())

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            recov_A_ = recov_A_buffer.push_and_pop(recov_A)
            loss_fake_cycle = criterion_GAN(D_A(recov_A_.detach()),fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake + loss_fake_cycle) / 3

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            recov_B_ = recov_B_buffer.push_and_pop(recov_B)
            loss_fake_cycle = criterion_GAN(D_B(recov_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake + loss_fake_cycle) / 3

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2
            loss_D_print.append(loss_D.data.cpu().numpy())

            valid_1 = torch.cat([torch.ones([real_A.size(0), *D_A.module.output_shape]),torch.zeros([real_A.size(0), *D_A.module.output_shape]),torch.zeros([real_A.size(0), *D_A.module.output_shape])],dim=1).to(dev)
            valid_2 = torch.cat([torch.zeros([real_A.size(0), *D_A.module.output_shape]),torch.ones([real_A.size(0), *D_A.module.output_shape]),torch.zeros([real_A.size(0), *D_A.module.output_shape])],dim=1).to(dev)
            valid_3 = torch.cat([torch.zeros([real_A.size(0), *D_A.module.output_shape]),torch.zeros([real_A.size(0), *D_A.module.output_shape]),torch.ones([real_A.size(0), *D_A.module.output_shape])],dim=1).to(dev)

            optimizer_D_C.zero_grad()
            real_C_1 = fake_C_buffer_1.push_and_pop(cd_preds_1[-1])
            loss_real_1 = criterion_GAN(D_C(real_C_1.detach()),valid_1)
            real_C_2 = fake_C_buffer_2.push_and_pop(cd_preds_2[-1])
            loss_real_2 = criterion_GAN(D_C(real_C_2.detach()),valid_2)
            real_C_3 = fake_C_buffer_3.push_and_pop(cd_preds_3[-1])
            loss_real_3 = criterion_GAN(D_C(real_C_3.detach()),valid_3)
            loss_D_C = (loss_real_1+loss_real_2+loss_real_3) / 3
            loss_D_C.backward()
            optimizer_D_C.step()
            loss_D_C_print.append(loss_D_C.data.cpu().numpy())

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        scheduler.step()
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_D_C.step()
        loss_mean = np.mean(loss_print)
        print("train_loss:", loss_mean)

        loss_G_mean = np.mean(loss_G_print)
        print("G_loss", loss_G_mean)
        loss_cycle_mean = np.mean(loss_cycle_print)
        print("cycle_loss", loss_cycle_mean)
        loss_identity_mean = np.mean(loss_identity_print)
        print("loss_identity:",loss_identity_mean)
        loss_D_mean = np.mean(loss_D_print)
        print("D_loss", loss_D_mean)
        loss_D_C_print = np.mean(loss_D_C_print)
        print("D_C_loss", loss_D_C_print)

        # logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """
        Begin Validation
        """
        total_step = -1
        batch_iter = 0
        test_loss = 0.0
        model.eval()
        G_AB.eval()
        G_BA.eval()
        evaluator_1.reset()
        evaluator_2.reset()
        evaluator_3.reset()
        evaluator_feature_fusion.reset()
        val_loss_list = []
        # val_loss_2_list = []
        # val_loss_3_list = []
        tbar = tqdm(val_loader, desc='\r')
        with torch.no_grad():
            for batch_img1, batch_img2, labels in tbar:
                # Set variables for training
                tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + opt.batch_size))
                batch_iter = batch_iter + opt.batch_size
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                real_A = Variable(batch_img1.type(Tensor))
                real_B = Variable(batch_img2.type(Tensor))

                # Get predictions and calculate loss
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                
                real_A_norm2 = unnormalize(real_A)
                real_A_norm2 = transform1(real_A_norm2)
                real_B_norm2 = unnormalize(real_B)
                real_B_norm2 = transform1(real_B_norm2)
                fake_A_norm2 = unnormalize(fake_A)
                fake_A_norm2 = transform1(fake_A_norm2)
                fake_B_norm2 = unnormalize(fake_B)
                fake_B_norm2 = transform1(fake_B_norm2)
                [cd_preds_1, cd_preds_2, cd_preds_3,  cd_preds] = model(real_A_norm2, real_B_norm2,
                                                                                        fake_B_norm2, fake_A_norm2)

                cd_loss = criterion(cd_preds_1, labels) +criterion(cd_preds_2, labels) + criterion(cd_preds_3, labels) + criterion(cd_preds, labels)
                val_loss_list.append(cd_loss.data.cpu().numpy())

                cd_preds_1 = cd_preds_1[-1]
                _, cd_preds_1 = torch.max(cd_preds_1, 1)
                cd_preds_2 = cd_preds_2[-1]
                _, cd_preds_2 = torch.max(cd_preds_2, 1)
                cd_preds_3 = cd_preds_3[-1]
                _, cd_preds_3 = torch.max(cd_preds_3, 1)
                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                evaluator_1.add_batch(labels, cd_preds_1)
                evaluator_2.add_batch(labels, cd_preds_2)
                evaluator_3.add_batch(labels, cd_preds_3)
                evaluator_feature_fusion.add_batch(labels, cd_preds)

            mIoU_1 = evaluator_1.Mean_Intersection_over_Union()
            mIoU_2 = evaluator_2.Mean_Intersection_over_Union()
            mIoU_3 = evaluator_3.Mean_Intersection_over_Union()
            mIoU_4 = evaluator_feature_fusion.Mean_Intersection_over_Union()
            Precision_1= evaluator_1.Precision()
            Precision_2 = evaluator_2.Precision()
            Precision_3 = evaluator_3.Precision()
            Precision_4 = evaluator_feature_fusion.Precision()
            Recall_1 = evaluator_1.Recall()
            Recall_2 = evaluator_2.Recall()
            Recall_3 = evaluator_3.Recall()
            Recall_4 = evaluator_feature_fusion.Recall()
            F1_1 = evaluator_1.F1()
            F1_2 = evaluator_2.F1()
            F1_3 = evaluator_3.F1()
            F1_4 = evaluator_feature_fusion.F1()
            val_loss = np.mean(val_loss_list)

            mean_val_metrics={}
            mean_val_metrics['val_loss'] = val_loss
            # mean_val_metrics['val_loss_2'] = val_loss_2
            # mean_val_metrics['val_loss_3'] = val_loss_3
            mean_val_metrics['cd_precisions_1'] = Precision_1.data.cpu()
            mean_val_metrics['cd_precisions_2'] = Precision_2.data.cpu()
            mean_val_metrics['cd_precisions_3'] = Precision_3.data.cpu()
            mean_val_metrics['cd_precisions_fusion'] = Precision_4.data.cpu()
            mean_val_metrics['cd_recalls_1'] = Recall_1.data.cpu()
            mean_val_metrics['cd_recalls_2'] = Recall_2.data.cpu()
            mean_val_metrics['cd_recalls_3'] = Recall_3.data.cpu()
            mean_val_metrics['cd_recalls_fusion'] = Recall_4.data.cpu()
            mean_val_metrics['cd_f1scores_1'] = F1_1.data.cpu()
            mean_val_metrics['cd_f1scores_2'] = F1_2.data.cpu()
            mean_val_metrics['cd_f1scores_3'] = F1_3.data.cpu()
            mean_val_metrics['cd_f1scores_fusion'] = F1_4.data.cpu()
            mean_val_metrics['cd_miou_1'] = mIoU_1
            mean_val_metrics['cd_miou_2'] = mIoU_2
            mean_val_metrics['cd_miou_3'] = mIoU_3
            mean_val_metrics['cd_miou_fusion'] = mIoU_4

            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

            """
            Store the weights of good epochs based on validation results
            """
            if ((mean_val_metrics['cd_precisions_fusion'] > best_metrics['cd_precisions_fusion'])
                    or
                    (mean_val_metrics['cd_recalls_fusion'] > best_metrics['cd_recalls_fusion'])
                    or
                    (mean_val_metrics['cd_f1scores_fusion'] > best_metrics['cd_f1scores_fusion'])):

                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp', exist_ok=True)
                with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(str(metadata), fout)

                if opt.local_rank==0:
                    torch.save(model.state_dict(), './tmp/checkpoint_cd_epoch_'+str(epoch)+'.pt')
                    torch.save(G_AB.state_dict(), './tmp/checkpoint_gab_epoch_'+str(epoch)+'.pt')
                    torch.save(G_BA.state_dict(), './tmp/checkpoint_gba_epoch_'+str(epoch)+'.pt')
                    torch.save(D_A.state_dict(), './tmp/checkpoint_da_epoch_'+str(epoch)+'.pt')
                    torch.save(D_B.state_dict(), './tmp/checkpoint_db_epoch_'+str(epoch)+'.pt')
                    torch.save(D_C.state_dict(), './tmp/checkpoint_dc_epoch_'+str(epoch)+'.pt')

                # comet.log_asset(upload_metadata_file_path)
                if mean_val_metrics['cd_f1scores_fusion'] > best_metrics['cd_f1scores_fusion']:
                    if opt.local_rank==0:
                        torch.save(model.state_dict(), './tmp/checkpoint_cd_epoch_'+'best'+'.pt')
                        torch.save(G_AB.state_dict(), './tmp/checkpoint_gab_epoch_'+'best'+'.pt')
                        torch.save(G_BA.state_dict(), './tmp/checkpoint_gba_epoch_'+'best'+'.pt')
                        torch.save(D_A.state_dict(), './tmp/checkpoint_da_epoch_'+'best'+'.pt')
                        torch.save(D_B.state_dict(), './tmp/checkpoint_db_epoch_'+'best'+'.pt')
                        torch.save(D_C.state_dict(), './tmp/checkpoint_dc_epoch_' + 'best'+'.pt')
                    with open('./tmp/metadata_epoch_' + 'best' + '.json', 'w') as fout:
                        json.dump(str(metadata), fout)                
                    best_metrics = mean_val_metrics

            print('An epoch finished.')
    writer.close()  # close tensor board
    print('Done!')