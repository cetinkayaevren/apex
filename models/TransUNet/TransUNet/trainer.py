import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from imutils import paths
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics import Dice

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from visual_prompt_gen.dataloader.dataloader import PromptingData
from visual_prompt_disk_optic.dataloader.dataloader import PromptingData as PromptingOpticData
from loss.loss_optic.multi_ce_dice_loss import DiceCELoss

generator1 = torch.Generator().manual_seed(42) # For train-val split#


def trainer_synapse(args, model, snapshot_path):
    from datasets_synapse.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_bkai(args, model, snapshot_path, DEVICE):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    root_file_path = "/data/hjlee/GenTransImgSeg_Thesis"

    TRAIN_VAL_IMAGE_PATH = root_file_path + "/dataset_exp/BKAI/train/image"
    TRAIN_VAL_MASK_PATH = root_file_path + "/dataset_exp/BKAI/train/mask"
    PLOT_PATH = root_file_path + f"/outputs/TransUNet_loss_diagram.png"

    # Take Data
    train_image_path = sorted(list(paths.list_images(TRAIN_VAL_IMAGE_PATH)))
    train_mask_path = sorted(list(paths.list_images(TRAIN_VAL_MASK_PATH)))

    train_val_dataset_masks = PromptingData(img_dirs = train_image_path, 
                                            gt_dirs = train_mask_path, 
                                            input_size=352, 
                                            augment = False, 
                                            device=DEVICE)
    
    #BKAI(img_dirs = train_image_path, gt_dirs = train_mask_path, transforms = transforms_data)

    train_dataset_masks, val_dataset_masks = random_split(train_val_dataset_masks, [700, 100], generator=generator1)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset_masks, shuffle=True,
                            batch_size=batch_size, 
                            num_workers=8,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    valLoader = DataLoader(val_dataset_masks, shuffle = False,
                        batch_size=batch_size,
                        num_workers=8,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)

    trainSteps = len(train_dataset_masks) // batch_size
    valSteps = len(val_dataset_masks) // batch_size

    #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    H = {"train_loss": [], "val_loss": []}


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    #model.train()
    ce_loss = BCEWithLogitsLoss() #CrossEntropyLoss() is changed because we have one-class segmentation
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        for i_batch, (images, masks, _) in enumerate(trainloader):
            image_batch, label_batch = images, masks
            image_batch, label_batch = image_batch.to(DEVICE), label_batch.to(DEVICE)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            _loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', _loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, _loss.item(), loss_ce.item()))
            totalTrainLoss += _loss

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num, dataformats='NCHW')

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        	# switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y, _) in valLoader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)

                _loss_ce = ce_loss(pred, y)
                _loss_dice = dice_loss(pred, y, softmax=True)
                _loss_val = 0.5 * _loss_ce + 0.5 * _loss_dice
                totalValLoss += _loss_val
        	
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        print("Train Loss: ", avgTrainLoss.cpu().detach().numpy())
        print("Val Loss: ", avgValLoss.cpu().detach().numpy())

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH)

    writer.close()
    return "Training Finished!"




def trainer_optic(args, model, snapshot_path, DEVICE):
    print("Training Optic Segmentation Model")
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    root_file_path = "/data/hjlee/GenTransImgSeg_Thesis"

    TRAIN_VAL_IMAGE_PATH = root_file_path + "/dataset_exp/Fundus/ORIGA_3/train/image"     #ORIGA RIM_ONE_r3
    TRAIN_VAL_MASK_PATH = root_file_path + "/dataset_exp/Fundus/ORIGA_3/train/mask"       #ORIGA RIM_ONE_r3
    PLOT_PATH = root_file_path + f"/outputs/TransUNet_optic_loss_diagram.png"

    # Take Data
    train_image_path = sorted(list(paths.list_images(TRAIN_VAL_IMAGE_PATH)))
    train_mask_path = sorted(list(paths.list_images(TRAIN_VAL_MASK_PATH)))

    train_val_dataset_masks = PromptingOpticData(img_dirs = train_image_path, 
                                            gt_dirs = train_mask_path, 
                                            input_size=352, 
                                            augment = False,
                                            device=DEVICE)
    
    #BKAI(img_dirs = train_image_path, gt_dirs = train_mask_path, transforms = transforms_data)

    train_dataset_masks, val_dataset_masks = random_split(train_val_dataset_masks, [0.8, 0.2], generator=generator1) #[450, 50]


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset_masks, shuffle=True,
                            batch_size=batch_size, 
                            num_workers=8,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    valLoader = DataLoader(val_dataset_masks, shuffle = False,
                        batch_size=1,
                        num_workers=8,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)

    trainSteps = len(train_dataset_masks) // batch_size
    valSteps = len(val_dataset_masks) // 1

    #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    H = {"train_loss": [], "val_loss": []}


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    #model.train()
    loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5, num_class=3)

    optimizer = optim.Adam(model.parameters(), lr=base_lr) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    dice = Dice(average="macro", num_classes=3, ignore_index=None).to(DEVICE)
    best_dice=0
    for epoch_num in iterator:
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        total_dice = 0

        for i_batch, (images, masks, _) in enumerate(trainloader):
            image_batch, label_batch = images, masks
            image_batch, label_batch = image_batch.to(DEVICE), label_batch.squeeze(1).to(DEVICE).long()
            outputs = model(image_batch)
            _loss = loss(outputs, label_batch)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', _loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, _loss.item()))
            totalTrainLoss += _loss

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0).unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num, dataformats='NCHW')

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        	# switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y, _) in valLoader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.squeeze(1).to(DEVICE).long())
                # make the predictions and calculate the validation loss
                pred = model(x)
                pred_class = pred.argmax(dim=1)
                total_dice += dice(pred_class, y.to(DEVICE).long()) / valSteps
                _loss_val = loss(pred, y)
                totalValLoss += _loss_val
        	
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        print("Train Loss: ", avgTrainLoss.cpu().detach().numpy())
        print("Val Loss: ", avgValLoss.cpu().detach().numpy())
        print("DICE Score: ", total_dice)

        if total_dice > best_dice:
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("The Best Model is written...")
            best_dice = total_dice

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH)

    writer.close()
    return "Training Finished!"