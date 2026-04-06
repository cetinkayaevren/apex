import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss

from torch.utils.data import random_split
from imutils import paths
import matplotlib.pyplot as plt
from pathlib import Path
from torchmetrics import Dice

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from visual_prompt_gen.dataloader.dataloader import PromptingData
from visual_prompt_disk_optic.dataloader.dataloader import PromptingData as PromptingOpticData
from loss.loss_optic.multi_ce_dice_loss import DiceCELoss
generator1 = torch.Generator().manual_seed(42) # For train-val split#



def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
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
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
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
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
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

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


def trainer_bkai_(args, model, snapshot_path, DEVICE):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    root_file_path = "/data/hjlee/GenTransImgSeg_Thesis"

    TRAIN_VAL_IMAGE_PATH = root_file_path + "/dataset_exp/BKAI/train/image"
    TRAIN_VAL_MASK_PATH = root_file_path + "/dataset_exp/BKAI/train/mask"
    PLOT_PATH = root_file_path + f"/outputs/SwinUNet_loss_diagram.png"

    # Take Data
    train_image_path = sorted(list(paths.list_images(TRAIN_VAL_IMAGE_PATH)))
    train_mask_path = sorted(list(paths.list_images(TRAIN_VAL_MASK_PATH)))


    train_val_dataset_masks = PromptingData(img_dirs = train_image_path, 
                                            gt_dirs = train_mask_path, 
                                            input_size=224, 
                                            augment = False, 
                                            device=DEVICE)
    
    train_dataset_masks, val_dataset_masks = random_split(train_val_dataset_masks, [700, 100], generator=generator1)

    print("The length of train set is: {}".format(len(train_dataset_masks)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset_masks, shuffle=True,
                            batch_size=batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    valLoader = DataLoader(val_dataset_masks, shuffle = False,
                        batch_size=batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)
    
    H = {"train_loss": [], "val_loss": []}

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = BCEWithLogitsLoss() #CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) #optim.Adam(model.parameters(), lr=base_lr)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10

    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, (images,labels, _) in tqdm(enumerate(trainloader), desc=f"Train: {epoch_num}", total=len(trainloader),
                                           leave=False):
            image_batch, label_batch = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch) #ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
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

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num, dataformats="NCHW")
        
        batch_ce_loss /= len(trainloader)
        batch_dice_loss /= len(trainloader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        
        H["train_loss"].append(batch_loss)

        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, (images,labels,_) in tqdm(enumerate(valLoader), desc=f"Val: {epoch_num}",
                                                   total=len(valLoader), leave=False):
                    image_batch, label_batch = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch) #ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(valLoader)
                batch_dice_loss /= len(valLoader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                
                H["val_loss"].append(batch_loss)
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))


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


############################################################################################

def trainer_optic_(args, model, snapshot_path, DEVICE):
    print("OPTIC Dataset segmentation task!!!!!")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    root_file_path = "/data/hjlee/GenTransImgSeg_Thesis"

    TRAIN_VAL_IMAGE_PATH = root_file_path + "/dataset_exp/Fundus/ORIGA_3/train/image"     #ORIGA
    TRAIN_VAL_MASK_PATH = root_file_path + "/dataset_exp/Fundus/ORIGA_3/train/mask"       #ORIGA
    PLOT_PATH = root_file_path + f"/outputs/SwinUNet_loss_diagram.png"

    # Take Data
    train_image_path = sorted(list(paths.list_images(TRAIN_VAL_IMAGE_PATH)))
    train_mask_path = sorted(list(paths.list_images(TRAIN_VAL_MASK_PATH)))


    train_val_dataset_masks = PromptingOpticData(img_dirs = train_image_path, 
                                            gt_dirs = train_mask_path, 
                                            input_size=224, 
                                            augment = False,
                                            device=DEVICE)
    
    train_dataset_masks, val_dataset_masks = random_split(train_val_dataset_masks, [0.8, 0.2], generator=generator1) #[450, 50]

    print("The length of train set is: {}".format(len(train_dataset_masks)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset_masks, shuffle=True,
                            batch_size=batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    valLoader = DataLoader(val_dataset_masks, shuffle = False,
                        batch_size=1,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)
    
    trainSteps = len(train_dataset_masks) // batch_size
    valSteps = len(val_dataset_masks) // 1

    H = {"train_loss": [], "val_loss": []}

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5, num_class=3)

    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4) #optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) #optim.Adam(model.parameters(), lr=base_lr)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    best_dice = 0
    dice = Dice(average="macro", num_classes=3, ignore_index=None).to(DEVICE)

    for epoch_num in iterator:
        totalTrainLoss = 0
        model.train()
        for i_batch, (images,labels, _) in tqdm(enumerate(trainloader), desc=f"Train: {epoch_num}", total=len(trainloader),
                                           leave=False):
            image_batch, label_batch = images.to(DEVICE), labels.squeeze(1).to(DEVICE).long()
            outputs = model(image_batch)

            _loss = loss(outputs, label_batch)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            totalTrainLoss += _loss
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', _loss, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num, dataformats="CHW")
        
        avgTrainLoss = totalTrainLoss / trainSteps
        logging.info('Train epoch: %d : loss : %f' % (
            epoch_num, avgTrainLoss))
        
        H["train_loss"].append(avgTrainLoss)

        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            totalValLoss = 0
            total_dice = 0
            with torch.no_grad():
                for i_batch, (images,labels,_) in tqdm(enumerate(valLoader), desc=f"Val: {epoch_num}",
                                                   total=len(valLoader), leave=False):
                    image_batch, label_batch = images.to(DEVICE), labels.squeeze(1).to(DEVICE).long()
                    outputs = model(image_batch)
                    output_class = outputs.argmax(dim=1)
                    total_dice += dice(output_class, label_batch.to(DEVICE).long()) / valSteps
                    _val_loss = loss(outputs, label_batch)
                    totalValLoss += _val_loss

                avgValLoss = totalValLoss / valSteps
                logging.info('Val epoch: %d, loss : %f, DICE : %f' % (
                    epoch_num, avgValLoss, total_dice))
                
                H["val_loss"].append(avgValLoss)
                if total_dice > best_dice:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_dice = total_dice
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))


    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"].cpu().numpy(), label="train_loss")
    plt.plot(H["val_loss"].cpu().numpy(), label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH)

    writer.close()
    return "Training Finished!"
