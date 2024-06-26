import os
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
# from torchvision import transforms
from dataset import BirdDataset
from utils.models import load_backbone_model, load_classifier
from utils.logs import Logger
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from sklearn.metrics import f1_score

def main(args):
    # Set up experiment log
    logger = Logger(args)

    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up transformation and augmentation
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(30),
    #     transforms.ToTensor()
    # ])
    train_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.VerticalFlip(),
        A.Rotate(10),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.MultiplicativeNoise(),
        A.Normalize(),
        ToTensorV2()
    ])

    # Load the training dataset and dataloader
    train_dataset = BirdDataset('train', include_upscale=True, transforms=train_transform)
    if args.val_rate > 0:
        val_transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            A.Normalize(),
            ToTensorV2()
        ])
        val_cut = int(args.val_rate * len(train_dataset))
        train_dataset = BirdDataset('train', include_upscale=True, transforms=val_transform, val_cut=val_cut)
        val_dataset = BirdDataset('val', include_upscale=True, transforms=val_transform, val_cut=val_cut)
        val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False)
    dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)

    # Load the backbone model & classifier
    backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights, image_size=args.img_size)
    if not args.unfreeze:
        for param in backbone_model.parameters():
            param.requires_grad = False
    if args.backbone.startswith('resnet'):
        backbone_model.fc = load_classifier(backbone_model.fc.in_features, args.num_classes, args.cls_type)
    elif any(model_name in args.backbone for model_name in ['densenet', 'swinv2', 'vitl']):
        backbone_model.classifier = load_classifier(backbone_model.classifier.in_features, args.num_classes, args.cls_type)
    elif args.backbone.startswith('vit'):
        backbone_model.heads.head = load_classifier(backbone_model.heads.head.in_features, args.num_classes, args.cls_type)
        # Parallelize the model
    if torch.cuda.device_count() > 1:
        backbone_model = DataParallel(backbone_model)
    backbone_model = backbone_model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(backbone_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Train on HR images
    for e in range(args.num_epochs):
        backbone_model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (_, upscale_img, label) in pbar:
            # Move data to device
            upscale_img = upscale_img.to(device)
            label = label.to(device)

            # Forward pass
            output = backbone_model(upscale_img)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            loss = criterion(output, label)
            f1score = f1_score(label.cpu().numpy(), output.argmax(1).cpu().numpy(), average='macro')
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % args.log_interval == 0:
                log_str = f'===== Epoch {e}, Iteration {i}, Loss: {loss.item():.4f}, F1 score: {f1score.item():.4f} ====='
                print(log_str)
                logger.logfile.write(log_str + '\n')

            logger.writer.add_scalar('Loss', loss.item(), e*len(dataloader) + i)
            logger.writer.add_scalar('F1 score', f1score.item(), e*len(dataloader) + i)
            logger.writer.flush()
            pbar.set_description(f'E{e} | Loss {loss.item():.2f} ')
        
        # End of epoch
        # Save model checkpoint
        if (e+1) % args.save_epoch_interval == 0:
            logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'epoch{e}')
        logger.save_checkpoint(backbone_model, optimizer, e, save_name='latest')
        
        # Validation
        if args.val_rate > 0:
            outputs_hr = []
            outputs_lr = []
            labels = []
            backbone_model.eval()
            with torch.no_grad():
                for i, (img, upscale_img, label) in enumerate(val_loader):
                    # Move data to device
                    img = img.to(device)
                    upscale_img = upscale_img.to(device)
                    label = label.to(device)

                    # Forward pass
                    output_hr = backbone_model(upscale_img)
                    output_lr = backbone_model(img)

                    if not isinstance(output_hr, torch.Tensor):
                        output_hr = output_hr.logits
                    if not isinstance(output_lr, torch.Tensor):
                        output_lr = output_lr.logits
                    
                    outputs_hr.append(output_hr)
                    outputs_lr.append(output_lr)
                    labels.append(label)
            
            outputs_hr = torch.cat(outputs_hr, dim=0)
            outputs_lr = torch.cat(outputs_lr, dim=0)
            labels = torch.cat(labels, dim=0)
            val_f1score_hr = f1_score(labels.cpu().numpy(), outputs_hr.argmax(1).cpu().numpy(), average='macro')
            val_f1score_lr = f1_score(labels.cpu().numpy(), outputs_lr.argmax(1).cpu().numpy(), average='macro')
            # Record validation metrics

            logger.writer.add_scalar('Val F1-score (HR)', val_f1score_hr.item(), e*len(val_loader) + i)
            logger.writer.add_scalar('Val F1-score (LR)', val_f1score_lr.item(), e*len(val_loader) + i)
            logger.writer.flush()
            log_str = f'===== Validation E{e},  F1 score (HR): {val_f1score_hr.item():.4f}, F1 score (LR): {val_f1score_lr.item():.4f} ====='
            logger.logfile.write(log_str + '\n')
            print(log_str)

            if args.early_stop:
                logger.check_early_stop(val_f1score_lr.item())
                if logger.stop:
                    break
        

    # End of training
    logger.writer.flush()
    logger.writer.close()
    logger.logfile.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='strat1_hr')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--bsz', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--unfreeze', action='store_true')
    parser.add_argument('--val_rate', type=float, default=0.0)
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_epoch_interval', type=int, default=10)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--cls_type', type=str, default='linear')
    args = parser.parse_args()
    main(args)