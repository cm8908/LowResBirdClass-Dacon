import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
# from torchvision import transforms
from dataset import BirdDataset, BirdDatasetFromDF
from utils.models import load_backbone_model, load_classifier
from utils.logs import Logger, load_checkpoint
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def train_hr(args, logger, backbone_model, optimizer, criterion, device, train_fold_df, val_fold_df, fold_idx):

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
    val_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    train_dataset = BirdDatasetFromDF(train_fold_df, 'train', include_upscale=True, transforms=train_transform)
    val_dataset = BirdDatasetFromDF(val_fold_df, 'val', include_upscale=True, transforms=val_transform)

    dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False)

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
            logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'epoch{e}_fold{fold_idx}')
        logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'latest_fold{fold_idx}')
        
        # Validation
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
                    
                outputs_hr.append(output_hr.cpu().numpy())
                outputs_lr.append(output_lr.cpu().numpy())
                labels.append(label.cpu().numpy())
            
        outputs_hr = np.concatenate(outputs_hr, axis=0)
        outputs_lr = np.concatenate(outputs_lr, axis=0)
        labels = np.concatenate(labels, axis=0)
        val_f1score_hr = f1_score(labels, outputs_hr.argmax(1), average='macro')
        val_f1score_lr = f1_score(labels, outputs_lr.argmax(1), average='macro')
        # Record validation metrics

        logger.writer.add_scalar('Val F1-score (HR)', val_f1score_hr, e*len(val_loader) + i)
        logger.writer.add_scalar('Val F1-score (LR)', val_f1score_lr, e*len(val_loader) + i)
        logger.writer.flush()
        log_str = f'===== Validation E{e},  F1 score (HR): {val_f1score_hr:.4f}, F1 score (LR): {val_f1score_lr:.4f} ====='
        logger.logfile.write(log_str + '\n')
        print(log_str)

        if args.early_stop:
            logger.check_early_stop(val_f1score_lr, backbone_model, optimizer, e)
            if logger.stop:
                logger.save_best_checkpoint(f'best_fold{fold_idx}')
                break
    return val_f1score_hr, val_f1score_lr

def train_lr(args, logger, backbone_model, optimizer, criterion, device, train_fold_df, val_fold_df, fold_idx):

    train_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.VerticalFlip(),
        A.Rotate(10),
        # A.GaussianBlur(),
        # A.GaussNoise(),
        # A.MultiplicativeNoise(),
        A.Normalize(),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    train_dataset = BirdDatasetFromDF(train_fold_df, 'train', include_upscale=False, transforms=train_transform)
    val_dataset = BirdDatasetFromDF(val_fold_df, 'val', include_upscale=False, transforms=val_transform)

    dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False)

    for e in range(args.num_epochs):
        backbone_model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (img, _, label) in pbar:
            # Move data to device
            img = img.to(device)
            label = label.to(device)

            # Forward pass
            output = backbone_model(img)
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
                logger.logfile.write(log_str + '\n')

            logger.writer.add_scalar('Loss', loss.item(), e*len(dataloader) + i)
            logger.writer.add_scalar('F1 score', f1score.item(), e*len(dataloader) + i)
            logger.writer.flush()
            pbar.set_description(f'E{e} | Loss {loss.item():.2f} ')
        
        # End of epoch
        # Save model checkpoint
        if (e+1) % args.save_epoch_interval == 0:
            logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'epoch{e}_ft_fold{fold_idx}')
        logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'latest_ft_fold{fold_idx}')
        
        # Validation
        outputs_lr = []
        labels = []
        backbone_model.eval()
        with torch.no_grad():
            for i, (img, _, label) in enumerate(val_loader):
                # Move data to device
                img = img.to(device)
                label = label.to(device)

                # Forward pass
                output_lr = backbone_model(img)
                
                if not isinstance(output_lr, torch.Tensor):
                    output_lr = output_lr.logits
                
                outputs_lr.append(output_lr.cpu().numpy())
                labels.append(label.cpu().numpy())
            
        outputs_lr = np.concatenate(outputs_lr, axis=0)
        labels = np.concatenate(labels, axis=0)
        val_f1score_lr = f1_score(labels, outputs_lr.argmax(1), average='macro')
        # Record validation metrics
        logger.writer.add_scalar('Val F1-score (LR)', val_f1score_lr, e*len(val_loader) + i)
        logger.writer.flush()
        log_str = f'===== Validation E{e}, F1 score: {val_f1score_lr:.4f} ====='
        logger.logfile.write(log_str + '\n')
        print(log_str)

        if args.early_stop:
            logger.check_early_stop(val_f1score_lr, backbone_model, optimizer, e)
            if logger.stop:
                logger.save_best_checkpoint(f'best_ft_fold{fold_idx}')
                break
    return val_f1score_lr

def main(args):
    # Set up experiment log
    logger = Logger(args)

    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the training dataset and dataloader
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    train_df = pd.read_csv(os.path.join('data', 'train.csv'))
    # Record average validation F1-scores
    fold_f1_hr_avg = 0
    fold_f1_hr_lr_avg = 0
    fold_f1_lr_avg = 0
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold_df = train_df.loc[train_idx,:]
        val_fold_df = train_df.loc[val_idx,:]
        # Phase 1: HR Training
        log_str = f'<Fold {fold_idx} HR training starts>'
        logger.logfile.write(log_str)
        print(log_str)
        
        # Load the backbone model & classifier
        backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights, image_size=args.img_size)
        if not args.unfreeze_hr:
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
        
        # Begin training loop
        val_f1_hr, val_f1_lr = train_hr(args, logger, backbone_model, optimizer, criterion, device, train_fold_df, val_fold_df, fold_idx)
        logger.reset_early_stop()
        log_str = f'<Fold {fold_idx} HR training ends>'
        logger.logfile.write(log_str)
        print(log_str)
        fold_f1_hr_avg += val_f1_hr
        fold_f1_hr_lr_avg += val_f1_lr
        

        # Phase 2: LR Training
        log_str = f'<Fold {fold_idx} LR training starts>'
        logger.logfile.write(log_str)
        print(log_str)
        
        # Load the backbone model & classifier
        backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights)
        # Freeze all backbone for fine-tuning
        if not args.unfreeze_lr:
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
        
        # Load the trained backbone checkpoint
        _, model_state_dict, _ = load_checkpoint(os.path.join(logger.save_dir, f'latest_fold{fold_idx}.pth'))
        backbone_model.load_state_dict(model_state_dict)
        
        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(backbone_model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        train_lr(args, logger, backbone_model, optimizer, criterion, device, train_fold_df, val_fold_df, fold_idx)
        logger.reset_early_stop()
        log_str = f'<Fold {fold_idx} LR training ends>'
        logger.logfile.write(log_str)
        print(log_str)
        fold_f1_lr_avg += val_f1_lr
        
    # Average validation F1-scores
    fold_f1_hr_avg /= args.num_folds
    fold_f1_hr_lr_avg /= args.num_folds
    fold_f1_lr_avg /= args.num_folds
    log_str = f'Avg. over folds: validation F1-score (phase1-HR): {fold_f1_hr_avg:.4f}\n\
        Avg. over folds: validation F1-score (phase1-LR): {fold_f1_hr_lr_avg:.4f}\n\
            Avg. over folds: validation F1-score (phase2-LR): {fold_f1_lr_avg:.4f}\n'
    logger.logfile.write(log_str)
    print(log_str)
    
    # End of training
    logger.writer.flush()
    logger.writer.close()
    logger.logfile.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='strat1_hr', 
                        help='원하는 실험 이름을 입력하세요.')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        help='Backbone architecture. utils/models.py의 load_model 함수 참조.')
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT', 
                        help='Pretrained weights to load. utils/models.py의 load_model 함수에 주석을 참고하세요.')
    parser.add_argument('--bsz', type=int, default=512, 
                        help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=25, 
                        help='Number of classes. (=25)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate.')
    parser.add_argument('--unfreeze_hr', action='store_true', 
                        help='HR training 시에 backbone weights를 freeze하지 않습니다.')
    parser.add_argument('--unfreeze_lr', action='store_true', 
                        help='LR training 시에 backbone weights를 freeze하지 않습니다.')
    parser.add_argument('--val_rate', type=float, default=0.0, 
                        help='Validation set 비율.')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Image size to be resized. Backbone 모델에 따라 다른 크기를 사용하는 것을 추천합니다. (e.g. SwinV2의 경우 256)')

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=10, 
                        help='매 몇번째 iteration마다 log를 출력할지 설정하세요.')
    parser.add_argument('--save_epoch_interval', type=int, default=10, 
                        help='매 몇번째 epoch마다 모델을 저장할지 설정하세요.')
    parser.add_argument('--early_stop', action='store_true', 
                        help='Early stopping을 사용합니다.')
    parser.add_argument('--early_stop_patience', type=int, default=5, 
                        help='Early stopping patience.')
    parser.add_argument('--cls_type', type=str, default='linear', 
                        help='backbone 뒤에 붙는 classifier layer의 아키텍쳐를 설정합니다. utils/models.py의 load_classifier 함수 참조.')
    parser.add_argument('--num_folds', type=int, default=5, 
                        help='K-Fold Cross Validation의 K값을 설정하세요.')
    args = parser.parse_args()
    main(args)