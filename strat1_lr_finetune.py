import os
import torch
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import transforms
from dataset import BirdDataset
from utils.models import load_backbone_model
from utils.logs import Logger, load_checkpoint
from torch.utils.data import DataLoader
from torch.nn import DataParallel

def main(args):
    # Set up experiment log
    logger = Logger(args)

    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up transformation and augmentation
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])

    # Load the training dataset and dataloader
    train_dataset = BirdDataset('train', include_upscale=False, transforms=transform)
    dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)

    # Load the backbone model & classifier
    backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights)
    # Freeze all backbone for fine-tuning
    if not args.unfreeze:
        for param in backbone_model.parameters():
            param.requires_grad = False
    if args.backbone.startswith('resnet'):
        backbone_model.fc = nn.Linear(backbone_model.fc.in_features, args.num_classes)
    elif args.backbone.startswith('densenet'):
        backbone_model.classifier = nn.Linear(backbone_model.classifier.in_features, args.num_classes)
        # Parallelize the model
    if torch.cuda.device_count() > 1:
        backbone_model = DataParallel(backbone_model)
    backbone_model = backbone_model.to(device)
    
    # Load the trained backbone checkpoint
    _, model_state_dict, _ = load_checkpoint(args.backbone_ckpt_path)
    backbone_model.load_state_dict(model_state_dict)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(backbone_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Train on HR images
    for e in range(args.num_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (img, _, label) in pbar:
            # Move data to device
            img = img.to(device)
            label = label.to(device)

            # Forward pass
            output = backbone_model(img)
            loss = criterion(output, label)
            accuracy = (output.argmax(1) == label).float().mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % args.log_interval == 0:
                log_str = f'===== Epoch {e}, Iteration {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f} ====='
                print(log_str)
                logger.logfile.write(log_str + '\n')
                logger.writer.flush()

            logger.writer.add_scalar('Loss', loss.item(), e*len(dataloader) + i)
            logger.writer.add_scalar('Accuracy', accuracy.item(), e*len(dataloader) + i)
            pbar.set_description(f'E{e} | Loss {loss.item():.2f} ')
        
        # Save model checkpoint
        if (e+1) % args.save_epoch_interval == 0:
            logger.save_checkpoint(backbone_model, optimizer, e, save_name=f'epoch{e}')
        logger.save_checkpoint(backbone_model, optimizer, e, save_name='latest')
        

    # End of training
    logger.writer.close()
    logger.logfile.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='strat1_hr')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--bsz', type=int, default=1024)
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone_ckpt_path', type=str, required=True)
    parser.add_argument('--unfreeze', action='store_true')

    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_epoch_interval', type=int, default=10)
    args = parser.parse_args()
    main(args)