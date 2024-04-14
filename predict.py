import os
import torch
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import transforms
from dataset import BirdDataset, index_to_label
from utils.models import load_backbone_model
from utils.logs import load_checkpoint
from torch.utils.data import DataLoader
from torch.nn import DataParallel

def main(args):
    # Make submission file
    submission = open(os.path.join('submissions', f'{args.submission_name}.csv'), 'w')
    submission.write('id,label\n')

    # Set up device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the training dataset and dataloader
    test_dataset = BirdDataset('test', transforms=transforms.ToTensor())
    dataloader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False)

    # Load the backbone model & classifier
    backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights)
    if args.backbone.startswith('resnet'):
        backbone_model.fc = nn.Linear(backbone_model.fc.in_features, args.num_classes)
    elif args.backbone.startswith('densenet'):
        backbone_model.classifier = nn.Linear(backbone_model.classifier.in_features, args.num_classes)
        # Parallelize the model
    if torch.cuda.device_count() > 1:
        backbone_model = DataParallel(backbone_model)
    backbone_model = backbone_model.to(device)
    
    # Load the trained checkpoint
    _, model_state_dict, _ = load_checkpoint(args.ckpt_path)
    backbone_model.load_state_dict(model_state_dict)
    backbone_model.eval()
    
    # Train on HR images
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (img, img_id) in pbar:
            # Move data to device
            img = img.to(device)

            # Forward pass
            output = backbone_model(img)
            for b in range(len(output)):
                prediction = index_to_label(output[b].argmax(dim=0))
            
                print(img_id[b], prediction)
                submission.write(f'{img_id[b]},{prediction}\n')

    submission.close()
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=25)
    
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--submission_name', type=str, required=True)

    args = parser.parse_args()
    main(args)