from glob import glob
import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
# from torchvision import transforms
from dataset import BirdDataset, index_to_label
from utils.models import load_backbone_model, load_classifier
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

    # Set up transformation and augmentation
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    # Load the training dataset and dataloader
    test_dataset = BirdDataset('test', transforms=transform, test_upscaled=args.test_upscaled)
    dataloader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False)

    # Load the backbone model & classifier
    backbone_model = load_backbone_model(args.backbone, weights=args.pretrained_weights)
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
    
    fold_ckpt_paths = sorted(glob(os.path.join(args.ckpt_dir, 'best_fold*.pth')))
    fold_preds = []
    for fold_ckpt_path in fold_ckpt_paths:
        print(f'Prediction using {fold_ckpt_path}')
        # Load the trained checkpoint
        _, model_state_dict, _ = load_checkpoint(fold_ckpt_path)
        backbone_model.load_state_dict(model_state_dict)
        backbone_model.eval()
        
        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            batch_preds = []
            batch_img_ids = []
            for i, (img, img_id) in pbar:
                # Move data to device
                img = img.to(device)

                # Forward pass
                output = backbone_model(img)
                if not isinstance(output, torch.Tensor):
                    output = output.logits
                batch_preds.append(output.argmax(dim=1).cpu().numpy())
                batch_img_ids.extend(img_id)
            batch_preds = np.concatenate(batch_preds)
        fold_preds.append(batch_preds)
    preds_ensemble = list(map(lambda x: np.bincount(x).argmax(), np.stack(fold_preds, axis=1)))

    print('Writing submission file...')
    for i in tqdm(range(len(preds_ensemble))):
        prediction = index_to_label(preds_ensemble[i])
            
        # print(img_id[i], prediction)
        submission.write(f'{batch_img_ids[i]},{prediction}\n')
    submission.close()
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--img_size', type=int, default=224)
    
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--submission_name', type=str, required=True)
    parser.add_argument('--cls_type', type=str, default='linear')
    
    parser.add_argument('--test_upscaled', action='store_true', help='If set to true, test dataset will be loaded with (super-resolution) upscaled images')

    args = parser.parse_args()
    main(args)