import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

TOTAL_LABELS = ['Ruddy Shelduck', 'Gray Wagtail', 'Indian Peacock',
    'Common Kingfisher', 'Common Rosefinch', 'Jungle Babbler',
    'Common Tailorbird', 'White-Breasted Waterhen', 'Sarus Crane',
    'Common Myna', 'Forest Wagtail', 'Indian Roller',
    'Northern Lapwing', 'Indian Grey Hornbill', 'Hoopoe',
    'Indian Pitta', 'Red-Wattled Lapwing', 'Cattle Egret',
    'White-Breasted Kingfisher', 'Rufous Treepie', 'White Wagtail',
    'House Crow', 'Coppersmith Barbet', 'Brown-Headed Barbet',
    'Asian Green Bee-Eater']
def label_to_index(label):
    return TOTAL_LABELS.index(label)
def index_to_label(index):
    return TOTAL_LABELS[index]


class BirdDataset(Dataset):
    def __init__(self, phase, data_root='./data', include_upscale=False, transforms=None, val_cut=0):
        assert phase in ['train', 'test', 'val']
        self.is_train = phase in ['train', 'val']

        self.data_root = data_root
        self.transforms = transforms
        df = pd.read_csv(os.path.join(self.data_root, f'{"train" if self.is_train else "test"}.csv'))
        if phase == 'train':
            df = df.iloc[val_cut:]
        elif phase == 'val':
            df = df.iloc[:val_cut]

        self.img_paths = df['img_path'].values
        if self.is_train:
            self.labels = df['label'].values
            self.include_upscale = include_upscale
            if self.include_upscale:
                self.upscale_img_paths = df['upscale_img_path'].values
        else:
            self.ids = df['id'].values
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Return a tuple of (img, upscale_img, label)
        `upscale_img` is None if `self.include_upscale` is False
        """
        img_path = os.path.join(self.data_root, self.img_paths[idx])
        img = Image.open(img_path)
        img = self.transforms(img) if self.transforms else img

        if self.is_train:
            upscale_img = 0
            if self.include_upscale:
                upscale_img_path = os.path.join(self.data_root, self.upscale_img_paths[idx])
                upscale_img = Image.open(upscale_img_path)
                upscale_img = self.transforms(upscale_img) if self.transforms else upscale_img
        
            label = label_to_index(self.labels[idx])
            return img, upscale_img, label
        else:
            return img, self.ids[idx]


if __name__ == '__main__':
    train_dataset = BirdDataset('train')
    print(len(train_dataset))
    print(train_dataset[0])