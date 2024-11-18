from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
import glob

class DataPrep(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label1_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label1_paths[idx]).convert('L')
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define a simple transform
transform = T.Compose([
    T.ToTensor(),  # Automatically converts PIL images to tensors and scales to [0, 1]
])


def data_pred(DATA_DIR, str='train', dataset='mass'):

    images = os.path.join(DATA_DIR, str)
    masks = os.path.join(DATA_DIR, str + '_labels')

    if dataset=='mass':
        image_paths = glob.glob(os.path.join(images, '*.tiff'))
        label_paths = glob.glob(os.path.join(masks, '*.tif'))

    elif dataset=='cityscale':
        image_paths = glob.glob(os.path.join(images, '*_sat.png'))
        label_paths = glob.glob(os.path.join(masks, '*_gt.png'))

    elif dataset=='deepglobe':
        image_paths = glob.glob(os.path.join(images, '*_sat.jpg'))
        label_paths = glob.glob(os.path.join(masks, '*_mask.png'))

    elif dataset=='equa':
        image_paths = glob.glob(os.path.join(images, '*.png'))
        label_paths = glob.glob(os.path.join(masks, '*.png'))

    elif dataset=='spacenet':
        image_paths = glob.glob(os.path.join(images, '*_rgb.png'))
        label_paths = glob.glob(os.path.join(masks, '*_gt.png'))

    image_paths.sort()
    label_paths.sort()
    return image_paths, label_paths

