import os
from PIL import Image
from torch.utils.data import Dataset

class BacteriaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        label_map = {'negative': 0, 'positive': 1}
        for label_name, label_id in label_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                self.samples.append((os.path.join(folder, img_name), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label