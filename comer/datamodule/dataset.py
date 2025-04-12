import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from PIL import Image
# from .transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        # if is_train and scale_aug:
            # trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            # ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
            # tr.Resize((1, 1)),
        ]
        self.transform = tr.Compose(
            trans_list
        )

    def __getitem__(self, idx):
        fname, caption = self.ds[idx]
        img = Image.open(fname)#.convert('RGB')
        img = self.transform(img)

        return fname, img, caption

    def __len__(self):
        return len(self.ds)
