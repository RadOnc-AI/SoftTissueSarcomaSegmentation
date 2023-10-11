import os
import numpy as np
import torchio as tio
import SimpleITK as sitk

from glob import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def crop(volume: sitk.Image, segmentation: sitk.Image, mode: str, margin_size: int = 10) -> (np.ndarray, np.ndarray):

    assert mode in ["exact", "fixed_margin", "random_margin"]

    vol_arr = sitk.GetArrayFromImage(volume)
    seg_arr = sitk.GetArrayFromImage(segmentation)

    incides = np.where(seg_arr == 1)
    z_min = np.min(incides[0])
    y_min = np.min(incides[1])
    x_min = np.min(incides[2])
    z_max = np.max(incides[0])
    y_max = np.max(incides[1])
    x_max = np.max(incides[2])

    if mode == "exact":
        img_crop = vol_arr[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        seg_crop = seg_arr[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    elif mode == "fixed_margin":
        margin = int(margin_size / 2)
        img_crop = vol_arr[(z_min-margin):(z_max+margin + 1), (y_min-margin):(y_max+margin + 1),
                           (x_min-margin):x_max+margin + 1]
        seg_crop = seg_arr[(z_min-margin):(z_max+margin + 1), (y_min-margin):(y_max+margin + 1),
                           (x_min-margin):x_max+margin + 1]

    elif mode == "random_margin":
        x_margin = int(np.random.randint(0, margin_size+1) / 2)
        y_margin = int(np.random.randint(0, margin_size+1) / 2)
        z_margin = int(np.random.randint(0, margin_size+1) / 2)
        img_crop = vol_arr[(z_min - z_margin):(z_max + z_margin + 1), (y_min - y_margin):(y_max + y_margin + 1),
                           (x_min - x_margin):x_max + x_margin + 1]
        seg_crop = seg_arr[(z_min - z_margin):(z_max + z_margin + 1), (y_min - y_margin):(y_max + y_margin + 1),
                           (x_min - x_margin):x_max + x_margin + 1]

    return (img_crop, seg_crop)


def z_standardize(volume: np.ndarray) -> np.ndarray:
    return (volume - np.mean(volume)) / (np.std(volume))


class SarcomaDataset(Dataset):
    def __init__(self, config, mode):

        assert mode in ["train", "val", "test"]

        self.config = config
        self.mode = mode


        # Train & Val Files
        self.train_val_img_files = [file for file in sorted(glob(os.path.join(config.train_dir, "*.nii")))
                                    if not "label" in file]
        self.train_val_seg_files = [file for file in sorted(glob(os.path.join(config.train_dir, "*.nii")))
                                    if "label" in file]
        self.train_img_files, self.val_img_files, self.train_seg_files, self.val_seg_files = train_test_split(
            self.train_val_img_files, self.train_val_seg_files, test_size=(1-config.train_val_ratio),
            random_state=42)

        assert len(self.train_img_files) == len(self.train_seg_files)
        assert len(self.val_img_files) == len(self.val_seg_files)

        # Test Files
        self.test_img_files = [file for file in sorted(glob(os.path.join(config.test_dir, "*.nii")))
                               if not "label" in file]
        self.test_seg_files = [file for file in sorted(glob(os.path.join(config.test_dir, "*.nii")))
                               if "label" in file]

        assert len(self.test_img_files) == len(self.test_seg_files)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_img_files)
        elif self.mode == "val":
            return len(self.val_img_files)
        else:
            return len(self.test_img_files)

    def __getitem__(self, idx):

        if self.mode == "train":
            self.img_files = self.train_img_files
            self.seg_files = self.train_seg_files

        elif self.mode == "val":
            self.img_files = self.val_img_files
            self.seg_files = self.val_seg_files

        elif self.mode == "test":
            self.img_files = self.test_img_files
            self.seg_files = self.test_seg_files

        img = tio.ScalarImage(self.img_files[idx])
        seg = tio.transforms.Resample(target=img)(tio.LabelMap(self.seg_files[idx]))

        subject = tio.Subject(
            image=img,
            segmentation=seg,
        )

        resample_img = tio.transforms.Resample(target=1, image_interpolation="linear")
        resample_seg = tio.transforms.Resample(target=1, label_interpolation="nearest")
        resampled_img = resample_img(subject.image)
        resampled_seg = resample_seg(subject.segmentation)

        subject = tio.Subject(
            image=resampled_img,
            segmentation=resampled_seg,
        )

        if self.config.data_augmentation:
            random_flip = tio.transforms.RandomFlip()
            random_rotate = tio.transforms.RandomAffine(scales=0, degrees=10, translation=0)

            transforms_dict = {
                tio.RandomBlur(): 0.5,
                tio.RandomNoise(): 0.5,
            }
            intensity_transform = tio.OneOf(transforms_dict)

            transforms = [random_flip, random_rotate, intensity_transform]
            transform = tio.Compose(transforms)

            subject = transform(subject)

        img_crop_arr, seg_crop_arr = crop(volume=subject.image.as_sitk(), segmentation=subject.segmentation.as_sitk(),
                                          mode=self.config.crop, margin_size=self.config.crop_margin)

        img_arr = z_standardize(img_crop_arr)
        seg_arr = seg_crop_arr

        img_arr = np.expand_dims(img_arr, axis=0)
        seg_arr = np.expand_dims(seg_arr, axis=0)

        patient_id = self.img_files[idx].split("/")[-1]

        return img_arr, seg_arr, patient_id
