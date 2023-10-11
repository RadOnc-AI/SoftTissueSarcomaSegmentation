import os
import numpy as np
import torchio as tio
import SimpleITK as sitk

from glob import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# import random
# import nibabel as nib
# from data.Custom_Transforms import BBoxCrop3D, get_patches, get_volume
# from scipy import ndimage as nd
# import SimpleITK
# from dipy.align.reslice import reslice


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


# def resample(img, nshape=None, spacing=None, new_spacing=None, order=0, mode='constant'):
#     """
#         Change image resolution by resampling
#
#         Inputs:
#         - spacing (numpy.ndarray): current resolution
#         - new_spacing (numpy.ndarray): new resolution
#         - order (int: 0-5): interpolation order
#
#         Outputs:
#         - resampled image
#         """
#     if nshape is None:
#         if spacing.shape[0] != 1:
#             spacing = np.transpose(spacing)
#
#         if new_spacing.shape[0] != 1:
#             new_spacing = np.transpose(new_spacing)
#
#         if np.array_equal(spacing, new_spacing):
#             return img
#
#         resize_factor = spacing / new_spacing
#         new_real_shape = img.shape * resize_factor
#         new_shape = np.round(new_real_shape)
#         real_resize_factor = new_shape / img.shape
#
#     else:
#         if img.shape == nshape:
#             return img
#         real_resize_factor = np.array(nshape, dtype=float) / np.array(img.shape, dtype=float)
#
#     image = nd.interpolation.zoom(img, real_resize_factor.ravel(), order=order, mode=mode)
#
#     return image


def z_standardize(volume: np.ndarray) -> np.ndarray:
    return (volume - np.mean(volume)) / (np.std(volume))


# def split_in_patches(volume, max_patch_size):
#     """Splits volume in patches if HxWxD > max_patch_size**3
#     """
#
#     shape = np.array(volume.shape[-3:])
#     divs = np.ceil(shape / np.array(max_patch_size)).astype(np.int8)
#     while np.prod(shape / np.array(divs)) > max_patch_size ** 3:
#         max_dim = np.argmax(shape / np.array(divs))
#         divs[max_dim] += 1  # divide in max dimension
#     print(f"New patch shape {shape / np.array(divs)}")
#     dim1_pad, dim2_pad, dim3_pad = np.mod(volume.shape, divs)
#     volume = np.pad(volume, ((0, divs[0] - dim1_pad), (0, divs[1] - dim2_pad), (0, divs[2] - dim3_pad)))
#     volume = get_patches(volume, divs, offset=(5, 5, 5))
#     return volume, divs  # P x H x W x D


# def get_bbox(segmentation):
#
#     seg_arr = segmentation
#
#     # seg_arr = segmentation.get_fdata()
#
#     incides = np.where(seg_arr == 1)
#     x_min = np.min(incides[0])
#     y_min = np.min(incides[1])
#     z_min = np.min(incides[2])
#     x_max = np.max(incides[0])
#     y_max = np.max(incides[1])
#     z_max = np.max(incides[2])
#
#     return [x_min, y_min, z_min, x_max, y_max, z_max]


# def get_patch_size(sitk_segmentation: sitk.Image, random_margin=False, fixed_margin=False):
#
#     seg_arr = sitk.GetArrayFromImage(sitk_segmentation)
#     z_size, y_size, x_size = seg_arr.shape
#
#     incides = np.where(seg_arr == 1)
#     z_min = np.min(incides[0])
#     y_min = np.min(incides[1])
#     x_min = np.min(incides[2])
#     z_max = np.max(incides[0])
#     y_max = np.max(incides[1])
#     x_max = np.max(incides[2])
#
#     width = x_max - x_min
#     height = y_max - y_min
#     depth = z_max - z_min
#
#     width_10 = int(width * 0.1)
#     height_10 = int(width * 0.1)
#     depth_10 = int(depth * 0.1)
#
#     if random_margin:
#         width = np.minimum(width + random.randint(0, width_10 + 1), x_size)
#         height = np.minimum(height + random.randint(0, height_10 + 1), y_size)
#         depth = np.minimum(depth + random.randint(0, depth_10 + 1), z_size)
#
#     elif fixed_margin:
#         width = np.minimum(width + 10, x_size)
#         height = np.minimum(height + 10, y_size)
#         depth = np.minimum(depth + 10, z_size)
#
#     return (width, height, depth)


# def preprocess(volume, bbox, padding, max_patch_size):
#     """Preproccessing of the volume :
#     1. Crops around the bounding box with padding indicating the additional paddings in each direction from the bounding box
#     2. Resamples the volume to a isotropic voxel size of 1mm^3
#     3. Z-score standarization
#     4. Creating patches if HxWxD > max_patch_size**3
#
#     Args:
#         volume NifTi: MRI Scan
#         bbox list: x_min,y_min,z_min,x_max,y_max,z_max
#         padding list: additional padding in all 6 directions (If data from the volume is available the data is taken, if not 0 padding is used)
#         max_patch_size int: Creating patches if HxWxD > max_patch_size**3
#
#     Returns:
#         dict: containing preprocessed volume and additional information
#     """
#     croppingObj = BBoxCrop3D(padding, False)
#
#     volume_data = volume.get_fdata()
#     print('Volume shape {}'.format(volume_data.shape))
#     volume_data = croppingObj.crop(bbox, volume_data)
#     # volume_data = volume_data[bbox[0]:bbox[1]+1,bbox[2]:bbox[3]+1,bbox[4]:bbox[5]+1]
#     print('Volume shape after cropping {}'.format(volume_data.shape))
#     shape_before_resample = volume_data.shape
#     affine = volume.affine
#     zooms = volume.header.get_zooms()[:3]
#     # volume, resampled_affine = reslice(volume_data, affine, zooms, (1, 1, 1))
#     volume = resample(volume_data, spacing=np.array(zooms), new_spacing=np.array([1, 1, 1]))
#     print('Volume shape after resampling {}'.format(volume.shape))
#     volume = z_standardize(volume)
#
#     divs = [1, 1, 1]
#     if volume.size > max_patch_size ** 3:
#         volume, divs = split_in_patches(volume, max_patch_size)  # P x H x W x D
#
#     return {'volume': volume} #, 'original_zooms': zooms, 'resampled_affine': resampled_affine, 'divs': divs,
#             # 'shape_before_resample': shape_before_resample}


class SarcomaDataset(Dataset):
    def __init__(self, config, mode):

        assert mode in ["train", "val", "test"]

        # self.args = args
        self.config = config
        self.mode = mode
        # ToDo: add split ratio and random_state to args

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

        # if not own:
        #     volume = nib.load(self.img_files[idx])
        #     mask = nib.load(self.seg_files[idx])
        #
        #     bbox = get_bbox(mask)
        #
        #     preprocessed = preprocess(volume=volume, bbox=bbox, padding=[10, 10, 10, 10, 10, 10], max_patch_size=200)
        #     volume = preprocessed['volume']
        #     # resampled_affine = preprocessed['resampled_affine']
        #     # original_zooms = preprocessed['original_zooms']
        #     # divs = preprocessed['divs']
        #     # shape_before_resample = preprocessed["shape_before_resample"]
        #
        #     if len(volume.shape) == 3:
        #         # volume = np.expand_dims(np.expand_dims(volume, axis=0), axis=0) #create channel and batch axis
        #         volume = np.expand_dims(volume, axis=0) #create channel and batch axis
        #     elif len(volume.shape) == 4: #patches
        #         volume = np.expand_dims(volume, axis=1) #create channel axis

        # Own preprocessing
        # elif own:

        # volume = nib.load(self.img_files[idx])
        # mask = nib.load(self.seg_files[idx])
        #
        # r_volume, r_affine_volume = reslice(data=volume.get_fdata(),
        #                                     affine=volume.affine,
        #                                     zooms=volume.header.get_zooms()[:3],
        #                                     new_zooms=(1, 1, 1),
        #                                     order=1,
        #                                     mode='constant')
        #
        # r_mask, r_affine_mask = reslice(data=mask.get_fdata(),
        #                                 affine=mask.affine,
        #                                 zooms=mask.header.get_zooms()[:3],
        #                                 new_zooms=(1, 1, 1),
        #                                 order=0,
        #                                 mode='nearest')
        #
        # bbox = get_bbox(r_mask)
        #
        # c_volume, c_mask = crop(bbox, r_volume, r_mask)
        #
        # c_volume = z_standardize(c_volume)
        #
        # img_arr = np.expand_dims(c_volume, axis=0)
        # seg_arr = np.expand_dims(c_mask, axis=0)
        #
        # patient_id = self.img_files[idx].split("/")[-1]
        #
        # return img_arr, seg_arr, patient_id

        # img = tio.transforms.ToCanonical()(tio.ScalarImage(self.img_files[idx]))

        ##### print(self.img_files[idx])

        img = tio.ScalarImage(self.img_files[idx])
        seg = tio.transforms.Resample(target=img)(tio.LabelMap(self.seg_files[idx]))

        subject = tio.Subject(
            image=img,
            # image=img,
            # segmentation=tio.LabelMap(self.seg_files[idx]),
            segmentation=seg,
        )

        # subject.check_consistent_attribute("orientation", relative_tolerance=0.5, absolute_tolerance=0.5)

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

        # if self.args.crop == "random_margin":
        #     patch_size = get_patch_size(subject.segmentation.as_sitk(), random_margin=True)
        #     sampler = tio.data.LabelSampler(patch_size=patch_size, label_name="segmentation")
        #     cropped = next(sampler(subject=subject, num_patches=1))
        #
        # if self.args.crop == "fixed_margin":
        #     patch_size = get_patch_size(subject.segmentation.as_sitk(), fixed_margin=True)
        #     sampler = tio.data.LabelSampler(patch_size=patch_size, label_name="segmentation")
        #     cropped = next(sampler(subject=subject, num_patches=1))
        #
        # if self.args.crop == "exact":
        #     crop = tio.transforms.CropOrPad(mask_name="segmentation")
        #     cropped = crop(subject)

        img_arr = z_standardize(img_crop_arr)
        seg_arr = seg_crop_arr

        # z_scale = tio.transforms.ZNormalization()
        # scaled_img = z_scale(cropped.image)

        # img_arr = SimpleITK.GetArrayFromImage(scaled_img.as_sitk())
        # seg_arr = SimpleITK.GetArrayFromImage(cropped.segmentation.as_sitk())

        # img_arr = np.swapaxes(img_arr, 0, 2)
        # seg_arr = np.swapaxes(seg_arr, 0, 2)

        img_arr = np.expand_dims(img_arr, axis=0)
        seg_arr = np.expand_dims(seg_arr, axis=0)

        patient_id = self.img_files[idx].split("/")[-1]

        return img_arr, seg_arr, patient_id
