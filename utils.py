import os
import torch
import mlflow
import random
import subprocess
import numpy as np
from monai.losses import DiceLoss
from model.UNet.unet3d import ResUNet3D
from ParamConfigurator import ParamConfigurator


class DiceCELoss:
    """TODO: Docstring"""

    def __init__(self, config: ParamConfigurator):
        self.config = config
        self.loss_CE = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.loss_Dice = DiceLoss(include_background=self.config.include_background, to_onehot_y=True, sigmoid=True,
                                  squared_pred=True)

    def __call__(self, outputs: torch.Tensor, segmentations: torch.Tensor) -> torch.Tensor:

        ce_outputs = outputs
        ce_segmentations = segmentations[:, 0, :, :, :].to(torch.long)
        loss_ce = self.loss_CE(ce_outputs, ce_segmentations)
        loss_dice = self.loss_Dice(outputs, segmentations)
        loss = loss_ce + loss_dice

        return loss


class DiceSimilarityCoefficient:
    """TODO: Docstring"""

    def __init__(self, config: ParamConfigurator):
        self.config = config

    def __call__(self, outputs: torch.Tensor, segmentations: torch.Tensor) -> float:

        numerator = torch.sum(outputs.argmax(dim=1).squeeze() * segmentations.squeeze()).item()
        denominator = torch.sum(outputs.argmax(dim=1).squeeze()).item() + torch.sum(segmentations.squeeze()).item()
        dice_score = ((2*numerator) / denominator)

        return dice_score


def set_seed(seed: int) -> None:
    """TODO: Docstring"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def save_conda_env(config: ParamConfigurator) -> None:
    """TODO: Docstring"""

    conda_env = os.environ['CONDA_DEFAULT_ENV']
    command = f"conda env export -n {conda_env} > {config.artifact_dir}environment.yml"
    subprocess.call(command, shell=True)
    mlflow.log_artifact(f"{config.artifact_dir}environment.yml")


def load_model(config: ParamConfigurator):
    """TODO: Docstring"""

    model = ResUNet3D(n_channels=1,
                      n_classes=2,
                      f_maps=config.f_maps,
                      levels=config.levels,
                      residual_block=config.residual_block,
                      se_block=config.se_block,
                      attention=config.attention,
                      trilinear=config.trilinear,
                      MHTSA_heads=config.MHTSA_heads,
                      MHGSA_heads=config.MHGSA_heads,
                      MSSC=config.MSSC
                      )
    model.name = config.model_name
    model = model.to(config.device)

    return model
