import shutil
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
from dataset import SarcomaDataset
from ParamConfigurator import ParamConfigurator
from tqdm import tqdm
import mlflow
from torch.utils.tensorboard import SummaryWriter
import uuid
from utils import set_seed, save_conda_env, load_model, DiceCELoss, DiceSimilarityCoefficient


def main(config: ParamConfigurator, writer: torch.utils.tensorboard.SummaryWriter):
    set_seed(config.seed)
    run_uuid = str(uuid.uuid4())
    run_dir = f"{config.artifact_dir}{run_uuid}"
    os.makedirs(run_dir, exist_ok=True)

    mlflow.log_params(config.__dict__)
    save_conda_env(config)

    model = load_model(config=config)

    train_data = SarcomaDataset(config=config, mode="train")
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                  pin_memory=config.pin_memory, num_workers=config.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = DiceCELoss(config=config)
    dice_score = DiceSimilarityCoefficient(config=config)

    best_val_dice = 0.0
    best_val_loss = np.inf

    for epoch in range(1, config.epochs + 1):
        running_train_loss = []
        running_train_dice = []
        running_val_dice = []
        running_val_loss = []
        running_test_dice = []
        running_test_loss = []

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch} | {config.epochs}")

                volumes, segmentations, patient_id = data
                volumes = volumes.to(torch.float32).cuda()
                segmentations = segmentations.to(torch.float32).cuda()

                optimizer.zero_grad()

                try:

                    outputs = model(volumes)

                    middle_slice_idx = int(outputs.shape[-3]/2)
                    output_arr = outputs.squeeze().argmax(dim=0)[middle_slice_idx].numpy(force=True)
                    seg_arr = segmentations.squeeze()[middle_slice_idx].numpy(force=True)

                    pred_tag = patient_id[0] + "_prediction_train"
                    writer.add_image(tag=pred_tag, img_tensor=output_arr,
                                     global_step=epoch, dataformats="WH")

                    true_tag = patient_id[0] + "_true_train"
                    writer.add_image(tag=true_tag, img_tensor=seg_arr,
                                     global_step=epoch, dataformats="WH")

                    loss = loss_fn(outputs, segmentations)
                    loss.backward()
                    optimizer.step()

                    running_train_loss.append(loss.item())

                    score = dice_score(outputs, segmentations)
                    running_train_dice.append(score)

                    tepoch.set_postfix(loss=np.mean(running_train_loss), dice=np.mean(running_train_dice))

                except:
                    continue

            epoch_loss = np.mean(running_train_loss).item()
            epoch_dice = np.mean(running_train_dice).item()
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_dice", epoch_dice, step=epoch)

        ##############
        # Validation #
        ##############

        val_data = SarcomaDataset(config=config, mode="val")
        val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False,
                                    pin_memory=config.pin_memory, num_workers=config.num_workers)

        model.eval()
        with tqdm(val_dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Validation")

                volumes, segmentations, patient_id = data
                volumes = volumes.to(torch.float32).cuda()
                segmentations = segmentations.to(torch.float32).cuda()

                try:
                    outputs = model(volumes)

                    middle_slice_idx = int(outputs.shape[-1] / 2)
                    output_arr = outputs.squeeze().argmax(dim=0)[:, :, middle_slice_idx].numpy(force=True)
                    seg_arr = segmentations.squeeze()[:, :, middle_slice_idx].numpy(force=True)

                    pred_tag = patient_id[0] + "_prediction_val"
                    writer.add_image(tag=pred_tag, img_tensor=output_arr,
                                     global_step=epoch, dataformats="WH")

                    true_tag = patient_id[0] + "_true_val"
                    writer.add_image(tag=true_tag, img_tensor=seg_arr,
                                     global_step=epoch, dataformats="WH")

                    loss = loss_fn(outputs, segmentations)
                    running_val_loss.append(loss.item())

                    score = dice_score(outputs, segmentations)
                    running_val_dice.append(score)

                    tepoch.set_postfix(loss=np.mean(running_val_loss), dice=np.mean(running_val_dice))

                except:
                    continue

            val_loss = np.mean(running_val_loss).item()
            val_dice = np.mean(running_val_dice).item()
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{run_dir}/best_loss_model.pth"
            torch.save(model.state_dict(), model_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_path = f"{run_dir}/best_dice_model.pth"
            torch.save(model.state_dict(), model_path)

        if epoch == config.epochs:
            model_path = f"{run_dir}/last_epoch_model.pth"
            torch.save(model.state_dict(), model_path)

    mlflow.log_artifacts(run_dir)

    ############################
    # Test on external dataset #
    ############################

    model = load_model(config=config)
    model.load_state_dict(torch.load(f"{run_dir}/best_dice_model.pth"))

    test_data = SarcomaDataset(config=config, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,
                                 pin_memory=config.pin_memory, num_workers=config.num_workers)

    model.eval()
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Test")

            volumes, segmentations, patient_id = data
            volumes = volumes.to(torch.float32).cuda()
            segmentations = segmentations.to(torch.float32).cuda()

            try:
                outputs = model(volumes)

                middle_slice_idx = int(outputs.shape[-1] / 2)
                output_arr = outputs.squeeze().argmax(dim=0)[:, :, middle_slice_idx].numpy(force=True)
                seg_arr = segmentations.squeeze()[:, :, middle_slice_idx].numpy(force=True)

                pred_tag = patient_id[0] + "_prediction_val"
                writer.add_image(tag=pred_tag, img_tensor=output_arr, dataformats="WH")

                true_tag = patient_id[0] + "_true_val"
                writer.add_image(tag=true_tag, img_tensor=seg_arr, dataformats="WH")

                loss = loss_fn(outputs, segmentations)
                running_test_loss.append(loss.item())

                score = dice_score(outputs, segmentations)
                running_test_dice.append(score)

                tepoch.set_postfix(loss=np.mean(running_test_loss), dice=np.mean(running_test_dice))

            except:
                continue

        test_loss = np.mean(running_test_loss).item()
        test_mean_dice = np.mean(running_test_dice).item()
        test_median_dice = np.median(running_test_dice).item()
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mean_dice", test_mean_dice)
        mlflow.log_metric("test_median_dice", test_median_dice)

        print(f"Loss: {test_loss}")
        print(f"Mean Dice: {test_mean_dice}")
        print(f"Median Dice: {test_median_dice}")
        shutil.rmtree(run_dir)


if __name__ == "__main__":

    writer = SummaryWriter()
    config = ParamConfigurator()

    mlflow.start_run()
    main(config, writer)
    mlflow.end_run()
