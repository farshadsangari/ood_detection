from utils import load_model, save_model
import torch
from torch import optim
import pandas as pd
from utils import AverageMeter
from tqdm import tqdm
from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR


def Training(
    model,
    criterion,
    train_loader,
    val_loader,
    config,
):
    global report
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = ExponentialLR(optimizer, gamma=config.gamma)
    if config.ckpt_load_path:
        model, optimizer = load_model(
            ckpt_path=config.ckpt_load_path, model=model, optimizer=optimizer
        )

    report = pd.DataFrame(
        columns=[
            "mode",
            "epoch",
            "batch_index",
            "batch_size",
            "losses_batch",
            "losses_avg",
        ]
    )

    for epoch in tqdm(range(1, config.max_epoch + 1)):
        loss_terms = {}
        for i, (key, value) in enumerate(criterion.name_terms_to_return):
            loss_terms[key] = AverageMeter() if value else 0
        model.train()
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True,
        )
        for batch_idx, (x, y) in loop_train:
            new_loss_terms = criterion.in_distribution_performance(
                model,
                data=x,
                labels=y,
                optimizer=optimizer,
                device=config.device,
                is_learning=True,
            )
            # Update losses
            for i, (key, value) in enumerate(criterion.name_terms_to_return):
                if value:
                    loss_terms[key].update(round(new_loss_terms[i], 3), x.size(0))

                else:
                    loss_terms[key] = new_loss_terms[i]

            # Create an OrderedDict
            avg_loss_terms = OrderedDict()
            for i, (key, value) in enumerate(criterion.name_terms_to_return):
                avg_loss_terms[key] = (
                    loss_terms[key].avg if value else new_loss_terms[i]
                )

            new_row = pd.DataFrame(
                {
                    "mode": "train",
                    "epoch": epoch,
                    "batch_index": batch_idx,
                    "batch_size": x.shape[0],
                    "losses_batch": [new_loss_terms],
                    "losses_avg": [avg_loss_terms],
                },
                index=[0],
            )

            report.loc[len(report)] = new_row.values[0]

            loop_train.set_description(f"Train - Epoch : {epoch}")
            loop_train.set_postfix(
                ordered_dict=avg_loss_terms,
                refresh=True,
            )
        if val_loader:
            model.eval()
            mode = "val"
            with torch.no_grad():
                loop_val = tqdm(
                    enumerate(val_loader, 1),
                    total=len(val_loader),
                    desc="val",
                    position=0,
                    leave=True,
                )

                for batch_idx, (x, y) in loop_val:
                    new_loss_terms = criterion.in_distribution_performance(
                        model,
                        data=x,
                        labels=y,
                        optimizer=optimizer,
                        device=config.device,
                        is_learning=False,
                    )
                    # Update losses
                    for i, (key, value) in enumerate(criterion.name_terms_to_return):
                        if value:
                            loss_terms[key].update(
                                round(new_loss_terms[i], 3), x.size(0)
                            )
                        else:
                            loss_terms[key] = new_loss_terms[i]

                    # Create an OrderedDict
                    avg_loss_terms = OrderedDict()
                    for i, (key, value) in enumerate(criterion.name_terms_to_return):
                        avg_loss_terms[key] = (
                            loss_terms[key].avg if value else new_loss_terms[i]
                        )

                    new_row = pd.DataFrame(
                        {
                            "mode": "val",
                            "epoch": epoch,
                            "batch_index": batch_idx,
                            "batch_size": x.shape[0],
                            "losses_batch": [new_loss_terms],
                            "losses_avg": [avg_loss_terms],
                        },
                        index=[0],
                    )

                    report.loc[len(report)] = new_row.values[0]

                    loop_val.set_description(f"Validation - Epoch : {epoch}")
                    loop_val.set_postfix(
                        ordered_dict=avg_loss_terms,
                        refresh=True,
                    )

        if epoch % config.ckpt_save_freq == 0:
            save_model(
                file_path=config.ckpt_save_root,
                file_name=f"ckpt_{config.model_name}_epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
            )

            report.to_csv(
                f"{config.report_root}/report_{config.model_name}_epoch_{epoch}.csv"
            )
    save_model(
        file_path=config.ckpt_save_root,
        file_name=f"ckpt_{config.model_name}_epoch_{epoch}.pt",
        model=model,
        optimizer=optimizer,
    )
    lr_scheduler.step()
    report.to_csv(f"{config.report_root}/report_{config.model_name}_epoch_{epoch}.csv")
    return model, optimizer, report
