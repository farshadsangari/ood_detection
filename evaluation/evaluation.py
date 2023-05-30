import torch
from utils import AverageMeter
from tqdm import tqdm
from collections import OrderedDict


def Evaluation(
    model,
    criterion,
    eval_loader,
    configs,
):
    loss_terms = {}
    for i, (key, value) in enumerate(criterion.name_terms_to_return):
        loss_terms[key] = AverageMeter() if value else 0

    model.eval()
    with torch.no_grad():
        loop_val = tqdm(
            enumerate(eval_loader, 1),
            total=len(eval_loader),
            desc="eval",
            position=0,
            leave=True,
        )

        for _, (x, y) in loop_val:
            if configs.eval.is_ood:
                new_loss_terms = criterion.ood_performance(
                    model=model,
                    data=x,
                    labels=y,
                    threshold=configs.eval.threshold,
                    device=configs.base.device,
                )
            else:
                new_loss_terms = criterion.in_distribution_performance(
                    model,
                    data=x,
                    labels=y,
                    optimizer=None,
                    device=configs.base.device,
                    is_learning=False,
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

            loop_val.set_description(f"Evaluation: ")
            loop_val.set_postfix(
                ordered_dict=avg_loss_terms,
                refresh=True,
            )
