import warnings
from losses import get_loss
from dataloaders import get_loaders
from learning import Training
from omegaconf import OmegaConf
from models import MyResNet18

warnings.filterwarnings("ignore")


def main(configs):
    global model
    train_dataset, validation_dataset, train_loader, validation_loader = get_loaders(
        configs, eval_phase=False
    )
    model = MyResNet18(**configs.model).to(configs.base.device)
    criterion = get_loss(
        loss_module_name=configs.loss.loss_module_name,
        is_ood=False,
        kwargs=configs.loss.loss_fn_args,
    )

    model, optimizer, report = Training(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=validation_loader,
        config=OmegaConf.merge(configs.model, configs.learning, configs.base),
    )
    return model, optimizer, report


if __name__ == "__main__":
    configs = OmegaConf.load("./configs/sol3.yaml")
    main(configs)
