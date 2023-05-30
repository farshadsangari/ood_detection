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
        configs.dataloader
    )
    a = validation_dataset.get_custom_classes_data(
        custom_classes=config.list_custom_classes_eval_phase
    )
    model = MyResNet18(**configs.model).to(configs.base.device)
    criterion = get_loss(**OmegaConf.merge(configs.loss))

    model, optimizer, report = Training(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=None,
        config=OmegaConf.merge(configs.model, configs.learning, configs.base),
    )
    return model, optimizer, report


if __name__ == "__main__":
    configs = OmegaConf.load("./configs/sol3.yaml")
    main(configs)
