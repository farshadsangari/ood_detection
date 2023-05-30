import warnings
from losses import get_loss
from dataloaders import get_loaders
from learning import Training
from omegaconf import OmegaConf
from models import MyResNet18
from utils import load_model
from evaluation import Evaluation

warnings.filterwarnings("ignore")


def main(configs):
    global model
    eval_loader = get_loaders(configs, eval_phase=True)
    model = MyResNet18(**configs.model).to(configs.base.device)

    model, optimizer = load_model(
        ckpt_path=configs.eval.path_model_eval, model=model, optimizer=None
    )

    criterion = get_loss(
        loss_module_name=configs.loss.loss_module_name,
        is_ood=configs.eval.is_ood,
        kwargs=configs.loss.loss_fn_args,
    )

    Evaluation(
        model=model, criterion=criterion, eval_loader=eval_loader, configs=configs
    )


if __name__ == "__main__":
    configs = OmegaConf.load("./configs/sol3.yaml")
    main(configs)
