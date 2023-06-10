import warnings
from losses import get_loss
from dataloaders import get_loaders
from learning import Training
from omegaconf import OmegaConf
from models import MyResNet18
from utils import load_model
from evaluation import Evaluation, find_threshold

warnings.filterwarnings("ignore")


def main(configs):
    global model
    list_of_inlier_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "horse",
        "ship",
        "truck",
    ]
    list_of_outlier_classes = ["frog"]
    configs.eval.path_model_eval = "./ckpts/ckpt_MyResNet18_frog_epoch_200.pt"
    configs.eval.list_custom_classes_eval_phase = list_of_inlier_classes
    inlier_loader = get_loaders(configs, eval_phase=True)
    configs.eval.list_custom_classes_eval_phase = list_of_outlier_classes
    outlier_loader = get_loaders(configs, eval_phase=True)

    model = MyResNet18(**configs.model).to(configs.base.device)

    model, optimizer = load_model(
        ckpt_path=configs.eval.path_model_eval, model=model, optimizer=None
    )

    criterion = get_loss(
        loss_module_name=configs.loss.loss_module_name,
        is_ood=configs.eval.is_ood,
        kwargs=configs.loss.loss_fn_args,
    )
    # Find threshold
    # find_threshold(
    #     model=model, criterion=criterion, eval_loader=inlier_loader, configs=configs
    # )

    configs.eval.threshold = 0.54
    configs.eval.is_ood = True
    Evaluation(
        model=model, criterion=criterion, eval_loader=outlier_loader, configs=configs
    )


if __name__ == "__main__":
    configs = OmegaConf.load("./configs/sol3.yaml")
    main(configs)
