import warnings
from losses import get_loss
from dataloaders import get_loaders
from learning import Training
from omegaconf import OmegaConf
from models import MyResNet18
from utils import load_model


warnings.filterwarnings("ignore")


def main(configs):
    global model
    eval_dataloader = get_loaders(configs.dataloader, learning_phase=False)
    x, y = eval_dataloader
    model = MyResNet18(**configs.model).to(configs.base.device)

    model, optimizer = load_model(
        ckpt_path=configs.eval.path_model_eval, model=model, optimizer=None
    )

    criterion = get_loss(**OmegaConf.merge(configs.loss))

    new_loss_terms = criterion(
        model,
        data=x,
        labels=y,
        optimizer=optimizer,
        device=configs.base.device,
        is_training=False,
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
        avg_loss_terms[key] = loss_terms[key].avg if value else new_loss_terms[i]


if __name__ == "__main__":
    configs = OmegaConf.load("./configs/sol3.yaml")
    main(configs)
