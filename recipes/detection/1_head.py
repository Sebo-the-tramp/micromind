from micromind import PhiNet

from micromind.yolo import microYOLO
from micromind.yolo import Microhead


def train_nn():

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=0.67,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=80,
        compatibility=False,
        downsampling_layers=[5, 7],  # S2
    )
    # define head
    head = Microhead(
        feature_sizes=[16, 32, 64],
        concat_layers=[6, 4],
        head_concat_layers=[15],
    )

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=80
    )  # build a new model from scratch DEFAULT_CFG

    # Train the model
    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=160,
        device="cpu",
        task="detect",
    )
    model.export()


if __name__ == "__main__":
    train_nn()
