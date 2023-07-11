from micromind import PhiNet

from ultralytics import YOLO

from yolo.model import microYOLO
from yolo.microyolohead import Microhead

import torchinfo


def train_nn():

    _alpha = 0.3

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=_alpha,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=1,
        compatibility=True,
        downsampling_layers=[5, 7],  # S2
        pool=False,
        h_swish=False,
        squeeze_excite=False,
    )

    # torchinfo head + backbone

    # squeeze and excite False
    # Salvare numero di flop
    #

    # define head
    head = Microhead(
        feature_sizes=[
            int(16 * _alpha / 0.67),
            int(32 * _alpha / 0.67),
            int(64 * _alpha / 0.67),
        ],
        concat_layers=[6, 4],
        head_concat_layers=[15],
    )

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=80, imgsz=320
    )  # build a new model from scratch DEFAULT_CFG

    torchinfo.summary(
        model.model,
        input_size=(1, 3, 320, 320),
        device="cpu",
        col_names=(
            "input_size",
            "output_size",
            "mult_adds",
            "num_params",
            "params_percent",
        ),
    )

    # load a model
    model1 = YOLO("yolov8n.yaml")  # build a new model from YAML

    # print model info
    torchinfo.summary(
        model1.model,
        input_size=(1, 3, 320, 320),
        device="cpu",
        col_names=(
            "input_size",
            "output_size",
            "mult_adds",
            "num_params",
            "params_percent",
        ),
        depth=2,
        verbose=1,
    )


"""
    # Train the model
    model1.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=320,
        device="cpu",
        task="detect",
    )
    #model.export()
"""

if __name__ == "__main__":
    train_nn()
