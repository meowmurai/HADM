#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model, DefaultInferencer
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.modeling import GeneralizedRCNNWithTTA, ema
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import json
import datetime

logger = logging.getLogger("detectron2")


def do_test_with_tta(cfg, model):
    # may add normal test results for comparison
    if "evaluator" in cfg.dataloader:
        model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret

    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret    
    
# inference without evaluator
def do_inference(cfg, model):
    input_dir = cfg.inference.input_dir
    output_dir = cfg.inference.output_dir
    output_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    bbox_file = cfg.inference.get("bbox_file", None)
    if bbox_file is not None and os.path.exists(bbox_file):
        with open(bbox_file, "r") as f:
            bboxes = json.load(f)
    else:
        bboxes = None
    os.makedirs(output_dir, exist_ok=True)
    inferencer = DefaultInferencer(cfg, model)
    output_json = {}
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    for path in tqdm(os.listdir(input_dir)):
        im_path = os.path.join(input_dir, path)
        im = read_image(im_path, format="BGR")
        h, w = im.shape[:2]
        aspect = max(h, w) / max(min(h, w), 1)
        logger.info(f"Processing {path}: {w}x{h} (aspect={aspect:.2f})")
        bbox = bboxes[path]['bbox'] if bboxes is not None and path in bboxes else None
        outputs = inferencer(im, bbox=bbox)["instances"]
        outputs_cpu = outputs.to("cpu")
        n_det = len(outputs_cpu)
        logger.info(f"  -> {n_det} detections")
        if n_det > 0:
            scores = outputs_cpu.get("scores").tolist()
            classes = outputs_cpu.get("pred_classes").tolist()
            class_names = [metadata.thing_classes[i] for i in classes]
            for i, (cls_name, score) in enumerate(zip(class_names, scores)):
                logger.info(f"     [{i}] {cls_name}: {score:.3f}")
        else:
            classes = []
            class_names = []
            scores = []
        v = Visualizer(im[:, :, ::-1], metadata=metadata)
        out = v.draw_instance_predictions(outputs_cpu)
        out.save(os.path.join(output_dir, path))
        output_json[path] = {
            "bbox": outputs_cpu.get("pred_boxes").tensor.tolist(),
            "score": scores if isinstance(scores, list) else outputs_cpu.get("scores").tolist(),
            "class": class_names
        }
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(output_json, f)
        
def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter,
                                use_wandb=args.wandb),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    elif args.inference:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        do_inference(cfg, model)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
