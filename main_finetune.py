import argparse
import copy
import torch
from datasets import get_loader
from engine_finetune import train_one_epoch
from models import get_model
import os
from datetime import datetime
from torch.utils import data
from util import (
    accuracy,
    AverageMeter,
    dump_logs,
)
from util.clip_utils import clip_config
from util.FTP import SGDP
from util.FTP import AdamP


def train(logdir, args):
    torch.manual_seed(0)
    dump_logs(logdir, "Let the games begin")
    device = torch.device("cuda")

    # Setup Dataloader
    t_loader = get_loader(
        "train", name=args.dataset, root=args.root, data_dir=args.data_dir, site=args.site, percent=args.percent
    )
    v_loader = get_loader(
        "val", name=args.dataset, root=args.root, data_dir=args.data_dir, site=args.site, percent=args.percent
    )
   
    trainloader = data.DataLoader(
        t_loader,
        shuffle=True,
        batch_size=args.batch_size*args.gpu_per_node,
        num_workers=args.n_workers,
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=args.batch_size*args.gpu_per_node,
        num_workers=args.n_workers,
    )
    n_classes = args.n_classes    

    # Setup Model and Load pretrain
    model_cfg = {"arch": args.arch}
    if args.load_pretrained is not None:
        if os.path.isfile(args.load_pretrained):
            info = "Loading model and optimizer from checkpoint '{}'".format(
                args.load_pretrained
            )
            dump_logs(logdir, info + "\n")

            with open(args.load_pretrained, "rb") as fp:
                checkpoint = torch.load(fp)

            if "clip" in args.load_pretrained:
                checkpoint = checkpoint.state_dict()
                clip_config(model_cfg, checkpoint, pretrained=True)
                checkpoint = {
                    k.replace("visual.", ""): v
                    for k, v in checkpoint.items()
                    if "transformer" not in k
                }

            elif "moco" in args.load_pretrained:
                checkpoint = checkpoint["state_dict"]
                checkpoint = {
                    k.replace("base_encoder.", "").replace("module.", ""): v
                    for k, v in checkpoint.items()
                }

            model = get_model(**model_cfg, num_classes=n_classes).to(device)

            model_dict = model.state_dict()
            filtered_checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model.load_state_dict(filtered_checkpoint, strict=False)
            info = "Loaded pretrained model '{}' and {}/{} layers".format(
                args.load_pretrained, len(filtered_checkpoint), len(model_dict)
            )
            dump_logs(logdir, info + "\n")
            print(info)
        else:
            info = "No pretrained model found at '{}'".format(args.load_pretrained)
            print(info)
            dump_logs(logdir, info + "\n")
            model = get_model(**model_cfg, num_classes=n_classes).to(device)
    else:
        info = "Use random initialization"
        dump_logs(logdir, info + "\n")
        print(info)
        model = get_model(**model_cfg, num_classes=n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimization parameters
    
    if args.opt == "sgdp":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": 0.0,
            "momentum": 0.9,
            "nesterov": True,
            "k": 1, 
            "exclude_set": {'module.head.weight','module.head.bias'}
        } 
        # Cache pre-trained model weights 
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params':params_to_opt,
                        'pre': params_anchor, 
                        'name': params_to_opt_name}]
        optimizer = SGDP(param_group,**optimizer_params)

    elif args.opt == "adamp":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": 0.0,
            "k": 1, 
            "exclude_set": {'module.head.weight','module.head.bias'}
        } 

        # Cache pre-trained model weights 
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params':params_to_opt,
                        'pre': params_anchor, 
                        'name': params_to_opt_name}]
        optimizer = AdamP(param_group,**optimizer_params)
    else:
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": 5.0e-4,
            "momentum": 0.9,
            "nesterov": True,
        }   
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # ================================ Training ==========================================
    start_epoch = 0
    best_acc1 = -100.0
    best_model = model
    for epoch in range(start_epoch, args.epoch):
        best_acc1, best_model = train_one_epoch(
            args,
            model,
            loss_fn,
            optimizer,
            scheduler,
            trainloader,
            valloader,
            device,
            logdir,
            epoch,
            best_acc1,
            best_model,
        )
    ##================== Testing ============================
    print("start testing")
    sites = ["real", "sketch", "painting", "infograph", "clipart"]
    datasets = [
        get_loader("test", name=args.dataset, root=args.root,  data_dir=args.data_dir, site=site)
        for site in sites
    ]
    loaders = [
        data.DataLoader(
            dataset,
            batch_size=args.batch_size * args.gpu_per_node * 2,
            num_workers=args.n_workers,
        )
        for dataset in datasets
    ]
    
    best_model.eval()
    with torch.no_grad():
        for site, loader in zip(sites, loaders):
            test_top1 = AverageMeter("Acc@1", ":6.2f")
            test_top5 = AverageMeter("Acc@5", ":6.2f")
            for i, (image, target) in enumerate(loader):
                image = image.to(device)
                target = target.to(device)
                logit = best_model(image)
                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                test_top1.update(acc1[0], image.size(0))
                test_top5.update(acc5[0], image.size(0))
                if i % args.print_interval == 0:
                    output = "{} test: [{}/{}]".format(
                        site,
                        i,
                        len(loader),
                    )
                    print(output)

            output = "{site} test results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t".format(
                site=site,
                top1=test_top1,
                top5=test_top5,
            )

            print(output)
            dump_logs(logdir, output + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--arch",
        nargs="?",
        type=str,
        default="clip_resnet50",
        help="Backbone architecture",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Specify data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Specify save directory",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="pretrained model direcotry",
    )
    parser.add_argument(
        "--id",
        nargs="?",
        type=str,
        default=None,
        help="Additional run information",
    )
    parser.add_argument(
        "--epoch",
        default=200,
        type=int,
        help="training epoch",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--print_interval",
        default=100,
        type=int,
        help="print interval",
    )
    parser.add_argument(
        "--val_freq",
        default=1,
        type=int,
        help="print interval",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--num_workers", default=4, type=int)

    # dataset parameters
    parser.add_argument(
        "--root",
        nargs="?",
        type=str,
        default= "./datasets/",
        help="data root",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="domainnet",
        help="dataset name",
    )
    parser.add_argument(
        "--site",
        nargs="?",
        type=str,
        default="real",
        help="DomainNet site",
    )
    parser.add_argument(
        "--percent",
        nargs="?",
        type=str,
        default="5",
        help="DomainNet percentage",
    )
    parser.add_argument(
        "--n_classes",
        nargs="?",
        type=int,
        default=345,
        help="dataset classes",
    )
    # optimizer
    parser.add_argument(
        "--opt",
        nargs="?",
        type=str,
        default="sgd",
        help="optimizer type",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Custom Learning Rate",
    )
    parser.add_argument(
        "--gpu_per_node", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "--k", default=1.0, type=float, help="Hyperparamter for FTP"
    )

    args = parser.parse_args()
    now = datetime.now()
    logdir = args.output_dir+"log/{}_{}".format(args.id,now.strftime("%d_%m_%Y_%H_%M_%S"))
    args.output_dir = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print("RUNDIR: {}".format(logdir))
    train(logdir, args)

