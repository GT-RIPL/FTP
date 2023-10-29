import copy
import time
import torch
from util import accuracy, AverageMeter, dump_logs, save_on_master

def train_one_epoch(
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
):

    train_time = AverageMeter("train_time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    prob_meter = AverageMeter("prob", ":.4e")

    model.train()
    for i, (image, target) in enumerate(trainloader):
        image = image.to(device)
        target = target.to(device)

        start_time = time.time() 
        logit = model(image)
        loss = loss_fn(logit, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure train time
        train_time.update(time.time() - start_time)

        # measure accuracy and record loss
        pred_dist = torch.nn.functional.softmax(logit, dim=1)
        acc1, acc5 = accuracy(logit, target, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        prob_meter.update(pred_dist.max(1)[0].mean().item(), pred_dist.size(0))


        if i % args.print_interval == 0:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Train_time {train_time.val:.3f} ({train_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t".format( 
                    epoch,
                    i,
                    len(trainloader),
                    train_time=train_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"],
                )
            )
            print(output)
            dump_logs(logdir, output + "\n")

    scheduler.step()
    ##================== Evaluation ============================
    if epoch % args.val_freq == 0:
        model.eval()
        with torch.no_grad():
            eval_top1 = AverageMeter("Acc@1", ":6.2f")
            eval_top5 = AverageMeter("Acc@5", ":6.2f")
            val_losses = AverageMeter("Loss", ":.4e")
            val_prob_meter = AverageMeter("prob", ":.4e")

            for i, (image, target) in enumerate(valloader):

                image = image.to(device)
                target = target.to(device)

                logit = model(image)
                loss = loss_fn(logit, target)

                pred_dist = torch.nn.functional.softmax(logit, dim=1)
                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                eval_top1.update(acc1[0], image.size(0))
                eval_top5.update(acc5[0], image.size(0))

                val_prob_meter.update(
                    pred_dist.max(1)[0].mean().item(), pred_dist.size(0)
                )
                val_losses.update(loss.item(), image.size(0))

                if i % args.print_interval == 0:
                    output = (
                        "Val: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                        "Prec@1 {top1.avg:.3f}\t"
                        "Prec@5 {top5.avg:.3f}".format(
                            epoch,
                            i,
                            len(valloader),
                            top1=eval_top1,
                            top5=eval_top5,
                            lr=optimizer.param_groups[-1]["lr"],
                        )
                    )
                    print(output)

            output = (
                "validation Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t"
                "Val Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    top1=eval_top1,
                    top5=eval_top5,
                    loss=val_losses,
                )
            )
            print(output)
            dump_logs(logdir, output + "\n")

            is_best = eval_top1.avg > best_acc1
            if is_best:
                best_model = copy.deepcopy(model)
                best_acc1 = max(eval_top1.avg, best_acc1)
                output_best = "Best Prec@1: %.3f" % (best_acc1)
                dump_logs(logdir, output_best + "\n")
                print(output_best)
        state = {
            "epoch": epoch,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        }
        save_on_master(state, logdir, epoch, is_best)
    return best_acc1, best_model
