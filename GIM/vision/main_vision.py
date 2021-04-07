import torch
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

#### own modules
from GIM.utils import logger
from GIM.vision.arg_parser import arg_parser
from GIM.vision.models import load_vision_model
from GIM.vision.data import get_dataloader


def validate(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss


def train(opt, model):
    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 200
    log_idx = 50

    starttime = time.time()
    cur_train_module = opt.train_module

    record = {}


    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        for step, (img, label) in enumerate(train_loader):
            print(step)

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            # loop through the losses of the modules and do gradient descent
            model.zero_grad()

            for idx, cur_losses in enumerate(loss):
                if len(loss) == 1 and opt.model_splits != 1: # 1 (normal end-to-end backprop) or 3 (default used in experiments of paper)
                    idx = cur_train_module
                    
                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                    
            for idx, cur_losses in enumerate(loss):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module
                
                optimizer[idx].step()

                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

            step_time = time.time() - starttime
            if ((step + 1) % log_idx) == 0:
                log_metrics(epoch, loss, starttime, step_time, step)

        if opt.validate:
            validation_loss = validate(opt, model, test_loader) #test_loader corresponds to validation set here
            logs.append_val_loss(validation_loss)

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])

        for name, param in model.named_parameters():
            if name == 'module.model.0.0.model.Conv1.conv1.weight':
                if 'conv1' not in record.keys():
                    record['conv1'] = {}
                    record['conv1']['kern1'] = []
                    record['conv1']['kern1'].append(param.grad[:,0,:,:].cpu())
                else:
                    record['conv1']['kern1'].append(param.grad[:,0,:,:].cpu())

            # store 3
            if name == 'module.model.0.1.model.Conv2.conv1.weight':
                if 'conv2' not in record.keys():
                    record['conv2'] = {}
                    record['conv2']['kern1'] = []
                    record['conv2']['kern1'].append(param.grad[ : , 1, : , : ].cpu())
                else:
                    record['conv2']['kern1'].append(param.grad[ : , 1, : , : ].cpu())

        # grads_weight = torch.sum(abs(model.module.encoder[0].model.Conv1.conv1.weight.grad))
        # logs.append_grad_weight(grads_weight.item())

        if epoch % log_idx == 0:
            logs.create_log(model, epoch=epoch, optimizer=optimizer)

        # summary_writer.add_histogram('conv1.weight.grad', model.module.encoder[0].model.Conv1.conv1.weight.grad, epoch)
        # summary_writer.add_histogram('conv2.weight.grad', model.module.encoder[1].model.Conv2.conv1.weight.grad, epoch)

        # for end to end
        # summary_writer.add_histogram('conv1.weight.grad', model.module.encoder[0].model.Conv1.conv1.weight.grad, epoch)
        # summary_writer.add_histogram('conv2.weight.grad', model.module.encoder[0].model.Conv2.conv1.weight.grad, epoch)
    return record


#Adds the loss, time taken loading data, and time taken completing steps to the logs
def log_metrics(epoch, loss, data_load_time, step_time, step):
    summary_writer.add_scalar("epoch", epoch, step)
    summary_writer.add_scalar(
            "time/data", data_load_time, step
    )
    summary_writer.add_scalar(
            "time/data", step_time, step
    )
#Creates a unique sub directory to log this run in
def get_summary_writer_log_dir(opt):
    # tb_log_dir_prefix = f'CNN_bs={opt.batch_size}_lr={opt.learning_rate}_run_'
    # i = 0
    # while i < 1000:
    #     tb_log_dir = opt.save_dir / (tb_log_dir_prefix + str(i))
    #     if not tb_log_dir.exists():
    #         return str(tb_log_dir)
    #     i += 1
    # return str(tb_log_dir)
    return "tensor-logs"

def draw_grad_weight_heatmap(data):

    data = np.array(data)
    data = np.transpose(data)

    kernels = np.arange(1, len(data)+1, 2)
    epochs = np.arange(1, len(data[0])+1, len(data[0])//2)

    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=0, vmax=1)
    ax.set_xticks(epochs)
    ax.set_yticks(kernels)
    ax.set_xticklabels(epochs)
    ax.set_yticklabels(kernels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
    ax.invert_yaxis()
    plt.xlabel('Epoch')
    plt.ylabel('Input')
    fig.tight_layout()
    name = name + ".png"
    plt.savefig(name)

if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    # init tensorboard
    log_dir = get_summary_writer_log_dir(opt)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        records = train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    # calc and create plot
    records['conv1']['kern1'] = [i.reshape(i.shape[0], i.shape[1]*i.shape[2]) for i in records['conv1']['kern1']]
    records['conv1']['kern1'] = [torch.abs(i) for i in records['conv1']['kern1']]
    records['conv1']['kern1'] = [torch.sum(i, dim=1) for i in records['conv1']['kern1']]
    records['conv1']['kern1'] = [i.numpy() for i in records['conv1']['kern1']]
    draw_grad_weight_heatmap(records['conv1']['kern1'])


    logs.create_log(model)
    summary_writer.close()
