import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
from models import Resnet101


save_folder = cfg.SAVE_FOLDER
os.makedirs(save_folder, exist_ok=True)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model = Resnet101(num_classes=2)

print("========= Model loaded =========")

if len(cfg.USING_GPU) > 1:
    devices = [int(i) for i in cfg.USING_GPU]
    str_ = ""
    for i in devices:
        str_ += str(i) + " "
    print("========= Using multi GPU {}=========".format(str_))
    model = nn.DataParallel(model, device_ids = devices)
else:
    print("========= Using single GPU {} =========".format(cfg.USING_GPU[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.USING_GPU[0]

if torch.cuda.is_available():
    model.cuda()

lr = 1e-3

optimizer = optim.SGD(model.parameters(), lr = lr, momentum = cfg.MOMENTUM, weight_decay = cfg.WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()

batch_size = cfg.BATCH_SIZE


# load train data
train_datasets = MyDataset('train')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle=True, num_workers=1)
class_dict = train_datasets.get_dict()


max_batch = len(train_datasets) // batch_size
epoch_size = len(train_datasets) // batch_size
max_iter = cfg.MAX_EPOCH * epoch_size

start_iter = 0
epoch = 0

stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
step_index = 0

model.train()


best_acc = 0
save_name = ""

for iteration in range(start_iter, max_iter):

    if iteration % epoch_size == 0:
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1

    if iteration in stepvalues:
        step_index += 1
    lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)
    images, labels = next(batch_iterator)

    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    train_acc = (train_correct.float()) / batch_size

    if train_acc >= best_acc:
        best_acc = train_acc
        save_name = "best.pth"
    else:
        save_name = 'epoch_{}.pth'.format(epoch)
    if iteration % epoch_size == 0:
        if epoch % 5 == 0 and epoch > 0:
            if len(cfg.USING_GPU) > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            'epoch': epoch,
                            'class_dict': class_dict}
                torch.save(checkpoint, os.path.join(save_folder, save_name))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch,
                            'class_dict': class_dict}
                torch.save(checkpoint, os.path.join(save_folder, save_name))

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))
