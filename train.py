import argparse
import os
import random
import numpy as np
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import utils, transforms as T
from PIL import Image
from efficientnet_pytorch import EfficientNet
import TransFG.modeling as transFG

CLASSES = 200


def create_index_map(label_list):
    index_map = {}
    for i in range(200):
        index_map[i] = []
    for index, label in enumerate(label_list):
        index_map[label].append(index)
    return index_map


def load_raw_data(info_path, img_path, transform, class_map, aug_num=1):

    # get raw info
    f_info = open(info_path, 'r')
    raw_info = [str(c).strip('\n') for c in f_info.readlines()]

    # preprocessing data
    x_raw = []
    y_raw = []
    data_size = len(raw_info)
    for i in range(data_size):
        if (i + 1) % 100 == 0:
            print('loading {}/{}'.format(i+1, data_size))
        img = Image.open('{}{}'.format(img_path, raw_info[i].split()[0]))
        cls = class_map[raw_info[i].split()[1]]
        for aug_i in range(aug_num):
            x_raw.append(transform(img).detach())
            y_raw.append(cls)

    return x_raw, y_raw


def to_batch(x, y, batch_size):
    data_size = len(y)
    ret_x, ret_y = [], []

    iter_num = (data_size + (batch_size - 1)) // batch_size
    for temp_batch in range(0, iter_num - 1):
        batch_x = x[temp_batch * batch_size: (temp_batch + 1) * batch_size]
        batch_y = y[temp_batch * batch_size: (temp_batch + 1) * batch_size]
        ret_x.append(batch_x)
        ret_y.append(batch_y)
    batch_x = x[-batch_size:]
    batch_y = y[-batch_size:]
    ret_x.append(batch_x)
    ret_y.append(batch_y)

    ret_x = torch.stack(ret_x)
    ret_y = torch.stack(ret_y)
    return ret_x, ret_y


def train_val_split(x, y, index_map, split_num, batch_size, aug_num):

    assert split_num < 15 and split_num > 0

    train_x, train_y, val_x, val_y = [], [], [], []
    for cls in index_map:
        for i, index in enumerate(index_map[cls]):
            if i >= split_num * aug_num:
                train_x.append(x[index])
                train_y.append(y[index])
            else:
                val_x.append(x[index])
                val_y.append(y[index])

    train_x, train_y = torch.stack(train_x), torch.tensor(train_y)
    val_x, val_y = torch.stack(val_x), torch.tensor(val_y)
    random_train_list = random.sample(range(len(train_y)), len(train_y))
    random_val_list = random.sample(range(len(val_y)), len(val_y))

    train_x = train_x[random_train_list]
    train_y = train_y[random_train_list]
    val_x = val_x[random_val_list]
    val_y = val_y[random_val_list]

    train_x, train_y = to_batch(train_x, train_y, batch_size)
    val_x, val_y = to_batch(val_x, val_y, batch_size)

    return train_x, train_y, val_x, val_y


def train(model, train_x, train_y, criterion, optimizer, device):

    model.to(device)
    model.train()

    inner_num = train_x.size()[0]
    train_loss = 0.0
    train_acc = 0.0

    for i in range(inner_num):
        optimizer.zero_grad()
        input, target = train_x[i].to(
            device).detach(), train_y[i].to(device).detach()
        output = model.forward(input)

        # count loss
        tmp_loss = criterion(output, target)
        tmp_loss.backward()
        optimizer.step()
        train_loss += tmp_loss.item()

        # count accuracy
        _, top_class = torch.max(output, dim=1)
        equals = top_class == target
        train_acc += equals.type(torch.FloatTensor).mean()

        del input
        del target
        torch.cuda.empty_cache()

    train_loss /= inner_num
    train_acc /= inner_num

    return train_loss, train_acc


def val(model, val_x, val_y, criterion, optimizer, device):

    with torch.no_grad():
        model.to(device)
        model.eval()

        inner_num = val_x.size()[0]
        val_loss = 0.0
        val_acc = 0.0

        for i in range(inner_num):
            optimizer.zero_grad()
            input, target = val_x[i].to(
                device).detach(), val_y[i].to(device).detach()
            output = model.forward(input)

            # count loss
            tmp_loss = criterion(output, target)
            val_loss += tmp_loss.item()

            # count accuracy
            _, top_class = torch.max(output, dim=1)
            equals = top_class == target
            val_acc += equals.type(torch.FloatTensor).mean()

            del input
            del target
            torch.cuda.empty_cache()

        val_loss /= inner_num
        val_acc /= inner_num

    return val_loss, val_acc


def build_model(args, pre_trained=True):
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=pre_trained)
        model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, CLASSES),
                                       torch.nn.LogSoftmax(dim=1))
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=pre_trained)
        model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, CLASSES),
                                       torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b0':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b1':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b2':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b4':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'efficientnet-b5':
        model = EfficientNet.from_pretrained(
            args.model, advprop=True, num_classes=CLASSES)
        model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, CLASSES),
                                        torch.nn.LogSoftmax(dim=1))
    elif args.model == 'transFG':
        pretrained_dir = 'TransFG/pretrained/{}.npz'.format(args.fg_type)
        config = transFG.CONFIGS[args.fg_type]
        config.slide_step = int(0.75 * args.batch_size)
        model = transFG.VisionTransformer(
            config, img_size=args.img_size, zero_head=True, num_classes=CLASSES, smoothing_value=0.0)
        model.load_from(np.load(pretrained_dir))

    return model


def main(args):

    # create result dir
    res_root = 'result/{}'.format(args.model)
    if not os.path.exists(res_root):
        os.mkdir(res_root)

    res_num = 1
    res_dir = None
    while True:
        res_dir = '{}/{:02d}'.format(res_root, res_num)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
            break
        res_num += 1

    # save config
    out_config = open('{}/config.cfg'.format(res_dir, res_num), 'w')
    out_config.write('img_size:{}\n'.format(args.img_size))
    out_config.write('aug_num:{}\n'.format(args.aug_num))
    out_config.write('model:{}\n'.format(args.model))
    out_config.write('fg_type:{}\n'.format(args.fg_type))
    out_config.write('batch_size:{}\n'.format(args.batch_size))
    out_config.write('epochs:{}\n'.format(args.epoch))
    out_config.write('patience_max:{}\n'.format(args.patience))
    out_config.write('learning_rate:{}\n'.format(args.lr))
    out_config.write('weight_decay:{}\n'.format(args.weight_decay))
    out_config.close()

    # load classes to dict
    f_class = open('data/classes.txt', 'r')
    class_list = [str(c).strip() for c in f_class.readlines()]
    text_map = {}
    for cls in class_list:
        text_map[int(cls.split('.')[0]) - 1] = cls
    class_map = {}
    for cls in class_list:
        class_map[cls] = int(cls.split('.')[0]) - 1  # 0 ~ 199

    # judge device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('the divice for training : {}'.format(device))

    transform_train = T.Compose([
        T.Resize([args.img_size, args.img_size]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=(-10, 10), translate=(0, 0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x_raw, y_raw = load_raw_data(info_path='data/training_labels.txt', img_path='data/training_images/',
                                 transform=transform_train, class_map=class_map, aug_num=args.aug_num)

    print('preprocessing data...')
    index_map = create_index_map(y_raw)
    train_x, train_y, val_x, val_y = train_val_split(
        x_raw, y_raw, index_map, 1, args.batch_size, args.aug_num)

    print('saving example images...')
    utils.save_image(train_x[0][0], 'example.png')

    print(train_x.size(), train_y.size(), val_x.size(), val_y.size())

    # setting model
    model = build_model(args)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    min_loss = np.inf
    tmp_patience = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # training loop
    print('running training process ...')
    for epoch in range(args.epoch):

        train_loss, train_acc = train(
            model, train_x, train_y, criterion, optimizer, device=device)
        val_loss, val_acc = val(model, val_x, val_y,
                                criterion, optimizer, device=device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print('[epoch : {}, training loss: {:.5f}, training acc : {:5f}, testing loss = {:.5f}, testing acc : {:.5f}]'
              .format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

        torch.save(model, '{}/bird_{:.5f}.pth'.format(res_dir, val_acc))
        print('{}/bird_{:.5f}.pth saved'.format(res_dir, val_acc))

        if min_loss > val_loss and train_acc < 0.995:
            print('testing loss improved from {:.5f} to {:.5f}'.format(
                min_loss, val_loss))
            min_loss = val_loss
            tmp_patience = 0
        elif tmp_patience + 1 < args.patience and train_acc < 0.99:
            print('testing loss not improved from {:.5f}, got {:.5f}'.format(
                min_loss, val_loss))
            tmp_patience += 1
        else:
            print('early stopping')
            break

    # dump result
    fig, ax = plt.subplots(2, figsize=(10, 10))
    ax[0].plot(train_losses, label='train_loss')
    ax[0].plot(val_losses, label='val_loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend()

    ax[1].plot(train_accs, label='train_acc')
    ax[1].plot(val_accs, label='test_acc')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend()

    # save model and training result
    plt.savefig('{}/result.png'.format(res_dir, res_num))

    print('{}/result.png saved'.format(res_dir))
    print('{}/config.png saved'.format(res_dir))
    print('process completed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Bird-Classification-script')
    parser.add_argument('--img_size', '-is', default=320, type=int)
    parser.add_argument('--aug_num', '-a', default=2, type=int)
    parser.add_argument('--batch_size', '-bs', default=6, type=int)
    parser.add_argument('--epoch', '-e', default=100, type=int)
    parser.add_argument('--model', '-m', default='transFG', type=str)
    parser.add_argument('--fg_type', '-fgt', default='ViT-B_16', type=str,
                        help='only useful when the model is transFG, will be [ViT-B_16, ViT-B_32]')
    parser.add_argument('--patience', '-p', default=2, type=int)
    parser.add_argument('--lr', default=5e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    args = parser.parse_args()
    main(args)
