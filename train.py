import glob
import cv2
import torch
import random
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np 
from sklearn import preprocessing

def create_index_map(label_list): 
    index_map = {}
    for i in range(200):
        index_map[i] = []
    for index, label in enumerate(label_list):
        index_map[label].append(index)
    return index_map

def to_batch(x, y, batch_size, device):
    data_size = len(y)
    ret_x, ret_y = [], []
    
    iter_num = (data_size + (batch_size - 1)) // batch_size
    for temp_batch in range(0, iter_num - 1):
        batch_x = x[temp_batch * batch_size : (temp_batch + 1) * batch_size]
        batch_y = y[temp_batch * batch_size : (temp_batch + 1) * batch_size]
        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)
        ret_x.append(batch_x)
        ret_y.append(batch_y)
    batch_x = x[-batch_size:]
    batch_y = y[-batch_size:]
    ret_x.append(batch_x)
    ret_y.append(batch_y)

    ret_x = torch.stack(ret_x).to(device)
    ret_y = torch.stack(ret_y).to(device)
    return ret_x, ret_y

def train_test_split(x, y, index_map, split_index, batch_size, device):

    train_x, train_y, test_x, test_y = [], [], [], []
    for cls in index_map:
        for i, index in enumerate(index_map[cls]):
            if len(index_map[cls]) - i > split_index:
                train_x.append(x[index])
                train_y.append(y[index])
            else :
                test_x.append(x[index])
                test_y.append(y[index])
    
    # must in same length
    assert len(y) == len(train_y) + len(test_y) and len(train_x) == len(train_y) and len(test_x) == len(test_y)

    train_x, train_y = torch.stack(train_x), torch.tensor(train_y)
    test_x, test_y = torch.stack(test_x), torch.tensor(test_y)
    random_train_list = random.sample(range(len(train_y)), len(train_y))
    random_test_list = random.sample(range(len(test_y)), len(test_y))
    
    train_x = train_x[random_train_list]
    train_y = train_y[random_train_list]
    test_x = test_x[random_test_list]
    test_y = test_y[random_test_list]

    train_x, train_y = to_batch(train_x, train_y, batch_size, device)
    test_x, test_y = to_batch(test_x, test_y, batch_size, device)

    return train_x, train_y, test_x, test_y

def train(model, train_x, train_y):

    model.train()

    inner_num = train_x.size()[0]
    train_loss = 0.0
    train_acc = 0.0

    for i in range(inner_num):
        optimizer.zero_grad()
        input = train_x[i]
        output = model.forward(input)

        # count loss
        tmp_loss = criterion(output, train_y[i])
        tmp_loss.backward()
        optimizer.step()
        train_loss += tmp_loss.item()

        # count accuracy
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == train_y[i].view(*top_class.shape)
        train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
    
    train_loss /= inner_num
    train_acc /= inner_num

    return train_loss, train_acc

def test(model, test_x, test_y):

    model.eval()

    inner_num = test_x.size()[0]
    test_loss = 0.0
    test_acc = 0.0

    for i in range(inner_num):
        optimizer.zero_grad()
        input = test_x[i]
        output = model.forward(input)

        # count loss
        tmp_loss = criterion(output, test_y[i])
        test_loss += tmp_loss.item()

        # count accuracy
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == test_y[i].view(*top_class.shape)
        test_acc += torch.mean(equals.type(torch.FloatTensor)).item()
    
    test_loss /= inner_num
    test_acc /= inner_num

    return test_loss, test_acc

# load classes to dict
f_class = open('data/classes.txt', 'r')
class_list = [str(c).strip() for c in f_class.readlines()]
text_map = {}
for cls in class_list:
    text_map[int(cls.split('.')[0]) - 1] = cls
class_map = {}
for cls in class_list:
    class_map[cls] = int(cls.split('.')[0]) - 1 # 0 ~ 199

# load training filenames
f_training_info = open('data/training_labels.txt', 'r')
training_info = [str(c).strip('\n') for c in f_training_info.readlines()]

# config files 
train_img_path = 'data/training_images/'
test_img_path = 'data/testing_images/'

# judge device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('the divice for training : {}'.format(device))

# data preprocessing object
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# parameters for training
epochs = 200
classes = 200
batch_size = 10
learning_rate = 0.0001

# setting training and testing data
print('preprocessing data...')
x_raw = [preprocess(Image.open('{}{}'.format(train_img_path, info.split()[0]))) for info in training_info]
y_raw = [class_map[info.split()[1]] for info in training_info]
index_map = create_index_map(y_raw)
train_x, train_y, test_x, test_y = train_test_split(x_raw, y_raw, index_map, 2, batch_size, device)

print(train_x.size(), train_y.size(), test_x.size(), test_y.size())

# setting model
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(0.2),
                                 torch.nn.Linear(1024, classes),
                                 torch.nn.LogSoftmax(dim=1))
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
model.to(device)

inner_train_num = train_x.size()[0]
inner_test_num = test_x.size()[0]
min_loss = np.inf
tmp_patience = 0
patience_max = 5

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):

    train_loss, train_acc = train(model, train_x, train_y)
    test_loss, test_acc = test(model, test_x, test_y)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print('-------------------------------------')
    print('epoch : {}'.format(epoch + 1))
    print("Train loss {:.5f}".format(train_loss))
    print("Test loss {:.5f}".format(test_loss))
    print("Train Accuracy {:.5f}".format(train_acc))
    print("Test Accuracy {:.5f}".format(test_acc))
    
    if min_loss > test_loss:
        print('training loss improved from {:.5f} to {:.5f}'.format(min_loss, test_loss))
        min_loss = test_loss
        tmp_patience = 0
    elif tmp_patience < patience_max:
        print('training loss not improved from {:.5f}, got {:.5f}'.format(min_loss, test_loss))
        tmp_patience += 1
    else :
        print('early stopping')
        break

fout = open('train_predict.txt', 'w')
x_raw = torch.stack(x_raw)
y_raw = torch.tensor(y_raw)
eval_x, eval_y = to_batch(x_raw, y_raw, batch_size, device)
match_cnt = 0
total_cnt = 0
for i in range(eval_x.size()[0]):
    model.eval()
    input = eval_x[i]
    output = model.forward(input)
    ps = torch.exp(output)
    _, top_class = ps.topk(1, dim=1)
    for cnt, index in enumerate(top_class):
        fout.write('{} {}\n'.format(training_info[i * batch_size + cnt].split(' ')[0], text_map[index.item()]))
        match_cnt += 1 if (training_info[i * batch_size + cnt].split(' ')[1] == text_map[index.item()]) else 0
        total_cnt += 1

fout.close()
print('match : {}/{}, global accuracy = {}'.format(match_cnt, total_cnt, match_cnt / total_cnt))

# dump result
fig, ax = plt.subplots(2, figsize=(10, 10))
ax[0].plot(train_losses, label='training loss')
ax[0].plot(test_losses, label='testing loss')
ax[0].set_title('Loss Curve')
ax[0].legend()

ax[1].plot(train_accs, label='training accuracy')
ax[1].plot(test_accs, label='testing accuracy')
ax[1].set_title('Accuracy Curve')
ax[1].legend()


# save model and training result
fname = 'result/resnet50'
plt.savefig('{}/result.png'.format(fname))
torch.save(model, '{}/dog_00.pth'.format(fname))

print('process complete.')






