import argparse
import torch
from torchvision import transforms as T
from PIL import Image


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


# judge device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('the divice for training : {}'.format(device))

# load model
print('loading model bird_0.89677.pth ...')  # best model
model = torch.load('bird_0.89677.pth', map_location=torch.device(device))

# data preprocessing object
img_size = 320
preprocess = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load classes to dict
f_class = open('data/classes.txt', 'r')
class_list = [str(c).strip() for c in f_class.readlines()]
text_map = {}
for cls in class_list:
    text_map[int(cls.split('.')[0]) - 1] = cls

# load testing filenames
print('loading data ...')
test_img_path = 'crop/testing_images/'
f_testing_info = open('data/testing_img_order.txt', 'r')
batch_size = 1

fout = open('answer.txt', 'w')
testing_info = [str(c).strip() for c in f_testing_info.readlines()]
test_x_raw = [preprocess(Image.open('{}{}'.format(
    test_img_path, info))) for info in testing_info]
test_x_raw = torch.stack(test_x_raw)
test_x, _ = to_batch(test_x_raw, torch.zeros(test_x_raw.size()[0]), batch_size)

total_cnt = 0
remain_bufsize = 0
print('writing result ...')
for i in range(test_x.size()[0]):
    print('processing {}/{}'.format(i+1, test_x.size()[0]))
    model.eval()
    input = test_x[i].to(device)
    output = model.forward(input)

    _, top_class = torch.max(output, dim=1)

    remain_bufsize = test_x_raw.size(
    )[0] - total_cnt if total_cnt + batch_size > test_x_raw.size()[0] else batch_size
    total_cnt += remain_bufsize
    for cnt, index in enumerate(top_class[-remain_bufsize:]):
        fout.write('{} {}\n'.format(
            testing_info[i * batch_size + cnt], text_map[index.item()]))

fout.close()
print('process completed.')
