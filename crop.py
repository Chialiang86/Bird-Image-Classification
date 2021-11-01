import cv2
import argparse


class Cropper():
    pos = []
    cnt = 0

    def __get_click_pose(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cnt += 1
            self.pos.append((x, y))

    def processing_train(self, x_raw, fnames, index_map, dest, start=0):
        end_flag = False
        itr = 0
        length = len(x_raw)
        for i in range(15):
            for cls in index_map:
                if itr + 1 < start:
                    itr += 1
                else:
                    print('processing {}/{} -> {}'.format(itr +
                          1, length, fnames[index_map[cls][i]]))
                    self.pos = []
                    cnt = 0
                    img = x_raw[index_map[cls][i]]
                    cv2.imshow('image {}'.format(
                        fnames[index_map[cls][i]]), img)
                    cv2.setMouseCallback('image {}'.format(
                        fnames[index_map[cls][i]]), self.__get_click_pose)
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(self.pos)

                    crop = img[self.pos[0][1]:self.pos[1]
                               [1], self.pos[0][0]:self.pos[1][0]]
                    cv2.imwrite(
                        '{}/{}'.format(dest, fnames[index_map[cls][i]]), crop)
                    itr += 1

                    if key == ord('q'):
                        end_flag = True
                        break

            if end_flag == True:
                break

    def processing_test(self, x_raw, fnames, dest, start=0):
        end_flag = False
        itr = 0
        length = len(x_raw)
        for i in range(length):
            if itr + 1 < start:
                itr += 1
            else:
                print(
                    'processing {}/{} -> {}'.format(itr+1, length, fnames[i]))
                self.pos = []
                cnt = 0
                img = x_raw[i]
                cv2.imshow('image {}'.format(fnames[i]), img)
                cv2.setMouseCallback('image {}'.format(
                    fnames[i]), self.__get_click_pose)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(self.pos)

                crop = img[self.pos[0][1]:self.pos[1]
                           [1], self.pos[0][0]:self.pos[1][0]]
                cv2.imwrite('{}/{}'.format(dest, fnames[i]), crop)
                itr += 1

                if key == ord('q'):
                    end_flag = True
                    break

            if end_flag == True:
                break


def create_index_map(y_raw):
    index_map = {}
    for i in range(200):
        index_map[i] = []
    for index, label in enumerate(y_raw):
        index_map[label].append(index)
    return index_map


def load_raw_data(info_path, img_path, class_map):
    # get raw info
    f_info = open(info_path, 'r')
    raw_info = [str(c).strip('\n') for c in f_info.readlines()]

    # preprocessing data
    data_size = len(raw_info)
    x_raw, y_raw, fnames = [], [], []
    for i in range(data_size):
        img = cv2.imread('{}{}'.format(img_path, raw_info[i].split()[0]))
        cls = class_map[raw_info[i].split()[1]]
        x_raw.append(img)
        y_raw.append(cls)
        fnames.append(raw_info[i].split()[0])

    return x_raw, y_raw, fnames


def main(args):
    pos = []
    class_map = {}
    class_map = {}

    f_class = open('data/classes.txt', 'r')
    class_list = [str(c).strip() for c in f_class.readlines()]
    for cls in class_list:
        class_map[cls] = int(cls.split('.')[0]) - 1  # 0 ~ 199

    print('loading data ...')
    x_train, y_train, fnames_train = load_raw_data(
        info_path='data/training_labels.txt', img_path='data/training_images/', class_map=class_map)
    index_map_train = create_index_map(y_train)

    f_testing_info = open('data/testing_img_order.txt', 'r')
    test_img_path = 'data/testing_images/'
    testing_info = [str(c).strip() for c in f_testing_info.readlines()]
    x_test = [cv2.imread('{}{}'.format(test_img_path, info))
              for info in testing_info]

    cropper = Cropper()
    dest_train = 'crop/training_images'
    dest_test = 'crop/testing_images'
    print('processing ...')
    cropper.processing_train(x_train, fnames_train,
                             index_map_train, dest_train, args.trains)
    cropper.processing_test(x_test, testing_info, dest_test, args.tests)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cropper')
    parser.add_argument('-trains', default=0, type=int)
    parser.add_argument('-tests', default=0, type=int)
    args = parser.parse_args()

    main(args)
