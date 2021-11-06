import os

# data split
import random
import shutil

from tqdm import tqdm


def data_split(input_dir) :
    # 입력 받은 데이터를 섞은 후 비율을 맞춰
    all_data = []
    img_data = []
    label_data = []
    for (path, dir, files) in os.walk(input_dir):
            for file in files:
                temp = file.split(".")
                if temp[-1] == "txt":
                    label_data.append(path + "\\" + file)
                else:
                    img_data.append(path + "\\" + file)
    # for (i, k) in zip(img_data, label_data):
    #     all_data.append([i, k])
    # 모든 데이터의 경로
    data_path = img_data
    # 이들을 비율로 맞추어서 test, train, val 데이터로 분리할 것
    # 한개의 배열을 섞는거라서, 그냥 shuffle 사용
    # test_split_data에서는 골고루 나오게 랜덤으로 섞을 수도 있는데,
    # 한개의 배열을 섞는데는 적절하지 않아 보였음, (편법으로 할수도있는데, 일단 이렇게 ㄱ)
    random.shuffle(data_path)
    print(len(data_path))
    # index
    # 80퍼센트는 train 데이터
    train_index = int(len(data_path)*0.8)

    # val_index = int(len(data_path)*0.8)
    # 나머지 20퍼센트는 Val & Test 데이터
    train_data_path = data_path[:train_index]

    # val_data_path = data_path[train_index:val_index]
    test_data_path = data_path[train_index:]
    # # temp = []
    # # for i, k in test_data_path:
    # #     temp.append(i)
    # #     temp.append(k)
    # test_data_path = temp

    # 랜덤하게 섞인 녀석들을 각각 분리시켜서 이동
    print("train data move start..")
    print("#"*20)
    # 절대 경로로 해야되는듯
    # input으로 지정받은 폴더에 train folder 만들고
    absPath = os.path.abspath(input_dir)
    imagePath = absPath + "/images"
    labelPath = absPath + "/labels"
    imageTrainPath = absPath + "/images" + "/train"
    imageValPath = absPath + "/images" + "/val"
    labelTrainPath = absPath + "/labels" + "/train"
    labelValPath = absPath + "/labels" + "/val"
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)
        if not os.path.exists(imageTrainPath):
            os.mkdir(imageTrainPath)
    if not os.path.exists(labelPath):
        os.mkdir(labelPath)
        if not os.path.exists(labelTrainPath):
            os.mkdir(labelTrainPath)

    for imagePath in tqdm(train_data_path):
        # 파일명 추출
        imageFileName = os.path.basename(imagePath)
        labelFileName = imageFileName.split(".")[0] + ".txt"
        labelFilePath = os.path.dirname(imagePath) + "/"+ labelFileName

        shutil.move(imagePath, imageTrainPath + "\\"+imageFileName)
        shutil.move(labelFilePath, labelTrainPath + "\\" + labelFileName)


    print("#" * 20)
    # 반복작업 싫긴한데.. 간단하게 만드는것도 귀찮으므로 복붙
    print("test data move start..")
    print("#" * 20)
    # 절대 경로로 해야되는듯
    # input으로 지정받은 폴더에 train folder 만들고
    absPath = os.path.abspath(input_dir)
    imagePath = absPath + "/images"
    labelPath = absPath + "/labels"
    imageTrainPath = absPath + "/images" + "/train"
    imageValPath = absPath + "/images" + "/val"
    labelTrainPath = absPath + "/labels" + "/train"
    labelValPath = absPath + "/labels" + "/val"
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)
        if not os.path.exists(imageValPath):
            os.mkdir(imageValPath)
    if not os.path.exists(labelPath):
        os.mkdir(labelPath)
        if not os.path.exists(labelValPath):
            os.mkdir(labelValPath)

    for imagePath in tqdm(test_data_path):
        # 파일명 추출
        imageFileName = os.path.basename(imagePath)
        labelFileName = imageFileName.split(".")[0] + ".txt"
        labelFilePath = os.path.dirname(imagePath) + "/" + labelFileName

        shutil.move(imagePath, imageValPath + "\\" + imageFileName)
        shutil.move(labelFilePath, labelValPath + "\\" + labelFileName)
    print("#" * 20)



data_split("E:/test")