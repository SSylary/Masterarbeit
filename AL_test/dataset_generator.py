from PIL import Image, ImageDraw
import csv
import numpy as np
import re
import os.path
import cv2
import random
import sys

import math
import time
import scipy
import matplotlib.pyplot as plt


class Block:
    def __init__(self, label):
        self.label = label
        self.color = label.split('_')[0]
        self.size = label.split('_')[1]
        if self.size == 's':
            self.height, self.width, self.depth = (60, 30, 30)
        elif self.size == 'm':
            self.height, self.width, self.depth = (60, 60, 30)
        elif self.size == 'l':
            self.height, self.width, self.depth = (120, 60, 30)
        self.dimensions = (self.height, self.width, self.depth)
        self.possible_crops = []
        self.choosenCrop = None

    # convert Polygon coordinate (str) into list of int tuples
    def str_to_tuple_list(self, stringList):
        stringList = re.sub('\),\(', ') (', stringList)
        stringList = re.sub('\(', '', stringList)
        stringList = re.sub('\)', '', stringList)
        stringList = re.sub(',', ' ', stringList)
        coordinateList = [float(elem) for elem in stringList.split(' ')]
        it = iter(coordinateList)
        coordinateList = zip(it, it)
        return list(coordinateList)

    # calculate center of Bounding Box (coordinates in original image)
    def getBboxCenter(self, bbox):
        center_x = (bbox[0][0] + bbox[1][0]) / 2
        center_y = (bbox[0][1] + bbox[1][1]) / 2
        return (center_x, center_y)

    # define all possible Crops for each block (center of bounding box + polygone (mask) coordinates)
    def defineAllPossibleCrops(self, polygon_filename):
        with open(polygon_filename, 'r') as f:
            data = csv.reader(f)
            next(data)  # ignore first line
            for row in data:
                if self.label == row[1]:
                    filename = row[0]
                    polygonCoorList = self.str_to_tuple_list(row[2])
                    bbox = self.getBoundingbox(polygonCoorList)
                    crop = self.getCrop(filename, bbox, polygonCoorList)
                    self.possible_crops.append({'filename': filename, 'polygonCoordinates': polygonCoorList,
                                                'bboxCenter': self.getBboxCenter(bbox), 'bbox': bbox, 'crop': crop})

    def getCrop(self, filename, bbox, polygon):
        im = Image.open(os.path.join(
            'pipenv_project/Mask_RCNN/AL_test/dataset_generator/labeled_images', filename)).convert('RGBA')
        imageArray = np.array(im)
        # create black background with same size as original image
        background = Image.new(
            'L', (imageArray.shape[1], imageArray.shape[0]), 0)
        # create an object that can be used to draw in the given image (background).
        draw = ImageDraw.Draw(background)
        draw.polygon(polygon, outline=255, fill=255)
        mask = np.array(background)
        # assemble new image
        newImageArray = np.empty(imageArray.shape, dtype='uint8')
        # colors (three first columns, RGB)
        newImageArray[:, :, :3] = imageArray[:, :, :3]
        # transparency (4th column)
        newImageArray[:, :, 3] = mask
        # back to Image from numpy
        newImage = Image.fromarray(newImageArray, "RGBA")
        crop = newImage.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
        return crop

    # get Bbox form polygon, (x_min, y_min)
    # rounded down / (x_max, y_max) rounded up
    def getBoundingbox(self, polygonCoorList):
        lowerLeft = (math.floor(min([elem[0] for elem in polygonCoorList])), math.floor(
            min([elem[1] for elem in polygonCoorList])))
        upperRight = (math.ceil(max([elem[0] for elem in polygonCoorList])), math.ceil(
            max([elem[1] for elem in polygonCoorList])))
        return [lowerLeft, upperRight]

    # choose a random Crop from the available ones 
    def chooseRandomCrop(self):
        rID = random.randrange(0, len(self.possible_crops))
        self.choosenCrop = self.possible_crops[rID]


class Histogram():
    def __init__(self, classes):
        self.labelsAppearance = []
        self.maxLabelSet = set()
        for cl in classes:
            self.labelsAppearance.append({'label': cl, 'numOfImages': 0, 'numOfObj': 0})

    def getMaxLabel(self):
        maxObjectNum = max([elem['numOfObj'] for elem in self.labelsAppearance])
        print('\nMAX OBJ: ' + str(maxObjectNum))
        for elem in self.labelsAppearance:
            if elem['numOfObj'] == maxObjectNum:
                self.maxLabelSet.add(elem['label'])
        # print(maxLabelList)
        return self.maxLabelSet, maxObjectNum

    # check if the number of generated objects is the same for all labels
    def haveSameNum(self):
        maxNum = self.getMaxLabel()[1]
        for elem in self.labelsAppearance:
            if elem['numOfObj'] != maxNum:
                return False
        return True

    def countImages(self, cropsToPaste):
        labelsToPaste = [crop['label'] for crop in cropsToPaste]
        for label in labelsToPaste:
            for elem in self.labelsAppearance:
                if elem['label'] == label:
                    elem['numOfObj'] += 1
        # remove duplicated strings to increment num of images that contains the label
        for label in list(set(labelsToPaste)):
            for elem in self.labelsAppearance:
                if elem['label'] == label:
                    elem['numOfImages'] += 1

    def plot(self, filename):
        # print('COUNTING: ' + str(self.labelsAppearance))
        labels = [elem['label'] for elem in self.labelsAppearance]
        numsOfObj = [elem['numOfObj'] for elem in self.labelsAppearance]
        numsOfImages = [elem['numOfImages'] for elem in self.labelsAppearance]
        plt.subplot(121)
        plt.bar(labels, numsOfObj, align='center')
        plt.xticks(rotation=90)
        plt.ylabel('Number of Objects')

        plt.subplot(122)
        plt.bar(labels, numsOfImages)
        plt.xticks(rotation=90)
        plt.ylabel('Number of images')

        plt.savefig(filename)


class CsvFile():

    def __init__(self, filename):
        self.filename = filename
        with open(filename + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label', 'polygon'])
            # writer.writerow(['frame', 'xmin', 'xmax','ymin','ymax', 'class_id', 'label'])
            f.close()

    def parseDict(self, cropsToPaste, imFilename):
        crops = []
        labels_to_id = {'orange_s': '13', 'orange_m': '14', 'orange_l': '15', 'red_s': '1', 'red_m': '2', 'red_l': '3',
                        'yellow_s': '4',
                        'yellow_m': '5', 'yellow_l': '6', 'green_s': '7', 'green_m': '8', 'green_l': '9',
                        'blue_s': '10', 'blue_m': '11', 'blue_l': '12'}
        for crop in cropsToPaste:
            crops.append([imFilename, crop['label'], crop['newPolygon']])
            # crops.append([imFilename + '.png',  crop['newBbox'][0][0], crop['newBbox'][1][0], crop['newBbox'][0][1], crop['newBbox'][1][1], labels_to_id[crop['label']], crop['label']])
        return crops

    def write(self, cropsToPaste, imFilename):
        crops = self.parseDict(cropsToPaste, imFilename)
        for row in crops:
            with open(self.filename + '.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.close()


class NewImage:
    IMAGE_HEIGHT = 2054  # pixels
    IMAGE_WIDTH = 2456
    count = 20
    csvFile = CsvFile('pipenv_project/Mask_RCNN/AL_test/dataset_generator/test_labels')  #

    def __init__(self, choosenBlocks):
        # -----TODO----- background need change
        fileList = ['bg1.jpg', 'bg2.png', 'bg3.png']
        # self.background = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
        bgId = fileList[random.choice([0,1,2])]
        im = Image.open(os.path.join('pipenv_project/Mask_RCNN/AL_test/dataset_generator/labeled_images/background', bgId)).convert('RGBA')
        self.background = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        self.crops = []
        # self.choosenBlocks = choosenBlocks
        for block in choosenBlocks:
            self.crops.append(
                {'label': block.label, 'im': block.choosenCrop['crop'],
                 'polygonCoordinates': block.choosenCrop['polygonCoordinates'],
                 'bboxCenter': block.choosenCrop['bboxCenter'], 'bbox': block.choosenCrop['bbox']})
        self.filename = str(self.count)
        NewImage.count += 1

    # choose random angle of rotation
    def randomRotate(self, crop):
        i = random.randint(0, 36)
        angle = 10 * i
        rotatedCrop = crop['im'].rotate(angle, expand=True)
        # rotatedCrop.show()
        return rotatedCrop, angle

    # rotate old polygon coordinates with the same angle used for crop rotation
    def rotatePolygon(self, points, center, angle):
        angle = - angle * scipy.pi / 180  # in radian
        return scipy.dot(scipy.array(points) - list(center), scipy.array([[scipy.cos(angle), scipy.sin(angle)], [-scipy.sin(angle), scipy.cos(angle)]])) + center

    # check if two bounding box are overlaping (after rotation)
    def doOverlap(self, crop1, crop2):
        # if one above the other
        if (crop1['y_max'] < crop2['y_min']) or (crop2['y_max'] < crop1['y_min']):
            return False
        # If one rectangle is on left side of other
        if (crop1['x_min'] > crop2['x_max']) or (crop2['x_min'] > crop1['x_max']):
            return False
        return True

    # check if each new  crop is overlapping with others
    def doMultipleOverlap(self, newCrop, cropsList):
        for crop in cropsList:
            if self.doOverlap(newCrop, crop):
                return True
            else:
                continue
        return False

    def defNewPolygCoordinate(self, rotatedPolygonCoor, tmpCrop1):
        newPolygonCoordinate = []
        for point in rotatedPolygonCoor:
            newPolygonCoordinate.append((float("{0:.2f}".format(point[0] + tmpCrop1['x_trans'])),
                                         float("{0:.2f}".format(point[1] + tmpCrop1['y_trans']))))
        # print('\n NEW POLYGON TRANSLATED: \n' + str(newPolygonCoordinate ))
        return newPolygonCoordinate

    # find appropriate random positions for the choosen blocks and save image
    def adaptCrop(self):
        self.cropsToPaste = []
        freeRegion1 = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH))
        # add first element in random position
        print('first crop')
        tmpCrop = {}
        tmpCrop['label'], [tmpCrop['im'], tmpCrop['angle']] = self.crops[0]['label'], self.randomRotate(self.crops[0])
        rotatedPolygonCoor = self.rotatePolygon(self.crops[0]['polygonCoordinates'], self.crops[0]['bboxCenter'], tmpCrop['angle'])
        [upperLeft, lowerRight] = Block.getBoundingbox(self, rotatedPolygonCoor)
        print('upperLeft', upperLeft)
        bboxSizeW = lowerRight[0] - upperLeft[0]
        bboxSizeH = lowerRight[1] - upperLeft[1]
        centerX = (lowerRight[0] + upperLeft[0]) // 2
        centerY = (lowerRight[1] + upperLeft[1]) // 2
        tmpCrop['oldCenter'] = [centerX, centerY]

        freeRegion1[:, IMAGE_WIDTH - bboxSizeW:] = 0
        freeRegion1[IMAGE_HEIGHT - bboxSizeH:, :] = 0
        tmpCrop.update(self.randomTranslate(tmpCrop, freeRegion1, bboxSizeW, bboxSizeH))
        tmpCrop['newPolygon'] = self.defNewPolygCoordinate(rotatedPolygonCoor, tmpCrop)
        tmpCrop['newBbox'] = Block.getBoundingbox(self, tmpCrop['newPolygon'])
        print('newbbox', tmpCrop['newBbox'])
        
        self.cropsToPaste.append(tmpCrop)
        print('---> first crop done')

        # freeRegion[tmpCrop['x_min']:tmpCrop['x_max'] +1, tmpCrop['y_min']:tmpCrop['y_max']+1] = 0
        # check overlapping in case of multiple blocks
        if len(self.crops) != 1:
            for crop in self.crops[1:]:
                flag = 1
                while flag:
                    freeRegion = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH))
                    tmpCrop = {}
                    tmpCrop['label'], [tmpCrop['im'], tmpCrop['angle']] = crop['label'],  self.randomRotate(crop)
                    rotatedPolygonCoor = self.rotatePolygon(crop['polygonCoordinates'], crop['bboxCenter'], tmpCrop['angle'])
                    # calculate the new bounding box after rotation
                    # print('old center', crop['bboxCenter'])
                    [upper_left, lower_right] = Block.getBoundingbox(self, rotatedPolygonCoor)
                    bboxW = lower_right[0] - upper_left[0]
                    bboxH = lower_right[1] - upper_left[1]
                    centerX = (lower_right[0] + upper_left[0]) // 2
                    centerY = (lower_right[1] + upper_left[1]) // 2
                    tmpCrop['oldCenter'] = [centerX, centerY]
                    freeRegion[:, IMAGE_WIDTH - bboxW:] = 0
                    freeRegion[IMAGE_HEIGHT - bboxH:, :] = 0
                    for existCrop in self.cropsToPaste:
                        upperPoint, lowerPoint = existCrop['newBbox']
                        shiftX = upperPoint[0] - bboxW if upperPoint[0] - bboxW > 0 else 0
                        shiftY = upperPoint[1] - bboxH if upperPoint[1] - bboxH > 0 else 0
                        # print('shiftX & Y', shiftX, shiftY)
                        freeRegion[shiftY:lowerPoint[1], shiftX:lowerPoint[0]] = 0

                    print(np.sum(freeRegion))
                    if np.sum(freeRegion) != 0:
                        flag = 0
                        tmpCrop.update(self.randomTranslate(tmpCrop, freeRegion, bboxW, bboxH))
                        tmpCrop['newPolygon'] = self.defNewPolygCoordinate(rotatedPolygonCoor, tmpCrop)
                        tmpCrop['newBbox'] = Block.getBoundingbox(self, tmpCrop['newPolygon'])
                        self.cropsToPaste.append(tmpCrop)
                    else:
                        flag += 1
                    '''
                    try different positions till getting the right one (100 tries)
                    If after 100 tries no appropriate position has been found
                    retry all crops again
                    '''
                    if flag > 100:
                        return NewImage.adaptCrop(self)

        print('\n\nFINAL CROPS TO PASTE' + str(self.cropsToPaste) + '\n\n')
        print('\n\n\n NEW IMAGE \n\n\n')
        # write labels to csv file
        NewImage.csvFile.write(self.cropsToPaste, self.filename)
        # paste defined crops and save image
        self.pasteCrops()
        # draw Boxes and masks to verify
        # self.drawBoxes(self.filename,self.cropsToPaste)
        return self.cropsToPaste

    # draw new bbox and transformated polygons to validate the labeling
    def drawBoxes(self, imFilename, cropsToPaste):
        image = cv2.imread('test/' + imFilename + '.png')
        for block in cropsToPaste:
            cv2.rectangle(image, (block['newBbox'][0][0], block['newBbox'][1][1]),
                          (block['newBbox'][1][0], block['newBbox'][0][1]), color=(0, 0, 255), thickness=2)
            poly = np.array([list(elem) for elem in block['newPolygon']], np.int32)
            cv2.polylines(image, [poly], True, (255, 255, 255), thickness=2)
            cv2.putText(image, block['label'], (block['newBbox'][0][0], block['newBbox'][1][1]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imwrite('labeled_images/' + imFilename + '_Bbox.png', image)

    # paste the randomly choosen Crop (rotated and translated) on a black background and save image
    def pasteCrops(self):
        for crop in self.cropsToPaste:
            # self.background.paste(crop['im'].convert('RGBA'), (crop['x_min'], crop['y_min']))
            imgCenterX , imgCenterY = crop['oldCenter'][0] + crop['x_trans'] ,crop['oldCenter'][1] + crop['y_trans']
            posX = imgCenterX - (crop['im'].size[0] //2)
            posY = imgCenterY - (crop['im'].size[1] //2)
            self.background.paste(crop['im'], mask=crop['im'].split()[3], box=(posX, posY)) # crop['newBbox'][0]
        self.saveImage()

    # Define random translation and make sure the crop is completely inside the image
    def randomTranslate(self, crop, freeRegion, bbx, bby):

        tmpCrop = {}
        tmpCrop['y_min'], tmpCrop['x_min'] = random.choice(np.transpose(np.nonzero(freeRegion)))
        tmpCrop['x_max'], tmpCrop['y_max'] = tmpCrop['x_min'] + bbx, tmpCrop['y_min'] + bby
        tmpCrop['newCenter'] = ((tmpCrop['x_min'] + tmpCrop['x_max']) // 2, (tmpCrop['y_min'] + tmpCrop['y_max']) // 2)
        tmpCrop['x_trans'], tmpCrop['y_trans'] = tmpCrop['newCenter'][0] - crop['oldCenter'][0], tmpCrop['newCenter'][1] - crop['oldCenter'][1]

        return tmpCrop


    def saveImage(self):
        # self.background.show()
        self.background.save(os.path.join('pipenv_project/Mask_RCNN/AL_test/dataset_generator/labeled_images/test', self.filename + '.png'))


# choose random blocks from the list
def chooseRandomBlocks(Blocklist, labels_to_id):
    # 5 is max number of blocks to have on a picture
    # To keep the instance will be used out in certain loopsC
    # Instances in each image is 3-5
    choose_blocks = []
    global countImgList
    global countClass
    notZeroRdList = filter((1).__le__, countImgList)
    if not notZeroRdList:
        return
    i = random.choice(list(notZeroRdList))
    idx = countImgList.index(i)
    countImgList[idx] -= 1
    rdNums = idx + 3
    print('Numbers of choosen Blocks = ' + str(rdNums))
    while rdNums:
        temp = random.choice(Blocklist)
        blockId = int(labels_to_id[temp.label]) - 1
        if countClass[blockId] != 0:
            choose_blocks.append(temp)
            countClass[blockId] -= 1
            rdNums -= 1
        else:
            continue
    return choose_blocks
    # return random.sample(Blocklist, k=rdIdx) # py3.5 choice


# save image with single crop
def save_single_crop(blocklist):
    for block in blocklist:
        for possible_crop in block.possible_crops:
            filename = possible_crop['filename']
            bbox = possible_crop['bbox']
            im = Image.open(os.path.join(
                'labeled_images', filename)).convert('RGBA')
            imageArray = np.array(im)
            polygon = possible_crop['polygonCoordinates']
            # create black background with same size as original image
            background = Image.new(
                'L', (imageArray.shape[1], imageArray.shape[0]), 255)
            # create an object that can be used to draw in the given image (background).
            draw = ImageDraw.Draw(background)
            draw.polygon(polygon, outline=1, fill=1)
            # background.show()
            mask = np.array(background)
            # assemble new image
            newImageArray = np.empty(imageArray.shape, dtype='uint8')
            # colors (three first columns, RGB)
            newImageArray[:, :, :3] = imageArray[:, :, :3]
            # transparency (4th column)
            newImageArray[:, :, 3] = mask * 255
            # back to Image from numpy
            newImage = Image.fromarray(newImageArray, "RGBA")
            newImageCropped = newImage.crop(
                (math.floor(bbox[0][0]), math.floor(bbox[0][1]), math.ceil(bbox[1][0]), math.ceil(bbox[1][1])))
            # newImageCropped.show()/fzi/ids/fn875/no_backup/xzc
            newImageCropped.save(os.path.join('labeled_images', os.path.splitext(possible_crop['filename'])[
                0] + '_' + block.label + '_' + str(block.possible_crops.index(possible_crop)) + '.png'))


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    IMAGE_HEIGHT = 2054  # pixels
    IMAGE_WIDTH = 2456
    polygon_filename = os.path.abspath('pipenv_project/Mask_RCNN/AL_test/dataset_generator/Polygones_labels.csv')

    labels = ['orange_l', 'blue_l', 'red_l', 'blue_s', 'orange_m', 'red_m', 'green_s',
              'orange_s', 'yellow_l', 'green_l', 'green_m', 'blue_m', 'yellow_s', 'red_s', 'yellow_m']
    labels_to_id = {'red_s': '1', 'red_m': '2', 'red_l': '3', 'yellow_s': '4', 'yellow_m': '5', 'yellow_l': '6',
                    'green_s': '7', 'green_m': '8', 'green_l': '9', 'blue_s': '10', 'blue_m': '11', 'blue_l': '12',
                    'orange_s': '13', 'orange_m': '14', 'orange_l': '15'}
    # define histogram
    histo = Histogram(labels)

    # create possible block objects
    blocklist = [Block(label) for label in labels]
    # define possible crops for each block
    for block in blocklist:
        block.defineAllPossibleCrops(polygon_filename)

    # Set the number of generated images
    numOfNewImage = 90
    averageNumOfObj = 4
    countImgList = [numOfNewImage//3 for _ in range(3)]
    # Set the number of objects of each class
    # Average instances in each image is (3+4+5)/3 = 4
    countClass = [numOfNewImage * averageNumOfObj // len(labels) for _ in range(15)]
    for i in range(numOfNewImage):
        choosenBlocks = chooseRandomBlocks(blocklist, labels_to_id)

        print('labels of Choosen Blocks = ' + str([block.label for block in choosenBlocks]) + '\n')
        for block in choosenBlocks:
            block.chooseRandomCrop()

        newImage = NewImage(choosenBlocks)
        cropsToPaste = newImage.adaptCrop()
        histo.countImages(cropsToPaste)

    histo.plot('stats.png')

'''
    # let number of object the same for all labels
    while not histo.haveSameNum():
        labelToNotGen = histo.getMaxLabel()[0]

        print('\nLABELS TO NOT GENERATE: \n ' + str(labelToNotGen) + '\n')

        newBlocklist = []
        for block in blocklist:
            if not block.label in labelToNotGen:
                newBlocklist.append(block)
        # avoid two labels on the same frames (because it could increase the max num of generated labels)
        blockSet = set(chooseRandomBlocks(newBlocklist))
        print('\nNEW CHOOSEN CROP \n ' + str([block.label for block in blockSet]) + '\n')
        choosenBlocks = list(blockSet)
        print('\nNEW CHOOSEN CROP_SET \n ' + str([block.label for block in choosenBlocks]) + '\n')
        for block in choosenBlocks:
            block.chooseRandomCrop()

        newImage = NewImage(choosenBlocks)
        cropsToPaste = newImage.adaptCrop()
        histo.countImages(cropsToPaste)

    histo.plot('uniform.png')
'''