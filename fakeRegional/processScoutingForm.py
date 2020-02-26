import cv2
import numpy as np
import matplotlib as plt
import os
import csv
import copy
import sys

showImages = False

class ScoutingFormData:
    def __init__(self):
        self.team = ""
        self.match = ""
        self.autoStart = ""
        self.autoCross = ""
        self.autoLow = ""
        self.autoHighOuter = ""
        self.autoHighInner = ""
        self.teleopLow = ""
        self.teleopHigh = ""
        self.shootLocation = ""
        self.controlPanel2 = ""
        self.controlPanel3 = ""
        self.climb = ""
        self.foul = ""
        self.card = ""
        self.disabled = ""
        self.playedDefense = ""


def FormatBlankData(data):
    if not data:
        data = 0

    return data


def ResizeImg(img, heightDesired):
    height, width, channels = img.shape

    ratio = width / height
    widthDesired = heightDesired * ratio

    img = cv2.resize(img, (int(widthDesired), int(heightDesired)))
    return img

def CropToForm(img):
    # resize image to consistent size
    height = 2000
    img = ResizeImg(img, height)
    
    # crop image to form
    cropWidth = int(height * 0.16)
    cropHeight = int(height * 0.09)
    imgCrop = img[cropHeight:, cropWidth:]

    if showImages:
        cv2.imshow("imgCrop", imgCrop)

    return imgCrop


def FindBubbles(img):
    # fill in bubbles
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgEdge = cv2.Canny(img, 75, 200)

    contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    radThreshMin = 15
    radThreshMax = 30
    imgGrayFill = imgGray.copy()
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        if rad < radThreshMax and rad > radThreshMin:
            imgGrayFill = cv2.circle(imgGrayFill, (int(x), int(y)), int(rad), color=(0, 0, 0), thickness=-1, lineType=8, shift=0)

    # count bubbles
    imgThreshFill = cv2.threshold(imgGrayFill, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(imgThreshFill, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bubbleContours = []
    bubbleCount = 0
    height, width = imgThreshFill.shape
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        edgePercentThresh = 0.01
        if rad < radThreshMax and rad > radThreshMin and x > width * edgePercentThresh \
            and x < width * (1 - edgePercentThresh) and y > height * edgePercentThresh and y < height * (1 - edgePercentThresh):

            bubbleContours.append(c)
            bubbleCount += 1

    expectedBubbleCount = 219
    if bubbleCount != expectedBubbleCount:
        print("\033[91m" + "Error incorrect bubble count" + "\033[0m")
        return [], True

    imgBubbleHighlight = img.copy()
    cv2.drawContours(imgBubbleHighlight, bubbleContours, -1, (0, 0, 255), 3)

    if showImages:
        cv2.imshow("imgGrayFill", imgGrayFill)
        cv2.imshow("imgBubbleHighlight", imgBubbleHighlight)

    return bubbleContours, False


def ReadScoutingFormData(img, bubbleContours):
    bubbleY = []
    for c in bubbleContours:
        (x, y), rad = cv2.minEnclosingCircle(c)
        bubbleY.append(y)
    bubbleContours = [x for (y, x) in sorted(zip(bubbleY, bubbleContours), key=lambda pair: pair[0])]

    # find number of bubbles in each row
    heightDiffThresh = 30
    bubbleMatrix = []
    bubbleCount = 0
    (x, yOld), rad = cv2.minEnclosingCircle(bubbleContours[0])
    for c in bubbleContours:
        (x, y), rad = cv2.minEnclosingCircle(c)
        if (y - yOld) > heightDiffThresh:
            bubbleMatrix.append(bubbleCount)
            bubbleCount = 0
        bubbleCount += 1
        yOld = y
    bubbleMatrix.append(bubbleCount)

    expectedBubbleMatrix = [10, 10, 10, 10, 10, 10, 10, 3, 1, 13, 13, 13, 17, 17, 17, 17, 16, 3, 1, 1, 3, 10, 2, 1, 1]
    if bubbleMatrix != expectedBubbleMatrix:
        print("\033[91m" + "Error incorrect bubble matrix" + "\033[0m")
        return [], True

    # find row values
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    bubbleFillThreshPercent = 0.5
    bubbleMatrix2 = []
    totalBubbleCount = 0
    for c2 in bubbleMatrix:
        bubbleCount = 0
        rowValue = []

        bubbleX = []
        rowBubbleContours = bubbleContours[totalBubbleCount:totalBubbleCount + c2]
        for c in rowBubbleContours:
            (x, y), rad = cv2.minEnclosingCircle(c)
            bubbleX.append(x)
        rowBubbleContours = [x for (y, x) in sorted(zip(bubbleX, rowBubbleContours), key=lambda pair: pair[0])]

        for c in rowBubbleContours:
            bubbleCount += 1
            
            area = cv2.contourArea(c)

            mask = np.zeros(imgThresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
            fillPixels = cv2.countNonZero(mask)
            percentFill = (fillPixels / area)

            if percentFill > bubbleFillThreshPercent:
                rowValue = bubbleCount

        bubbleMatrix2.append(rowValue)
        totalBubbleCount += c2

    # convert bubble data to form data
    scoutingFormData = ScoutingFormData()
    if bubbleMatrix2[0] and bubbleMatrix2[1] and bubbleMatrix2[2] and bubbleMatrix2[3]:
        scoutingFormData.team = \
            (bubbleMatrix2[0] - 1) * 1000 \
            + (bubbleMatrix2[1] - 1) * 100 \
            + (bubbleMatrix2[2] - 1) * 10 \
            + (bubbleMatrix2[3] - 1)
    else:
        print("\033[91m" + "Error team not defined" + "\033[0m")
        return [], True

    if bubbleMatrix2[4] and bubbleMatrix2[5] and bubbleMatrix2[6]:
        scoutingFormData.match = \
            (bubbleMatrix2[4] - 1) * 100 \
            + (bubbleMatrix2[5] - 1) * 10 \
            + (bubbleMatrix2[6] - 1)
    else:
        print("\033[91m" + "Error match not defined" + "\033[0m")
        return [], True

    if bubbleMatrix2[7]:
        scoutingFormData.autoStart = bubbleMatrix2[7]
    else:
        print("\033[91m" + "Error auto start location not defined" + "\033[0m")
        return [], True

    scoutingFormData.autoCross = FormatBlankData(bubbleMatrix2[8])
    scoutingFormData.autoLow = FormatBlankData(bubbleMatrix2[9])
    scoutingFormData.autoHighOuter = FormatBlankData(bubbleMatrix2[10])
    scoutingFormData.autoHighInner = FormatBlankData(bubbleMatrix2[11])

    if bubbleMatrix2[13]:
        scoutingFormData.teleopLow = bubbleMatrix2[13]
    else:
        scoutingFormData.teleopLow = FormatBlankData(bubbleMatrix2[12])

    if bubbleMatrix2[16]:
        scoutingFormData.teleopHigh = bubbleMatrix2[16]
    elif bubbleMatrix2[15]:
        scoutingFormData.teleopHigh = bubbleMatrix2[15]
    else:
        scoutingFormData.teleopHigh = FormatBlankData(bubbleMatrix2[14])

    if bubbleMatrix2[17]:
        scoutingFormData.shootLocation = bubbleMatrix2[17]
    else:
        print("\033[91m" + "Error shoot location not defined" + "\033[0m")
        return [], True

    scoutingFormData.controlPanel2 = FormatBlankData(bubbleMatrix2[18])
    scoutingFormData.controlPanel3 = FormatBlankData(bubbleMatrix2[19])
    scoutingFormData.climb = FormatBlankData(bubbleMatrix2[20])
    scoutingFormData.foul = FormatBlankData(bubbleMatrix2[21])
    scoutingFormData.card = FormatBlankData(bubbleMatrix2[22])
    scoutingFormData.disabled = FormatBlankData(bubbleMatrix2[23])
    scoutingFormData.playedDefense = FormatBlankData(bubbleMatrix2[24])

    return scoutingFormData, False


def CreateOutputFileFromMatchSchedule(matchScheduleFilepath, outputFilepath):
    # read match schedule
    if not os.path.isfile(matchScheduleFilepath):
        print("\033[91m" + "Error failed to read match schedule" + "\033[0m")
        return
    
    # create output file if haven't already
    if os.path.isfile(outputFilepath):
        return
    
    print()
    print("\033[95m" + "Processing match schedule..." + "\033[0m")
    
    match = ScoutingFormData()
    matchList = []
    
    with open(matchScheduleFilepath, 'r',  newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")

        # skip headers
        next(csvReader)

        # loop through rows and read matches
        for row in csvReader:
            match.match = int(row[0].replace("Qualification ",""))
            match.team = int(row[2]) # red 1
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[3]) # red 2
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[4]) # red 3
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[5]) # blue 1
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[6]) # blue 2
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[7]) # blue 3
            matchList.append(copy.deepcopy(match))

    with open(outputFilepath, "w",  newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        
        # write headers
        csvWriter.writerow([
            "Match",
            "Team",
            "Auto Start Location",
            "Auto Cross",
            "Auto Low Goals",
            "Auto High Outer Goals",
            "Auto High Inner Goals",
            "Teleop Low Goals",
            "Teleop High Goals",
            "Shoot Location",
            "Control Panel 2",
            "Control Panel 3",
            "Climb",
            "Foul",
            "Card",
            "Disabled",
            "Played Defense",
        ])
        
        # write matches
        for match in matchList:
            csvWriter.writerow([
                match.match,
                match.team,
                match.autoStart,
                match.autoCross,
                match.autoLow,
                match.autoHighOuter,
                match.autoHighInner,
                match.teleopLow,
                match.teleopHigh,
                match.shootLocation,
                match.controlPanel2,
                match.controlPanel3,
                match.climb,
                match.foul,
                match.card,
                match.disabled,
                match.playedDefense,
            ])

    print("\033[92m" + "Processed match schedule" + "\033[0m")
    return


def WriteScoutingFormDataToOutputFile(scoutingFormData, outputFilepath):
    with open(outputFilepath, "r",  newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        tempData = []
        
        # loop through rows and find match
        matchFound = False
        rowCnt = 0
        for row in csvReader:
            # skip headers
            if not rowCnt == 0:
                # write form data to output file if match found
                if (int(row[0]) == scoutingFormData.match) and (int(row[1]) == scoutingFormData.team):
                    matchFound = True
                    tempRow = row
                    tempRowCnt = rowCnt
                    tempRow[0] = scoutingFormData.match
                    tempRow[1] = scoutingFormData.team
                    tempRow[2] = scoutingFormData.autoStart
                    tempRow[3] = scoutingFormData.autoCross
                    tempRow[4] = scoutingFormData.autoLow
                    tempRow[5] = scoutingFormData.autoHighOuter
                    tempRow[6] = scoutingFormData.autoHighInner
                    tempRow[7] = scoutingFormData.teleopLow
                    tempRow[8] = scoutingFormData.teleopHigh
                    tempRow[9] = scoutingFormData.shootLocation
                    tempRow[10] = scoutingFormData.controlPanel2
                    tempRow[11] = scoutingFormData.controlPanel3
                    tempRow[12] = scoutingFormData.climb
                    tempRow[13] = scoutingFormData.foul
                    tempRow[14] = scoutingFormData.card
                    tempRow[15] = scoutingFormData.disabled
                    tempRow[16] = scoutingFormData.playedDefense

            tempData.append(row)
            rowCnt += 1

    if not matchFound:
        print("\033[91m" + "Error match not found in match schedule" + "\033[0m")
        return True

    with open(outputFilepath, "w",  newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        tempData[tempRowCnt] = tempRow
        csvWriter.writerows(tempData)

    return False


if __name__== "__main__":
    workDir = os.getcwd()

    # read match schedule and create output file
    matchScheduleFilename = "matchSchedule.csv"
    matchScheduleFilepath = os.path.join(workDir, matchScheduleFilename)
    
    outputFilename = "Raw Form Data.csv"
    outputFilepath = os.path.join(workDir, outputFilename)
    
    CreateOutputFileFromMatchSchedule(matchScheduleFilepath, outputFilepath)
    
    # loop through unprocessed images
    unprocessedDirName = os.path.join(workDir, "unprocessedForms")
    processedDirName = os.path.join(workDir, "processedForms")
    unprocessedDir = os.fsencode(unprocessedDirName)
    for file in os.listdir(unprocessedDir):
        unprocessedFilename = os.fsdecode(file)
        unprocessedFilepath = os.path.join(unprocessedDirName, unprocessedFilename)
        if unprocessedFilename.endswith(".jpg") or unprocessedFilename.endswith(".jpeg") or unprocessedFilename.endswith(".JPG"):
            print()
            print("\033[95m" + "Processing " + unprocessedFilename + "..." + "\033[0m")
            
            img = cv2.imread(unprocessedFilepath, cv2.IMREAD_COLOR)
            if img is None:
                print("\033[91m" + "Error failed to read image" + "\033[0m")
                continue

            if showImages:
                cv2.imshow("imgThresh", img)

            imgCrop = CropToForm(img)

            bubbleContours, isError = FindBubbles(imgCrop)
            if isError:
                continue

            scoutingFormData, isError = ReadScoutingFormData(imgCrop, bubbleContours)
            if isError:
                continue
                
            isError = WriteScoutingFormDataToOutputFile(scoutingFormData, outputFilepath)
            if isError:
                continue
            
            # move image to processed
            processedFilename = str(scoutingFormData.match) + "_" + str(scoutingFormData.team) + ".jpg"
            processedFilepath = os.path.join(processedDirName, processedFilename)
            if os.path.isfile(processedFilepath):
                os.remove(processedFilepath)
            os.rename(unprocessedFilepath, processedFilepath)
            
            print("\033[92m" + "Processed " + processedFilename + "\033[0m")
            
        else:
            continue
            
        sys.stdout.flush()

    if showImages:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
