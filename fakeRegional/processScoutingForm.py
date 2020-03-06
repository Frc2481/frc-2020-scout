import cv2
import numpy as np
import matplotlib as plt
import os
import csv
import copy
import sys

showImages = True
height = 1000
bubbleRad = height * 0.009

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
        self.teleopHighOuter = ""
        self.teleopHighInner = ""
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
    imgHeight, imgWidth = img.shape

    ratio = imgWidth / imgHeight
    widthDesired = heightDesired * ratio

    img = cv2.resize(img, (int(widthDesired), int(heightDesired)))
    return img

def CropToForm(img):
    # crop image to form
    cropWidthMin = int(height * 8.5 / 11 * 0.15)
    cropWidthMax = int(height * 8.5 / 11 * 0.95)
    cropHeightMin = int(height * 0.05)
    cropHeightMax = int(height * 0.95)
    imgCrop = img[cropHeightMin:cropHeightMax, cropWidthMin:cropWidthMax]

    if showImages:
        cv2.imshow("imgCrop", imgCrop)

    return imgCrop

def FindBubbles(img):
    # set detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = bubbleRad * 2 * 0.8
    params.minThreshold = 127
    params.maxThreshold = 255

    params.filterByArea = True
    params.minArea = np.pi * (bubbleRad * 0.7) ** 2
    params.maxArea = np.pi * (bubbleRad * 2) ** 2

    params.filterByCircularity = False
    params.minCircularity = 0
    params.maxCircularity = 1

    params.filterByColor = False
    params.blobColor = 0

    params.filterByConvexity = False
    params.minConvexity = 0
    params.maxConvexity = 1

    params.filterByInertia = True
    params.minInertiaRatio = 0.7
    params.maxInertiaRatio = 1
    
    # find bubbles
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    imgBubbles = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # sort bubbles found by y value
    keypointsArray = []
    for k in keypoints:
        keypointsArray.append([k.pt[0], k.pt[1], k.size])
    sortedKeypoints = sorted(keypoints, key=lambda x: x.pt[1])

    if showImages:
        cv2.imshow("imgBubbles", imgBubbles)

    # check total number of bubbles found
    bubbleCount = len(keypoints)
    expectedBubbleCount = 220
    if bubbleCount != expectedBubbleCount:
        print("\033[91m" + "Error incorrect bubble count" + "\033[0m")
        return [], True
    
    # check bubbles found in each row
    bubbleMatrix = []
    rowBubbleCount = 0
    yOld = sortedKeypoints[0].pt[1]
    for k in sortedKeypoints:
        if (k.pt[1] - yOld) > params.minDistBetweenBlobs:
            bubbleMatrix.append(rowBubbleCount)
            rowBubbleCount = 0
        rowBubbleCount += 1
        yOld = k.pt[1]
    bubbleMatrix.append(rowBubbleCount)

    expectedBubbleMatrix = [10, 10, 10, 10, 10, 10, 10, 3, 1, 13, 13, 13, 17, 17, 17, 17, 17, 3, 1, 1, 3, 10, 2, 1, 1]
    if bubbleMatrix != expectedBubbleMatrix:
        print("\033[91m" + "Error incorrect bubble matrix" + "\033[0m")
        return [], True

    return sortedKeypoints, bubbleMatrix, False

def FindFilledBubbles(img, sortedKeypoints, bubbleMatrix):
    # find filled bubbles
    imgThresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    bubbleFillThreshPercent = 0.7
    bubbleMatrix2 = []
    totalBubbleCount = 0
    for c2 in bubbleMatrix:
        bubbleCount = 0
        rowValue = []
        
        # sort row by x value
        sortedKeypointsRow = sortedKeypoints[totalBubbleCount:totalBubbleCount + c2]
        sortedKeypointsRow = sorted(sortedKeypointsRow, key=lambda x: x.pt[0])

        for k in sortedKeypointsRow:
            bubbleCount += 1

            mask = np.zeros(img.shape, dtype="uint8")
            mask = cv2.circle(mask, (int(k.pt[0]), int(k.pt[1])), int(bubbleRad), 255, -1)
            mask2 = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
            percentFill = 1 - (cv2.countNonZero(mask2) / cv2.countNonZero(mask))

            if percentFill > bubbleFillThreshPercent:
                rowValue = bubbleCount

        bubbleMatrix2.append(rowValue)
        totalBubbleCount += c2
    
    return bubbleMatrix2, False

def ReadScoutingFormData(bubbleMatrix2):
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
    scoutingFormData.teleopLow = FormatBlankData(bubbleMatrix2[12])

    if bubbleMatrix2[14]:
        scoutingFormData.teleopHighOuter = bubbleMatrix2[14] + 17
    else:
        scoutingFormData.teleopHighOuter = FormatBlankData(bubbleMatrix2[13])

    if bubbleMatrix2[16]:
        scoutingFormData.teleopHighInner = bubbleMatrix2[16] + 17
    else:
        scoutingFormData.teleopHighInner = FormatBlankData(bubbleMatrix2[15])

    scoutingFormData.shootLocation = FormatBlankData(bubbleMatrix2[17])

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
            "Teleop High Outer Goals",
            "Teleop High Inner Goals",
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
                match.teleopHighOuter,
                match.teleopHighInner,
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
                    tempRow[8] = scoutingFormData.teleopHighOuter
                    tempRow[9] = scoutingFormData.teleopHighInner
                    tempRow[10] = scoutingFormData.shootLocation
                    tempRow[11] = scoutingFormData.controlPanel2
                    tempRow[12] = scoutingFormData.controlPanel3
                    tempRow[13] = scoutingFormData.climb
                    tempRow[14] = scoutingFormData.foul
                    tempRow[15] = scoutingFormData.card
                    tempRow[16] = scoutingFormData.disabled
                    tempRow[17] = scoutingFormData.playedDefense

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
    
    outputFilename = "rawFormData.csv"
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
            
            img = cv2.imread(unprocessedFilepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("\033[91m" + "Error failed to read image" + "\033[0m")
                continue

            img = ResizeImg(img, height)
            if showImages:
                cv2.imshow("img", img)
            
            # imgCrop = CropToForm(img)

            sortedKeypoints, bubbleMatrix, isError = FindBubbles(img)
            if isError:
                continue

            bubbleMatrix2, isError = FindFilledBubbles(img, sortedKeypoints, bubbleMatrix)
            if isError:
                continue

            scoutingFormData, isError = ReadScoutingFormData(bubbleMatrix2)
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
