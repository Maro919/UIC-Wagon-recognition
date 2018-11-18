import cv2
import numpy as np
import time
import scipy
import pytesseract
import csv
from imutils.object_detection import non_max_suppression
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
import argparse


#imgMat = scipy.misc.imread('face.png') to jest tablica tablic wartości w obrazie
#print(imgMat) 

i = 42
draw = False
view = True
f_detect = True

pos = [575,560]
size = 60

tab = []

template = cv2.imread('..\\hack\\Training\\Templates\\sign.jpg')
width_tep, height_tep = template.shape[:-1]
addicional_template = 0

ad_tmp_w = 0
ad_tmp_h = 0

comparison_img = 0
s_comparison_img = 0
first_frame = 0
#pixelDiffFrame = 0
crop_pos = []
last_frame = 0
pre_last_frame = 0
current_frame = 0
f = 51
g = 'left'
l = -1
uic_0_1 = 'locomotive'
uic_label = 0
draw = True
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
with open('employee_file2.csv', mode='w') as csv_file:
    fieldnames = ['team_name', 'train_number', 'left_right', 'frame_number', 'wagon', 'uic_0_1', 'uic_label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    

    #for i in range(30):
    while(True):
        x='..\\hack\\Training\\0_37\\0_37_left\\0_64_left_'+str(i)+'.jpg'
        img1 = cv2.imread(x)

        #amountOfUnchanged = 0
        #amountOfFrameCells = len(img1) * len(img1[0])
        #img1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        add=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(add,190,255,cv2.THRESH_BINARY)
        rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        #blur = cv2.GaussianBlur(rgb, (5, 5), 0)
        #temp_gray = cv2.cvtColor(template,cv2.COLOR_GRAY2RGB)
        
        if f_detect == True:
            comparison_img = img1
            pixelDiffFrame = img1
            #first_frame = img1
            threshold = 0.55
        else:
            comparison_img = img1[int(crop_pos[1] - 1):int(crop_pos[1] + height_tep+1), int(crop_pos[0] - 1):int(crop_pos[0] + width_tep+1)]
            threshold = 0.4
            if g == 'left':
                s_comparison_img = img1[int(ad_tmp_h - 4):int(ad_tmp_h + height_tep + height_tep + height_tep + 4), int(ad_tmp_w - width_tep - 20):int(crop_pos[0] - 6)]
            else:
                s_comparison_img = img1[int(ad_tmp_h - 4):int(ad_tmp_h + height_tep + height_tep + height_tep + 4), int(ad_tmp_w + width_tep + 6):int(ad_tmp_w + width_tep + width_tep + 14) ]
            s_threshold = 0.9
            s_res = cv2.matchTemplate(img1,addicional_template,cv2.TM_CCOEFF_NORMED)
            loc2 = np.where (s_res >= s_threshold)
        res = cv2.matchTemplate(comparison_img,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        certain_loc = np.where( res >= 0.48)

        if view == True:
            rgb = cv2.cvtColor(add,cv2.COLOR_GRAY2RGB)


        #for xx in range(len(first_frame)):
        #    for yy in range(len(first_frame[xx])):
        #        for zz in range(3):
                #print(pixelDiffFrame[xx][yy][zz])
        #            pixelDiffFrame[xx][yy][zz] = (first_frame[xx][yy][zz]) ^ (img1[xx][yy][zz])
        #        if pixelDiffFrame[xx][yy][zz] == first_frame[xx][yy][zz]:
        #            amountOfUnchanged += 1
        
        #percentOfDifference = amountOfUnchanged / amountOfFrameCells
        #print(percentOfDifference)

        if draw:
            #print(pytesseract.image_to_string(img1))
            tab = []
            for j in range(int(1024/size)):
                for ii in range(int(1280/size)):
                    w = 0
                    b = 0
                    #if (mask[(j*size)+int(size/2),(ii*size)+int(size/2)] == 0):
                    #    continue
                    for yy in range(size):
                        if w > (size*size)*.05:
                            break
                        for xx in range(size):
                            if mask[(j*size)+yy,(ii*size)+xx] != 0:
                                w+=1
                                if w > (size*size)*.05:
                                    break
                    if w > (size*size)*.05:
                        cv2.rectangle(rgb,(((ii)*size),((j)*size)),(((ii)*size)+size,((j)*size)+size),(0,255,0),2)
                        tab.append([ii,j])
            #draw = ~draw
            print(len(tab))
            count = 0
            maxval = 0
            mpos = [0,0]
            for u in tab:
                cnt=0
                for o in tab:
                    if o[1]<=u[1]+8 and o[1]>=u[1] and o[0]<=u[0]+15 and o[0]>=u[0]:
                        cnt+=1
                if cnt > maxval:
                    maxval = cnt
                    mpos = u
                
                    
            cv2.rectangle(rgb,((mpos[0]*size),(mpos[1]*size)),((mpos[0]*size)+size*15,(mpos[1]*size)+size*8),(0,255,255),2)

            if f_detect == True:
                isTrain = 1
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(rgb, pt, (pt[0] + width_tep, pt[1] + height_tep), (0,255,255), 2)
                    crop_pos = pt
                    ad_tmp_w = crop_pos[0]
                    ad_tmp_h = crop_pos[1] + height_tep - 20
                    #print(ad_tmp_pos)
                    if g == 'left':
                        addicional_template = img1[int(ad_tmp_h):int(ad_tmp_h + height_tep + height_tep + height_tep), int(ad_tmp_w - width_tep - 20):int(crop_pos[0] - 10) ]
                    else:
                        addicional_template = img1[int(ad_tmp_h):int(ad_tmp_h + height_tep + height_tep + height_tep), int(ad_tmp_w + width_tep + 10):int(ad_tmp_w + width_tep + width_tep + 10) ]
                    #cv2.imshow('addicional', addicional_template)
                    isTrain = 0
                    break
                
                if isTrain == 1:
                    current_frame = False
                else:
                    current_frame = True
                    f_detect = False
            else:
                breaking_point = 0
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(rgb, pt, (pt[0] + width_tep, pt[1] + height_tep), (0,255,255), 2)
                    breaking_point += 1
                    break
                for pt in zip(*certain_loc[::-1]):
                    cv2.rectangle(rgb, pt, (pt[0] + width_tep, pt[1] + height_tep), (0,255,255), 2)
                    breaking_point += 1
                    break
                for pt in zip(*loc2[::-1]):
                #    #cv2.rectangle(rgb, pt, (pt[0] + width_tep, pt[1] + height_tep), (0,255,255), 2)
                    breaking_point += 2
                    break
                #print(breaking_point)
                if last_frame == True:
                    pre_last_frame = True
                elif last_frame == False:
                    pre_last_frame = False

                last_frame = current_frame

                if breaking_point >= 2:
                    current_frame = True
                else:
                    current_frame = False


                    #break
                    #print(count) 
            if last_frame == False and pre_last_frame == False and current_frame == True:
                print('Nowy Wagon')
                l+=1  
                if (l>0):
                    uic_0_1 = 0                                                          
        #print(mask[600,600])
        #left rect
        #cv2.rectangle(rgb,(675 ,460),(705, 490),(255,0,0),2)
        #right rect
        #cv2.rectangle(rgb,(515 ,450),(545, 480),(255,0,0),2)
        rgb[600,600]=(255,0,0)
        #print(pos[0],pos[1])
        cv2.imshow("add",rgb)

        writer.writerow({'team_name': 'A Jack\'s and Jason\'s', 'train_number': f, 'left_right': g, 'frame_number': i, 'wagon': l, 'uic_0_1': uic_0_1, 'uic_label': uic_label})
        i+=1
        y = cv2.waitKey(20)
        #print(y)
        if y == 27:
            break
        # else: 
        #     if y == ord('z'):
        #         i-=1
        #     else:
        #         if y == ord('f'):
        #             draw = ~draw
        #         else:
        #             if y == ord('g'):
        #                 view = ~view
        #             else:
                       

    
    
    

    
    
    

    

# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image ../Hack/Training/0_0/0_0_left/0_0_left_56.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

        tabliczka=[]
        def decode_predictions(scores, geometry):
            # grab the number of rows and columns from the scores volume, then
            # initialize our set of bounding box rectangles and corresponding
            # confidence scores
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []
            # loop over the number of rows
            for yyy in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, yyy]
                xData0 = geometry[0, 0, yyy]
                xData1 = geometry[0, 1, yyy]
                xData2 = geometry[0, 2, yyy]
                xData3 = geometry[0, 3, yyy]
                anglesData = geometry[0, 4, yyy]

                # loop over the number of columns
                for xxx in range(0, numCols):
                    # if our score does not have sufficient probability,
                    # ignore it
                    if scoresData[xxx] < args["min_confidence"]:
                        continue

                    # compute the offset factor as our resulting feature
                    # maps will be 4x smaller than the input image
                    (offsetX, offsetY) = (xxx * 4.0, yyy * 4.0)

                    # extract the rotation angle for the prediction and
                    # then compute the sin and cosine
                    angle = anglesData[xxx]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height
                    # of the bounding box
                    h = xData0[xxx] + xData2[xxx]
                    w = xData1[xxx] + xData3[xxx]

                    # compute both the starting and ending (x, y)-coordinates
                    # for the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[xxx]) + (sin * xData2[xxx]))
                    endY = int(offsetY - (sin * xData1[xxx]) + (cos * xData2[xxx]))
                    startX = int(endX - w )
                    startY = int(endY - h )

                    # add the bounding box coordinates and probability score
                    # to our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[xxx])

            # return a tuple of the bounding boxes and associated confidences
            return (rects, confidences)

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str,
            help="path to input image")
        ap.add_argument("-east", "--east", type=str,
            help="path to input EAST text detector")
        ap.add_argument("-c", "--min-confidence", type=float, default=0.4,
            help="minimum probability required to inspect a region")
        ap.add_argument("-w", "--width", type=int, default=1024,
            help="nearest multiple of 32 for resized width")
        ap.add_argument("-e", "--height", type=int, default=1024,
            help="nearest multiple of 32 for resized height")
        ap.add_argument("-p", "--padding", type=float, default=0.115,
            help="amount of padding to add to each border of ROI")
        args = vars(ap.parse_args())

        # load the input image and grab the image dimensions

        add1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        #red,mask=cv2.threshold(add,180,0,cv2.THRESH_BINARY)
        for yyy in range(len(add)):
            for xxx in range(len(add[yyy])):
                add1[yyy][xxx]=255-add1[yyy][xxx]
                
        red=cv2.cvtColor(add1,cv2.COLOR_GRAY2RGB)
        


        #TEKST MA BYĆ CZARNY WTEDYY BEDZIE DZIAŁAĆ!

        image = red

        orig = image.copy()
        (origH, origW) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args["width"], args["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
       

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))

        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r:r[0][1])

        # loop over the results
        for ((startX, startY, endX, endY), text) in results:
            # display the text OCR'd by Tesseract

            if(len (tabliczka)<8):
                tabliczka+=format(text)	
            for ddd in range (len(tabliczka)):
                if((type(tabliczka[ddd])==int and tabliczka[ddd]<=9 and tabliczka[ddd] >=0) or tabliczka[ddd]=='-' ):
                    if(len (tabliczka)>=8 or len (tabliczka)==12 or len (tabliczka)==10):	
                        break
                    else: 
                        del tabliczka[:]
                        break
                else:
                    del tabliczka[:]
                    break  
        
            if tabliczka!=None:
                
                print(tabliczka)
        
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            output = orig.copy()
            cv2.rectangle(output, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(output, text, (startX, startY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # show the output image
        	#cv2.imshow("Text Detection", output)
            #cv2.waitKey(0)
            
