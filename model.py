import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def calculateSSIM(template_img, query_img):
    min_w = min(template_img.shape[1], query_img.shape[1])
    min_h = min(template_img.shape[0], query_img.shape[0])
    
    img1 = cv2.resize(template_img, (min_w, min_h))
    img2 = cv2.resize(query_img, (min_w, min_h))
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score = ssim(img1, img2)
    return score

def computeORB(template_img, query_img):
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    kpts1, descs1 = orb.detectAndCompute(template_img,None)
    kpts2, descs2 = orb.detectAndCompute(query_img,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
    
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Not enough keypoints for homography calculation")
        return None, None, None, None, None
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h, w = template_img.shape[:2]
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts, M) if M is not None else None
    return dst, dst_pts, kpts1, kpts2, dmatches



# Values for specifying search area of features 1 to 7
search_area_list = [[200,270,160,330],
                    [1050,1500,250,400],
                    [50,400,0,100],
                    [750,1050,0,100],
                    [850,1050,280,380],
                    [700,820,290,370],
                    [400,650,0,100]]

# Values of max_area and min_area for each feature for features 1 to 7
feature_area_limits_list = [[10000,14000],
                            [9000,15000],
                            [17000,21500],
                            [19000,28000],
                            [17500,23000],
                            [6500,9000],
                            [10000,16000]]

score_set_list = []               # Stores the ssim score set of each feature
best_extracted_img_list = []      # Stores the extracted image with highest SSIM score for each feature
avg_ssim_list = []                # Stores the avg ssim value for each feature
NUM_OF_FEATURES = 7               # Number of features


def testFeature_1_2_7(gray_test_image, test_img, blur_test_img, denomination):
    i = 0
    j = 0
    NUMBER_OF_TEMPLATES = 6
    global score_set_list                # Stores the ssim score set of each feature
    global best_extracted_img_list       # Stores the extracted image with highest SSIM score for each feature
    global avg_ssim_list                 # Stores the avg ssim value for each feature

    
    #Progress bar
    #global myProgress
    #myProgress =myProgress['value']
    
    # Iterating for each feature
    for j in range(NUM_OF_FEATURES):
        print('ANALYSIS OF FEATURE ' + str(j+1))

        score_set = []           # SSIM scores for each teamplate of current feature will be stored here
        max_score = -1           # Stores max SSIM score
        max_score_img = None     # Stores extraced image with max SSIM score for the current feature
        
        # Performing feature detection, extraction and comparison for each template stored in dataset 
        for i in range(NUMBER_OF_TEMPLATES):
            print('---> Template ' + str(i+1) + ' :')
            
            # Current template 
            template_path = template_path = rf"Dataset\{denomination}_Features Dataset\{j+1}\{i+1}.jpg"
       
            template_img = cv2.imread(template_path)
            print(f"Trying to load: {template_path}")
 
            template_img_blur = cv2.GaussianBlur(template_img, (5,5), 0)
            template_img_gray = cv2.cvtColor(template_img_blur, cv2.COLOR_BGR2GRAY)
            test_img_mask = gray_test_image.copy()
            
            # Creating a mask to search the current template.
            search_area = search_area_list[j]

            test_img_mask[:, :search_area[0]] = 0
            test_img_mask[:, search_area[1]:] = 0
            test_img_mask[:search_area[2], :] = 0
            test_img_mask[search_area[3]:, :] = 0
            
            # Feature detection using ORB 
            dst, dst_pts, kpts1, kpts2, dmatches = computeORB(template_img_gray, test_img_mask)
            
            # Error handling
            if dst is None:
                print(f"Skipping feature {j+1} due to insufficient matches.")
                continue
            
            query_img = test_img.copy()
            
            # drawing polygon around the region where the current template has been detected on the test currency note -- the blue polygon
            res_img1 = cv2.polylines(query_img, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
            st.image(res_img1, caption="Detected Feature", use_column_width=True)
            # draw match lines between the matched descriptors
            res_img2 = cv2.drawMatches(template_img, kpts1, res_img1, kpts2, dmatches[:20],None,flags=2)
            st.image(res_img2, caption="Matched Descriptors", use_column_width=True)

            # Find the details of a bounding rectangle that bounds the above polygon --- green rectangle
            (x, y, w, h) = cv2.boundingRect(dst) # This gives us details about the rectangle that bounds this contour  
            
            # Checking if the area of the detected region is within the min and max area allowed for current feature 
            min_area = feature_area_limits_list[j][0]
            max_area = feature_area_limits_list[j][1]

            feature_area = w*h

            if feature_area < min_area or feature_area > max_area:
                (x, y, w, h) = cv2.boundingRect(dst_pts) 

                feature_area = w*h
                if feature_area < min_area or feature_area > max_area: 
                    # If even area of 2nd rect is outside limits, then Discard current template
                    print('Template Discarded- Area of extracted feature is outside permitted range!')
                    continue

            # Draw the rectangle
            cv2.rectangle(res_img1, (x,y), (x+w, y+h), (0,255,0), 3)
            st.image(res_img1, caption="Bounding Rectangle", use_column_width=True)
            

            # SSIM calculation
            # Crop out the region inside the green rectangle (matched region)
            crop_img = blur_test_img[y:y+h, x:x+w]

            plt.rcParams["figure.figsize"] = (5, 5)
            score = calculateSSIM(template_img_blur, crop_img)

            score_set.append(score)
            print('SSIM score: ', score, '\n')
            
            # Keeping details about extracted region with highest SSIM score
            if score > max_score:
                max_score = score
                max_score_img = crop_img
                
            # #Progress bar- Updating the progess
            # myProgress = myProgress + (75.0/(NUM_OF_FEATURES*NUMBER_OF_TEMPLATES))
            # progress['value'] = myProgress 
            
        # Storing necessary data
        score_set_list.append(score_set)
        print('SSIM score set of Feature ' + str(j+1) + ': ', score_set, '\n')
        
        if len(score_set) != 0:
            avg_ssim_list.append(sum(score_set)/len(score_set))
            print('Average SSIM of Feature ' + str(j+1) + ': ',sum(score_set)/len(score_set),'\n')
        else:
            print('No SSIM scores were found for this feature!')
            avg_ssim_list.append(0.0)
            print('Average SSIM of Feature ' + str(j+1) + ': 0','\n')
        
        best_extracted_img_list.append([max_score_img, max_score])

    # Printing all details for features 1- 7
    print('Final Score- set list:','\n')

    for x in range(len(score_set_list)):
        print('Feature',x+1,':',score_set_list[x])
    print('\n')

    print('Final Average SSIM list for each feature:','\n')

    for x in range(len(avg_ssim_list)):
        print('Feature',x+1,':',avg_ssim_list[x])
        
    results = []
    for i, score in enumerate(avg_ssim_list):
        status = "Pass" if score > 0.5 else "Fail"
        results.append((max_score_img, score, status))

    # Calculate average and max SSIM scores
    all_scores = avg_ssim_list
    avg_ssim = np.mean(all_scores) if all_scores else 0.0
    max_ssim = np.max(all_scores) if all_scores else 0.0

    # Return the results along with avg_ssim and max_ssim
    return results, avg_ssim, max_ssim

def testFeature_8(image, denomination):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if denomination == "2000":
        crop = gray[80:230, 10:30]
    else:
        crop = gray[120:240, 12:35]
    
    _, thresh = cv2.threshold(crop, 130, 255, cv2.THRESH_BINARY)
    
    whitePixelValue = 255      # White pixel   
    blackPixelValue = 0        # Black pixel

    width = thresh.shape[1]    # width of thresholded image

    # Result will be stored here
    result = []                # will contain the number of black regions in each column (if the colums is non- erroneos)
    num_of_cols = 0            # will contain the number of non- erroneos columns

    # Non erroneous columns are those columns which contains less noise.

    print('Number of black regions found in each column: ')
    
    # iteration over each column in the cropped image
    for j in range(width):
        col =thresh[:, j:j+1]     # Extracting each column of thresholded image
        count = 0                 # Counter to count number of black regions in each extracted column
        
        # Iterating over each row (or pixel) in the current columm
        for i in range(len(col)-1):

            # Taking two consecutive pixels and storing their intensity value
            pixel1_value = col[i][0]
            pixel2_value = col[i+1][0]

            #----------------------------------------------
            # This part modifies any error pixels, if present.
            # Actually in a binary threshold, all pixels should be either white or black.
            # If due to some error pixels other than white or black are present, then the pixel is taken as white pixel

            if pixel1_value != 0 and pixel1_value != 255:
                pixel1_value = 255
            if pixel2_value != 0 and pixel2_value != 255:
                pixel2_value = 255

            #-------------------------------------------------


            # If current pixel is white and next pixel is black, then increment the counter.
            # This shows that a new black region has been discovered.
            if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                count += 1

        # If the counter is less than 10, it is a valid column. (less noise is present)
        if count > 0 and count < 10:
            print(count)
            result.append(count)
            num_of_cols += 1
        else:
            # discard the count if it is too great e.g. count> 10 (Erroneous Column)
            # This removes/ drops those columns which contain too much noise
            print(count, 'Erroneous -> discarded') 
    
    # Printing necessary details
    print('\nNumber of columns examined: ', width)
    print('Number of non- erroneous columns found: ', num_of_cols)
    
    if num_of_cols != 0:
        average_count = sum(result)/num_of_cols
    else:
        average_count = -1
        print('Error occured- Division by 0')

    print('\nAverage number of black regions is: ', average_count)
    
    # Storing the thresholded image and average number of bleed lines detected 
    global left_BL_result
    left_BL_result = [thresh, average_count]
    return left_BL_result[1]

def testFeature_9(image, denomination):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if denomination == "2000":
        crop = gray[90:230, 1140:1160]
    else:
        crop= gray[120:260, 1135:1155]
    _, thresh = cv2.threshold(crop, 130, 255, cv2.THRESH_BINARY)

    whitePixelValue = 255      # White pixel   
    blackPixelValue = 0        # Black pixel

    width = thresh.shape[1]    # width of thresholded image

    # Result will be stored here
    result = []                # will contain the number of black regions in each column (if the colums is non- erroneos)
    num_of_cols = 0            # will contain the number of non- erroneos columns

    # Non erroneous columns are those columns which contains less noise.

    print('Number of black regions found in each column: ')
    
    # Iteration over each column in the cropped image
    for j in range(width):
        col =thresh[:, j:j+1]     # Extracting each column of thresholded image
        count = 0                 # Counter to count number of black regions in each extracted column
        
        # Iterating over each row (or pixel) in the current columm
        for i in range(len(col)-1):

            # Taking two consecurive pixels and storing their intensity value
            pixel1_value = col[i][0]
            pixel2_value = col[i+1][0]

            #----------------------------------------------
            # This part modifies any error pixels, if present.
            # Actually in a binary threshold, all pixels should be either white or black.
            # If due to some error pixels other than white or black are present, then the pixel is taken as white pixel

            if pixel1_value != 0 and pixel1_value != 255:
                pixel1_value = 255
            if pixel2_value != 0 and pixel2_value != 255:
                pixel2_value = 255

            #-------------------------------------------------

            # If current pixel is white and next pixel is black, then increment the counter.
            # This shows that a new black region has been discovered.
            if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                count += 1

        # If the counter is less than 10, it is a valid column. (less noise is present)
        if count > 0 and count < 10:
            print(count)
            result.append(count)
            num_of_cols += 1
        else:
            # discard the count if it is too great e.g. count> 10 (Erroneous Column)
            # This removes/ drops those columns which contain too much noise
            print(count, 'Erroneous -> discarded')

    # Printing necessary details
    print('\nNumber of columns examined: ', width)
    print('Number of non- erroneous columns found: ', num_of_cols)

    if num_of_cols != 0:
        average_count = sum(result)/num_of_cols
    else:
        average_count = -1
        print('Error occured- Division by 0')


    print('\nAverage number of black regions is: ', average_count)

    # Storing the thresholded image and average number of bleed lines detected 
    global right_BL_result
    right_BL_result = [thresh, average_count]
    return right_BL_result[1]

def testFeature_10(image, denomination):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = gray[360:440, 760:1080]
    crop_bgr = image[360:440, 760:1080]

    print('\n\nANALYSIS OF FEATURE 10 : NUMBER PANEL \n')

    test_passed = False        # If true, then the test is successful
    res_img_list = []          # List of images of successful cases
    count = 0                  # Stores number of cases whihc are successful
    num = 1
    
    # THRESHOLDING at multiple values
    # Start from 90 as threshold value, increase the threshold value by 5 every time and check if 9 characters are detected in the thresholded image of number panel
    # If 9 characters are detected in at least one of the cases, the currency number panel is verified.
    # If more than 1 cases pass, the best image will be choosen from the successful cases.
    if denomination == "2000":
        val = 90
    else:
        val = 95
    for thresh_value in range(val, 155, 5):
        # Thresholding at current value
        _, thresh = cv2.threshold(crop, thresh_value, 255, cv2.THRESH_BINARY)

        print('---> Threshold ' + str(num) + ' with Threshold value ' + str(thresh_value) + ' :')
        num += 1

        copy = crop_bgr.copy()

        # Finding all the contours in the image of the number panel- CONTOUR DETECTION 
        img = cv2.bitwise_and(crop, crop, mask=thresh)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        h_img, w_img = img.shape[:2]

        # cv2.drawContours(copy, contours, -1, (0, 0, 255), 1)

        # Storing the details of all the BOUNDING RECTANGLES for each contour
        bounding_rect_list = []    

        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)

            if x != 0:
                bounding_rect_list.append([x,y,w,h])

        # Sorting the list of rectangles
        # Rectangles will get sorted according to the x coordinate of the top left corner
        bounding_rect_list.sort()

        # ELIMINATION OF ERRONEOUS RECTANGLES
        # Min area is taken as 150
        min_area = 150
        res_list = []

        # Storing all rectangles having area greater than the min_area in a separate list
        for i in range(0, len(bounding_rect_list)):
            if i>= len(bounding_rect_list):
                break
            if bounding_rect_list[i][2]*bounding_rect_list[i][3] > min_area:
                res_list.append(bounding_rect_list[i])

        # Eliminating the rectangles that are present within a bigger rectangle
        i = 0
        while i<len(res_list):
            [x,y,w,h] = res_list[i]
            j = i+1
            while j<len(res_list):
                [x0,y0,w0,h0] = res_list[j]

                if (x+w) >= x0+w0:
                    res_list.pop(j)
                else:
                    break
            i+= 1

        # Eliminating unnecessary rectangles
        i = 0
        while i<len(res_list):
            [x,y,w,h] = res_list[i]

            if (h_img-(y+h)) > 40:     #  Eliminating the rectangles whose lower edge is further away from lower edge of the image 
                res_list.pop(i)
            elif h<17:
                res_list.pop(i)        # Eliminating the rectangles whose height is less than 17 pixels    
            else:
                i += 1
        
        for rect in res_list:          # Drawing the remaining rectangles
            [x,y,w,h] = rect
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)        

        # COUNTING REMAINING RECTANGLES
        # result of each image
        if len(res_list) == 9:         # If number of rectangles detected is equal to 9, test passed
            test_passed = True
            res_img_list.append(copy)
            count += 1
            print('Test Successful: 9 letters found!')
        else:
            print('Unsuccessful!')

        # If three consecutive cases pass the test, then break 
        if count == 3:
            break

    # Choosing the BEST IMAGE to be displayed   
    # Even if a single case passes the test, then the currency number panel is verified.
    # Selecting the best image to display
    if count == 0:                    # If none of the cases passes the test
        best_img = crop_bgr
    elif count == 1:                  # If 1 case passes the test, then the image used in 1st case is selected as the best image
        best_img = res_img_list[0]
    elif count == 2:                  # If 2 cases pass the test, then the image used in 2nd case is selected as best image
        best_img = res_img_list[1]
    else:                             # If >= 3 cases pass the test, then the image used in 3rd case is selected as best image
        best_img = res_img_list[2]
       
    
    # Displaying final result

    if(test_passed):
        print('Test Passed!- 9 characters were detected in the serial number panel.')
    else:
        print('Test Failed!- 9 characters were NOT detected in the serial number panel.')
    
    # Storing the thresholded image and the result
    global number_panel_result
    number_panel_result = [best_img, test_passed]
    return number_panel_result[1]

def analyze_features(image):
    num_features = 7
    results = []
    
    for i in range(num_features):
        dummy_feature = image[50:150, 50 + i*50:100 + i*50]
        ssim_score = np.random.uniform(0.3, 1.0)
        status = "Pass" if ssim_score > 0.5 else "Fail"
        results.append((dummy_feature, ssim_score, status))
    
    avg_ssim = np.mean([x[1] for x in results])
    max_ssim = np.max([x[1] for x in results])
    return results, avg_ssim, max_ssim

st.title("Currency Note Authentication")
denomination = st.selectbox("Select Denomination of Note", ["500", "2000"])
uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    test_img = cv2.resize(image, (1165,455))
    blur_test_img = cv2.GaussianBlur(test_img, (5,5), 0)
    gray_test_image = cv2.cvtColor(blur_test_img, cv2.COLOR_BGR2GRAY)

    st.subheader("Processing Image...")
    
    results, avg_ssim, max_ssim =testFeature_1_2_7(gray_test_image, test_img, blur_test_img, denomination)
    
    st.subheader("Results")
    st.write(f"**Number of authentic features:** {sum(1 for x in results if x[2] == 'Pass')}/7")
    st.write(f"**Average SSIM Score:** {avg_ssim:.2f}")
    st.write(f"**Maximum SSIM Score:** {max_ssim:.2f}")
    
    st.subheader("Feature Analysis")
    for idx, (feature_img, ssim_score, status) in enumerate(results):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(feature_img, caption=f"Feature {idx+1}", use_column_width=True)
        with col2:
            st.write(f"**SSIM Score:** {ssim_score:.2f}")
            st.write(f"**Status:** {status}")
    
    st.subheader("Additional Feature Analysis")
    
    # testFeature_1_2_7(gray_test_image, test_img, blur_test_img, denomination)
    left_bleed_lines = testFeature_8(image, denomination)
    right_bleed_lines = testFeature_9(image,denomination)
    number_panel = testFeature_10(image,denomination)
    
    
    # st.image(left_bleed_lines, caption="Left Bleed Lines", use_column_width=True)
    # st.image(right_bleed_lines, caption="Right Bleed Lines", use_column_width=True)
    # st.image(number_panel, caption="Number Panel", use_column_width=True)
    if (denomination == "2000"):
        min_ssim_score_list = [0.45, 0.4, 0.45, 0.45, 0.5, 0.4, 0.5]
    else:
        min_ssim_score_list = [0.4, 0.4, 0.5, 0.4, 0.5, 0.45, 0.5]
    successful_features_count = 0
    result_list = []
    # Feature 1 to 7 analysis 
    for i in range(NUM_OF_FEATURES):
        avg_score = avg_ssim_list[i]
        img, max_score = best_extracted_img_list[i]
        min_allowed_score = min_ssim_score_list[i]
        
        if avg_score >= min_allowed_score or max_score >= 0.79:
            successful_features_count += 1
            status = "Pass"
        else:
            status = "Fail"
            
        result_list.append((f"Feature {i+1}", status, avg_score))
    # Feature 8 - Left Bleed Lines
    img, line_count = left_BL_result
    if denomination == "2000":
        ll = 6.7
        hl = 7.6
    else:
        ll = 4.7
        hl = 5.6

    if ll <= line_count <= hl:
        successful_features_count += 1
        status = "Pass"
    else:
        status = "Fail"
    result_list.append(("Left Bleed Lines", status, line_count))
    # Feature 9 - Right Bleed Lines  
    img, line_count = right_BL_result
    if ll <= line_count <= hl:
        successful_features_count += 1
        status = "Pass"
    else:
        status = "Fail"
    result_list.append(("Right Bleed Lines", status, line_count))
    # Feature 10 - Number Panel
    img, number_panel_status = number_panel_result
    if number_panel_status:
        successful_features_count += 1
        status = "Pass"
    else:
        status = "Fail"
    result_list.append(("Number Panel", status, number_panel))
    # Display final results
    st.subheader("Authentication Result")
    
    success_rate = (successful_features_count / 10) * 100
    
    if success_rate >= 50:
        st.success(f"Currency Note is AUTHENTIC ({success_rate:.1f}% features verified)")
    else:
        st.error(f"Currency Note appears to be FAKE ({success_rate:.1f}% features verified)")
    # Display detailed results table
    st.write("Detailed Feature Analysis:")
    for feature, status, score in result_list:
        if isinstance(score, float):
            score_text = f"{score:.2f}"
        else:
            score_text = str(score)
            
        if status == "Pass":
            st.success(f"{feature}: {score_text}")
        else:
            st.error(f"{feature}: {score_text}")