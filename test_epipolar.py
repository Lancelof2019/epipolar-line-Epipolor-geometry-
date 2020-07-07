import cv2
import numpy as np

img_org1=cv2.imread("test1.jpg")
img_org2=cv2.imread("test2.jpg")
#print("The pixcel of img is")
#print(img_org1.shape)
gray_imga=cv2.cvtColor(img_org1,cv2.COLOR_BGR2GRAY)

gray_imgb=cv2.cvtColor(img_org2,cv2.COLOR_BGR2GRAY)

gray_img1=cv2.resize(gray_imga,(500,500),0,0)
gray_img2=cv2.resize(gray_imgb,(500,500),0,0)

cv2.imshow("gray_img1",gray_img1)
cv2.waitKey(0)

cv2.imshow("gray_img2",gray_img2)
cv2.waitKey(0)

sift1=cv2.xfeatures2d.SIFT_create()

keypoints1,descriptors1=sift1.detectAndCompute(gray_img1,None)
#print("The lenth of descriptor1 is : ",descriptors1.shape)
###(1920,128)a
#print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#print("The keypoint of gray_img1 is :")

#print(keypoints1)
#print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

keypoints2,descriptors2=sift1.detectAndCompute(gray_img2,None)
show_img1=cv2.drawKeypoints(gray_img1,keypoints1,img_org1)
show_img2=cv2.drawKeypoints(gray_img2,keypoints1,img_org2)
cv2.imshow("keypoints1",show_img1)

cv2.waitKey(0)

cv2.imwrite("show1.jpg",show_img1)

cv2.imshow("Keypoints2",show_img2)
cv2.waitKey(0)
cv2.imwrite("show2.jpg",show_img2)


#print("key ponits")
#print(keypoints1)
#print(len(keypoints1))
#print("*******************************")
#print("descriptors")
#print(descriptors1)
#print(descriptors1.shape)


###I try to write KNN for distance####

min=[0,0]
#knnSet=np.zeros(((descriptors1.shape[0]), 2))
knn_list=[]
print(descriptors1.shape[0])
print(descriptors2.shape[0])
record_point=[]
for i in range(descriptors1.shape[0]):
    initial_distance=cv2.norm(descriptors1[i],descriptors2[0],cv2.NORM_L2)
    min[0]=initial_distance
    min[1]=initial_distance
    for j in range(descriptors2.shape[0]):
        distance=cv2.norm(descriptors1[i],descriptors2[j],cv2.NORM_L2)
        #if i==0 and j==0:
          # min[0]=distance
           #min[1]=distance
        if  min[0]>distance:
            temp1=min[0]
            # min0_index=j
            min[0]=distance
            min[1]=temp1
            desc2_index=j
            desc1_index=i
            desc3_index=j-1
        elif min[1]>distance:
             min[1]=distance
             desc3_index=j
             desc1_index=i
    # print("******************************************************************************************************")
    # print(f"The shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc2_index}] is min[0]")
    # print(min[0])
    # print(f"The second shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc3_index}] is min[1]")
    # print(min[1])
    # print("******************************************************************************************************")
    #knnSet[i][0]=min[0]
    #knnSet[i][1]=min[1]
    #knnSet[i][0]=cv2.DMatch(i,desc2_index,min[0])
    #knnSet[i][1]=cv2.DMatch(i,desc3_index,min[1])
    knn_list=knn_list+[[cv2.DMatch(i,desc2_index,min[0]),cv2.DMatch(i,desc3_index,min[1])]]
    #print("*******************************************************")
    #print(f"The Dmatch between {i} and {desc2_index} is :")
    #print(cv2.DMatch(i,desc2_index,min[0]))
    #print(f"The Dmatch between {i} and {desc3_index} is :")
    #print(cv2.DMatch(i,desc3_index,min[0]))a
print(len(knn_list))

knnSet=np.array(knn_list)
print(knnSet.shape)
print(knnSet[9][0])
print(knnSet[9][1])
ratio_threshold=0.8
matches=[]
for m,n in knnSet:
    if m.distance < ratio_threshold*n.distance:
       #matches.append([m])a
       print("*****************************")
       #print(m.distance)
       print("#########################m is ######################:")
       print(m)
       print("###############m's distanc is######################")
       print(m.distance)
       print("#########################n is ######################:")
       print(n)
       print("###############n's distanc is######################")
       print(n.distance)
       print("*****************************")
       matches.append(m)
matches_array=np.array(matches)
print("The length of matches is :",matches_array.shape)


for m,n in enumerate(matches_array):
    print(m)
    print(n)
#print(len(matches))

img_matches=cv2.drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, matches,outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), flags=2)

cv2.imshow("performance",img_matches)
cv2.waitKey(0)
cv2.imwrite("compare_performance.jpg",img_matches)
#print(len(knnSet))
#print(knnSet.size)
###the elements
#print(len(knnSet))


pts1=[]
pts2=[]

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(matches[2])
print(matches[2].trainIdx)
print(keypoints1[matches[2].trainIdx].pt)
print(keypoints2[matches[2].trainIdx].pt)

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#pts1=pts1.append(keypoints[])
for i in range(len(matches)):
    pts2.append(keypoints2[matches[i].trainIdx].pt)
    pts1.append(keypoints1[matches[i].queryIdx].pt)

print("*********************************************************************************")
print("The point of pts1 :")
print(pts1)
print(len(pts1))
print("*********************************************************************************")

print("*********************************************************************************")
print("The point of pts2 :")
print(pts2)
print(len(pts2))
print("*********************************************************************************")
#print("%%%%%%%%%%%%%%%%%%%%The descriptor is%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#print(descriptors1)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

print(pts1)

print(pts2)


F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

print("###########################################")
print("The fundmantal matrix is")
print(F)
print("###########################################")
print("The mask is ")
print(mask)
print("###########################################")




#def drawlines(img1, img2, lines, pts1, pts2):
#    ''' img1 - image on which we draw the epilines for the points in img2
#        lines - corresponding epilines '''
r, c = gray_img1.shape
#img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
#img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
for r, pt1, pt2 in zip(lines1, pts1, pts2):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -r[2]/r[1]])
    x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
    img1 = cv2.line(gray_img1, (x0, y0), (x1, y1), color, 1)
    img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    img2 = cv2.circle(gray_img2, tuple(pt2), 5, color, -1)
   # return img1, img2
cv2.imwrite("epipolar_img1.jpg",img1)
cv2.imshow("epipolar_img1",img1)
cv2.waitKey(0)
cv2.imwrite("epipolar_img2.jpg",img2)
cv2.imshow("epipolar_img2",img2)
cv2.waitKey(0)

for r,wpt1,wpt2 in zip(lines2,pts1,pts2):
    wcolor = tuple(np.random.randint(0, 255, 3).tolist())
    wx0, wy0 = map(int, [0, -r[2]/r[1]])
    wx1, wy1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
    img3 = cv2.line(gray_img2, (wx0, wy0), (wx1, wy1), wcolor, 1)
    img3 = cv2.circle(img3, tuple(wpt1), 5, wcolor, -1)
    img4 = cv2.circle(gray_img1, tuple(wpt2), 5, wcolor, -1)

cv2.imwrite("epipolar_img3.jpg",img3)
cv2.imshow("epipolar_img3",img3)
cv2.waitKey(0)
cv2.imwrite("epipolar_img4.jpg",img4)
cv2.imshow("epipolar_img4",img4)
cv2.waitKey(0)



#lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
#lines1 = lines1.reshape(-1, 3)

#cv2.imwrite("Fundamental_matrix.jpg",lines1)
#cv2.imshow("lines1",lines1)
#cv2.waitKey(0)
