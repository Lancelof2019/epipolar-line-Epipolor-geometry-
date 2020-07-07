import cv2
import numpy as np
import open3d as o3d

from fundamental_matrix import testFundamentalMat, computeFRANSAC
from triangulate import testTriangulate, triangulate_all_points
from decompose import testDecompose, relativeTransformation

def ratioTest(knnmatches, ratio_threshold):
    matches = []
    for inner_list in knnmatches:
        first  = inner_list[0]
        second = inner_list[1]

        if (first.distance < ratio_threshold * second.distance):
            matches.append(first)
    return matches

def main():
    testFundamentalMat()
    testTriangulate()
    testDecompose()

    img1 = cv2.imread("../data/img1.jpg")
    img2 = cv2.imread("../data/img2.jpg")

    assert (img1.data), "Image 1 is not properly read"
    assert (img2.data), "Image 2 is not properly read"

    K = np.array([[2890, 0, 1440], [0, 2890, 960], [0, 0, 1]])

    #******************Feature Matching*******************#
    detector = cv2.ORB_create(20000)
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    print("Num Features: " + str(len(keypoints1)) + " " + str(len(keypoints2)))

    FLANN_INDEX_LSH = 6
    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 12,
                               key_size = 20,
                               multi_probe_level = 2)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    knnmatches = matcher.knnMatch(descriptors1, descriptors2, 2)

    matches = ratioTest(knnmatches, 0.8)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                                  outImg=None, matchColor=(0, 255, 0),
                                  singlePointColor=(0, 255, 0), flags=2)
    # pdb.set_trace()
    img_matches = cv2.resize(img_matches, None, fx=0.2, fy=0.2)
    cv2.imshow("All Matches", img_matches)
    cv2.waitKey(0)

    print ("Matches after ratio test: " + str(len(matches)))

    #**************************Compute F and E**************************#
    points1 = []
    points2 = []
    for m in matches:
        points1.append(keypoints1[m.queryIdx].pt)
        points2.append(keypoints2[m.trainIdx].pt)

    F, inliers = computeFRANSAC(points1, points2)
    print ("RANSAC inliers " + str(len(inliers)))

    ## Compute E and normalize
    E = np.dot(np.dot(np.transpose(K), F), K)
    E *= 1.0 / E[2, 2]

    inlier_matches = []
    inlier_points1 = []
    inlier_points2 = []
    for i in inliers:
        inlier_matches.append(matches[i])
        inlier_points1.append(points1[i])
        inlier_points2.append(points2[i])

    ## Draw filtered matches (ransac inliers)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches,
                                  outImg=None, matchColor=(0, 255, 0),
                                  singlePointColor=(0, 255, 0), flags=2)
    img_matches = cv2.resize(img_matches, None, fx=0.2, fy=0.2)
    cv2.imshow("RANSAC Inlier Matches", img_matches)
    cv2.waitKey(0)

    ## Compute relative transformation
    View1 = np.eye(3, 4)
    View2 = relativeTransformation(E, inlier_points1, inlier_points2, K)

    ## Triangulate inlier matches
    wps = triangulate_all_points(View1, View2, K, inlier_points1, inlier_points2)
    # small sanity check to remove low angle triangulate points and some more outliers
    wps_updated = []
    for p in wps:
        if p[2] < 10:
            wps_updated.append(p)

    ## Rendering in Open3D
    colors = []
    for i in range(len(wps_updated)):
        colors.append(img1[int(inlier_points1[i][1]), int(inlier_points1[i][0])])
    colors = np.array(colors)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(wps_updated)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    main()
