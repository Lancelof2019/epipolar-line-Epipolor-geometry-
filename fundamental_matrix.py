import numpy as np


def computeF(points1, points2):

    assert (len(points1) == 8), "Length of points1 should be 8!"
    assert (len(points2) == 8), "Length of points2 should be 8!"

    A = np.zeros((8, 9)).astype('int')
    for i in range(8):
        '''
        TODO 2.2
        ## Step 1
        ## Fill the Matrix A
        '''
        para1=points1[i]
        para2=points2[i]
        para_order=[para1[0]*para2[0],para1[0]*para2[1],para1[0],para1[1]*para2[0],para1[1]*para2[1],para1[1],para2[0],para2[1],1]
        A[i]=para_order

    '''
    ## TODO 2.2
    ## Step 2
    ## Solve Af = 0 with SVD
    '''
    U,S,VT=np.linalg.svd(A)
    print("*********************************************")
    print("U is :")
    print(U)
    print("Sigma is :")
    print(S)
    print("VT is : ")
    print(VT)
    print("*********************************************")
    print("The last of eigenvector of A")
    f_vector=VT[-1]
    print(VT[-1])
    print(len(VT[-1]))
    
    F_tmp=np.array(f_vector)
    
    print(F_tmp)
    
    F=F_tmp.reshape(3,3)
    print(F)
    
    F_value=np.linalg.norm(F)
    print("The value of F is:",F_value)
    #Frow,Fcol=F.shape
    #F_array_temp=
    #for op in range(Frow):
    F=F/F_value
    print("The normalized F is")
    
    print(np.linalg.norm(F))
    print("The original F is")
    print(F)
    #print(np.linalg.norm(F))
    
    
    U,S,VT=np.linalg.svd(F)
    
    print("*********************************************")
    print("U is :")
    print(U)
    print("Sigma is :")
    print(S)
    print("VT is : ")
    print(VT)
    print("*********************************************")
    
    
   

    
    
    
    
    
    
    
    
    '''
    ## TODO 2.2
    ## Step 3
    ## - Enforce Rank(F) = 2
    '''
    F_estimate=np.dot(np.dot(U,np.diag([1,1,0])),VT)
    F=F_estimate
    print("New F matrix with method of estimation is ")

    ## Step 4 - Normalize F
    F = F * (1.0 / F[2, 2])
    return F

def epipolarConstraint(p1, p2, F, t):

    p1h = np.array([p1[0], p1[1], 1])
    p2h = np.array([p2[0], p2[1], 1])

    ## TODO 2.1
    ## - Compute the normalized epipolar line
    ## - Compute the distance to the epipolar line
    ## - Check if the distance is smaller than t
    pts1=p1
    pts2=p2
    

    line2=np.dot(F,p1h)
   
    sub_line2=[line2[0],line2[1]]
    
    normline2=np.linalg.norm(sub_line2)
    line2=line2/normline2
    print("The normlized line correclates :")
    print(line2)
      
    line1=np.dot(np.transpose(F),ph2)
    sub_line1=[line1[0],line1[1]]
    normline1=np.linalg.norm(sub_line1)
    line1=line1/normline1
    
    pts1_set=[]
    pts2_set
    threshold=4
    subtest_pts1=list(pts1)
    print(subtest_pts1)
    for i in range(len(pts1)):
         list_pts1=list(pts1[i])
         list_pts1.append(1)
         distance=np.dot(np.transpose(list_pts1),line2)
         distance=np.abs(distance)
         if (distance >=t):
            pts1_set.append(list_pts1)
    return False

def numInliers(points1, points2, F, threshold):
    inliers = []
    for i in range(len(points1)):
        if (epipolarConstraint(points1[i], points2[i], F, threshold)):
            inliers.append(i)

    return inliers

def computeFRANSAC(points1, points2):

    ## The best fundamental matrix and the number of inlier for this F.
    bestInlierCount = 0
    threshold = 4
    iterations = 10000

    for k in range(iterations):
        if k % 1000 == 0:
            print (str(k) + " iterations done.")
        subset1 = []
        subset2 = []
        for i in range(8):
            x = np.random.randint(0, len(points1)-1)
            subset1.append(points1[x])
            subset2.append(points2[x])
        F = computeF(subset1, subset2)
        num = numInliers(points1, points2, F, threshold)
        if (len(num) > bestInlierCount):
            bestF = F
            bestInlierCount = len(num)
            bestInliers = num

    return (bestF, bestInliers)

def testFundamentalMat():
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11), (11, 2), (-3, 14), (236, -514), (-5, 1)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15),
                                    (215, -156), (151, 83), (1544, 15), (451, -55)]

    F = computeF(points1, points2)

    print ("Testing Fundamental Matrix...")
    print ("Your result:" + str(F))

    Href = np.array([[0.001260822171230067,  0.0001614643951166923, -0.001447955678643285],
                 [-0.002080014358205309, -0.002981504896782918, 0.004626528742122177],
                 [-0.8665185546662642,   -0.1168790312603214,   1]])

    print ("Reference: " + str(Href))

    error = Href - F
    e = np.linalg.norm(error)
    print ("Error: " + str(e))

    if (e < 1e-10):
        print ("Test: SUCCESS!")
    else:
        print ("Test: FAIL!")
    print ("============================")
