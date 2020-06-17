import cv2
import numpy as np


def distance(f1, f2):    
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2 - x1)**2+ (y2 - y1)**2)

def filteringByDistance(kp, distE=0.5):
    size = len(kp)
    mask = np.arange(1,size+1).astype(np.bool8) # all True   
    for i, f1 in enumerate(kp):
        if not mask[i]:
            continue
        else: # True
            for j, f2 in enumerate(kp):
                if i == j:
                    continue
                if distance(f1, f2)<distE:
                    mask[j] = False
    np_kp = np.array(kp)
    return list(np_kp[mask])

src = cv2.imread('Chessboard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

fastF = cv2.FastFeatureDetector_create(threshold=30)
mserF = cv2.MSER_create(10)
blobF = cv2.SimpleBlobDetector_create()
goodF = cv2.GFTTDetector_create(maxCorners= 20,minDistance = 10)

kp= fastF.detect(gray)
print('len(kp)=', len(kp))

filtered_kp = filteringByDistance(kp, 10)
print('len(filtered_kp)=', len(filtered_kp))
dst = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))   
cv2.imshow('dst',  dst)

briskF = cv2.BRISK_create()
filtered_kp, des = briskF.compute(gray, filtered_kp)
print('des.shape=', des.shape)
print('des=', des)

dst2 = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))  
for f in filtered_kp:
    x, y = f.pt
    size = f.size
    rect = ((x, y), (size, size), f.angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(dst2, [box], True, (0,255,0), 2)
    cv2.circle(dst2, (round(x), round(y)), round(f.size/2), (255,0,0), 2)
cv2.imshow('dst2',  dst2)
cv2.waitKey()
cv2.destroyAllWindows()
