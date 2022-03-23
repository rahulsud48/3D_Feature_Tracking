---
Author Rahul
---

FP.1 : Match 3D Objects
-- The approach that I have used for matching the bounding bax with previous frames is to use Intersection Over Union or simply (IoU). A separate function is created in camFusion_Student.cpp named as IoU. IoU helps in calculating the ratio of overlap of two bounding boxes.
I have also updated matchBoundingBoxes fuction in camFusion_Student.cpp

FP.2 : Compute Lidar-based TTC
-- I have updated the fucntion computeTTCLidar in camFusion_Student.cpp with the formula 
TTC = abs(minXCurr_ * dT / (minXPrev_-minXCurr_));

FP.3 : Associate Keypoint Correspondences with Bounding Boxes
--clusterKptMatchesWithROI is updated in camFusion_Student.cpp

FP.4 : Compute Camera-based TTC
-- I have updated the fucntion computeTTCCamera in camFusion_Student.cpp with the formula 
TTC = -dT / (1 - medDistRatio);

FP.5 and FP.6: Performance Evaluation 1 & 2
--Performance Analysis
1. TTC calculations for Lidar in case of using Harris Detector are inaccurate because the number of points detected is very low
2. When the number of keypoints matched and detected have high ratio, then Lidar and Camera TTC is close to each other.
3. Some combinations such as orb and brief produced spurious results in image yyc of 271 because the number of keypoints detected were too few.
