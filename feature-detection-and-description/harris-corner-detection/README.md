
# Harris Corner Detection 
-----------

## Contents

- [GOAL](#GOAL)
- [Theory](#Theory)
- [Harris Corner Detector in OpenCV ](#Harris-Corner-Detector-in-OpenCV )
- [Corner with SubPixel Accuracy ](#Corner-with-SubPixel Accuracy )

## GOAL

In this chapter,

- We will understand the concepts behind Harris Corner Detection.
- We will see the functions: [cv.cornerHarris()](https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345), [cv.cornerSubPix()](https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e).

## Theory

In last chapter, we saw that corners are regions in the image with large variation in intensity in all the directions. One early attempt to find these corners was done by Chris Harris & Mike Stephens in their paper A Combined Corner and Edge Detector in 1988, so now it is called Harris Corner Detector. He took this simple idea to a mathematical form. It basically finds the difference in intensity for a displacement of $ (u,v) $ in all directions. This is expressed as below:

$$ E(u,v) = \sum_{x,y} \underbrace{w(x,y)}_\text{window function} \, [\underbrace{I(x+u,y+v)}_\text{shifted intensity}-\underbrace{I(x,y)}_\text{intensity}]^2 $$

Window function is either a rectangular window or gaussian window which gives weights to pixels underneath.

We have to maximize this function $ E(u,v) $ for corner detection. That means, we have to maximize the second term. Applying Taylor Expansion to above equation and using some mathematical steps (please refer any standard text books you like for full derivation), we get the final equation as:

$$ E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix} $$

where

$$ M = \sum_{x,y} w(x,y) \begin{bmatrix}I_x I_x & I_x I_y \\ I_x I_y & I_y I_y \end{bmatrix} $$

Here, $ Ix $ and $ Iy $ are image derivatives in $ x $ and $ y $ directions respectively. (Can be easily found out using [cv.Sobel()](https://docs.opencv.org/3.4.1/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)).

Then comes the main part. After this, they created a score, basically an equation, which will determine if a window can contain a corner or not.

$$ R = det(M) - k(trace(M))^2 $$

where

 - $ det(M) = \lambda_1 \lambda_2 $
 - $ trace(M) = \lambda_1 + \lambda_2 $
 - $ \lambda_1 $ and $ \lambda_1 $ are the eigen values of M
 
So the values of these eigen values decide whether a region is corner, edge or flat.

 - When $ |R| $ is small, which happens when $ \lambda_1 $ and $ \lambda_2 $ are small, the region is flat.
 - When $ R<0 $, which happens when $ \lambda_1 >> \lambda_2 $ or vice versa, the region is edge.
 - When $ R $ is large, which happens when $ \lambda_1 $ and $ \lambda_2 $ are large and $ \lambda_1 \sim \lambda_2 $, the region is a corner.

It can be represented in a nice picture as follows:

![harris-region](data/harris-region.jpg)

So the result of Harris Corner Detection is a grayscale image with these scores. Thresholding for a suitable give you the corners in the image. We will do it with a simple image.
