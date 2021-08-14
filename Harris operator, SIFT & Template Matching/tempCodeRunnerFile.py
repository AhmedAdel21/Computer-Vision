srcImg[R>0.01*R.max()]=[0,0,255]
filename = 'cat.jpg'
srcImg = imread(filename)
grayImg = rgb2gray(srcImg)
features = dola(grayImg)
result_image = srcImg
for match in features:
    result_image = cv2.circle(result_image, (match[1], match[0]), radius=0, color=(0, 0, 255), thickness=-1)
cv2.imshow("result", result_image)
cv2.waitKey(0)