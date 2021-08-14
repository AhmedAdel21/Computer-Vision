import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def match_ssd(img, templ):
    if(len(img.shape) == 2):
        t_len, t_wid = templ.shape
        length, width = img.shape
    else:
        t_len, t_wid, _ = templ.shape
        length, width, _ = img.shape
    for i in range(length-t_len):
        for j in range(width-t_wid):
            mask = img[i:t_len+i, j:t_wid+j] 
            # print(mask.shape)
            diff = mask - templ
            ssd = np.sum(np.square(diff))
            if ssd == 0:
                result = (i, j)
                return result
    print("can't find match")

def match_normalized(img, templ):
    img = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)
    templ = cv2.cvtColor(templ, cv2.COLOR_RGB2GRAY)
    templ = templ - templ.mean()
    corr = signal.correlate2d(img, templ, boundary='symm', mode='same')
    result = np.unravel_index(np.argmax(corr), corr.shape)
    return result




main_image = cv2.imread("lena.jpg")#, cv2.IMREAD_GRAYSCALE)
main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
template_img = main_image[150:250, 200:300]

# y,x = match_ssd(main_image, template_img)
# print(x, y)
(y,x) = match_normalized(main_image, template_img)
print(x, y)
fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1, 3, 3]})
ax1, ax2, ax3 = axs
ax1.imshow(template_img)
ax1.set_axis_off()
ax1.set_title('Template image')

ax2.imshow(main_image)
ax2.set_axis_off()
ax2.set_title('Original Image')
ax3.imshow(main_image)
ax3.set_axis_off()
ax3.set_title('`SSD Match_template`\nresult')
# highlight matched region
if((len(main_image.shape) == 2)):
    height, width = template_img.shape
else:
    height, width, _ = template_img.shape
rect = plt.Rectangle((x, y), width, height, edgecolor='b', facecolor='none')
ax3.add_patch(rect)
plt.show()

