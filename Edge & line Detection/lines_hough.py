from os import O_TEMPORARY
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from numpy.core.shape_base import _accumulate


def line_detection_non_vectorized(image, edge_image, num_rhos=180, num_thetas=180):
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
  #
  d = np.sqrt(np.square(edge_height) + np.square(edge_width))
  dtheta = (180 / num_thetas)
  drho = ((2 * d) / num_rhos)
  #
  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)
  #
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  #
  accumulator = np.zeros((len(rhos), len(thetas)))
  #
  output_img=np.zeros((edge_height,edge_width)) 
  #
  figure = plt.figure(figsize=(12, 12))
  subplot1 = figure.add_subplot(1, 4, 1)
  subplot1.imshow(image)
  subplot2 = figure.add_subplot(1, 4, 2)
  subplot2.imshow(edge_image, cmap="gray")
  subplot3 = figure.add_subplot(1, 4, 3)
  subplot3.set_facecolor((0, 0, 0))
  subplot4 = figure.add_subplot(1, 4, 4)
  subplot4.imshow(output_img)
  #
  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho))
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        subplot3.plot(xs, ys, color="white", alpha=0.05)
   

  out_img=image.copy()
    # for y in range(accumulator.shape[0]):
    #   for x in range(accumulator.shape[1]):
    #     if accumulator[y][x] > t_count:
    #       rho = rhos[y]
    #       theta = thetas[x]
    #       a = np.cos(np.deg2rad(theta))
    #       b = np.sin(np.deg2rad(theta))
    #       x0 = (a * rho) + edge_width_half
    #       y0 = (b * rho) + edge_height_half
    #       x1 = int(x0 + 100 * (-b))
    #       y1 = int(y0 + 100 * (a))
    #       x2 = int(x0 - 100 * (-b))
    #       y2 = int(y0 - 100 * (a))
    #       subplot3.plot([theta], [rho], marker='o', color="yellow")
    #       subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2],color="white"))
    #       out_img=cv2.line(out_img,(x1,y1),(x2,y2),(255,255,255),1) 
  indices, top_thetas, top_rhos=peak_votes(accumulator,rhos,thetas,50)
  for i in range(len(indices)):
      rho = top_rhos[i]
      theta = top_thetas[i]
      a = np.cos(np.deg2rad(theta))
      b = np.sin(np.deg2rad(theta))
      x0 = (a * rho) + edge_width_half
      y0 = (b * rho) + edge_height_half
      x1 = int(x0 + 200 * (-b))
      y1 = int(y0 + 200 * (a))
      x2 = int(x0 - 200 * (-b))
      y2 = int(y0 - 200 * (a))
      # subplot3.plot([theta], [rho], marker='o', color="yellow")
      out_img=cv2.line(out_img,(x1,y1),(x2,y2),(0,255,0),1) 


  cv2.imwrite("linesOutput2.jpg",out_img)
  # cv2.imshow("output" ,out_img)
  # cv2.imshow("input",image)
  # cv2.waitKey(0)

  # subplot1.title.set_text("Original Image")
  # subplot2.title.set_text("Edge Image")
  # subplot3.title.set_text("Hough Space")
  # subplot4.title.set_text("Detected Lines")
  # plt.show()

def peak_votes(accumulator, rhos, thetas,n):
    """ Finds the max number of votes in the hough accumulator """
    idx = np.argpartition(accumulator.flatten(), -n)[-n:]
    indices = idx[np.argsort((-accumulator.flatten())[idx])]
    top_rhos = rhos[(indices / accumulator.shape[1]).astype(int)]
    top_thetas = thetas[indices % accumulator.shape[1]]

    return indices, top_thetas, top_rhos

def hough_lines(path):
    image = cv2.imread(path)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image=cv2.GaussianBlur(edge_image,(5,5),1)
    edge_image=cv2.dilate(edge_image,(3,3),iterations=1)
    edge_image = cv2.Canny(edge_image, 100, 200)
    line_detection_non_vectorized(image, edge_image)

if __name__ == "__main__":
    hough_lines("linesInput2.jpg")


