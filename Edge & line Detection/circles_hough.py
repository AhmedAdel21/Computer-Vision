from cv2 import cv2
import numpy as np
from collections import defaultdict

def find_hough_circles(image, edge_image, r_min=30 , r_max=200, delta_r=1, num_thetas=100, bin_threshold=.4, post_process = True):
  img_height, img_width = edge_image.shape[:2]
  
  dtheta = int(360 / num_thetas)
  thetas = np.arange(0, 360, step=dtheta)
  

  rs = np.arange(r_min, r_max, step=delta_r)
  
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  

  # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
  circle_candidates = []
  for r in rs:
    for t in range(int(num_thetas)):
      circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
  

  accumulator = defaultdict(int)
  
  for y in range(img_height):
    for x in range(img_width):
      if edge_image[y][x] != 0: #white pixel
        # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
        for r, rcos_t, rsin_t in circle_candidates:
          x_center = x - rcos_t
          y_center = y - rsin_t
          accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
  
  # Output image with detected lines drawn
  output_img = image.copy()
  # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
  out_circles = []
  
  # Sort the accumulator based on the votes for the candidate circles 
  for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = candidate_circle
    current_vote_percentage = votes / num_thetas
    if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
      out_circles.append((x, y, r, current_vote_percentage))
      
  
  # Post process the results, can add more post processing later.
  if post_process :
    pixel_threshold = 5
    postprocess_circles = []
    for x, y, r, v in out_circles:
      # Exclude circles that are too close of each other
      # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    out_circles = postprocess_circles
  
    
  # Draw shortlisted circles on the output image
  for x, y, r, v in out_circles:
    output_img = cv2.circle(output_img, (x,y), int(r), (0,255,0), 2)
  
  cv2.imwrite("CirclesOutput.jpg", output_img)
  # cv2.imshow("out",output_img)
  # cv2.waitKey(0)
def circles_hough(path):
    image = cv2.imread(path)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(edge_image, 100, 200)
    # cv2.imshow("input",edge_image)
    # cv2.waitKey(0)
    find_hough_circles(image, edge_image)

if __name__ == "__main__":
    circles_hough("CirclesInput.jpg")
