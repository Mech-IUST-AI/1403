
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

satellite_image_path = r"C:\Users\mohammad\Downloads\tiff\test\24478855_15.tiff"  
mask_path = r"C:\Users\mohammad\Downloads\tiff\test_labels\24478855_15.tif"

satellite_image = np.array(Image.open(satellite_image_path))
mask = np.array(Image.open(mask_path))


if len(mask.shape) > 2:
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

overlayed_image = cv2.addWeighted(satellite_image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)

plt.figure(figsize=(10, 10))
plt.imshow(overlayed_image)
plt.title("Select two points on the image")
plt.show()


print("Click on two points in the image window to define the start and end points.")


points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        points.append((int(event.xdata), int(event.ydata)))
        print(f"Point selected: {points[-1]}")
        if len(points) == 2:
            plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(overlayed_image)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

start_point = points[0]
end_point = points[1]


def mask_to_graph(mask):
    h, w = mask.shape
    graph = nx.grid_2d_graph(h, w)

    
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                graph.remove_node((y, x))
    return graph

graph = mask_to_graph(mask)


start_node = (start_point[1], start_point[0])  
end_node = (end_point[1], end_point[0])

try:
    shortest_path = nx.shortest_path(graph, source=start_node, target=end_node, weight=None)
except nx.NetworkXNoPath:
    print("No path found between the selected points.")
    shortest_path = []


#if shortest_path:
 #   path_image = overlayed_image.copy()
 #   for y, x in shortest_path:
  #      path_image[y, x] = [255, 0, 0] 
#
 #   plt.figure(figsize=(10, 10))
  #  plt.imshow(path_image)
   # plt.title("Shortest Path")
    #plt.show()
    

if shortest_path:
    path_image = overlayed_image.copy()

    
    for y, x in shortest_path:
        cv2.circle(path_image, (x, y), radius=2, color=(255, 0, 0), thickness=-1) 

    plt.figure(figsize=(10, 10))
    plt.imshow(path_image)
    plt.title("Shortest Path")
    plt.show()
