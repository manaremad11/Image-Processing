import cv2
import numpy as np


class RegionGrowingAgent:
    def __init__(self, img_path="image.png", threshold=15, render=False):
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.threshold = threshold
        self.render = render

    def click_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        seed = (y, x)
        self.region_growing(seed)

    def region_growing(self, seed):
        print('processing started')

        region = np.zeros_like(self.image)
        region[seed] = 255

        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbors.append((i, j))

        queue = [seed]
        while len(queue) > 0:
            point = queue.pop(0)

            for neighbor in neighbors:
                neighbor_point = (point[0] + neighbor[0], point[1] + neighbor[1])
                # check if neighbor is inside the image
                if neighbor_point[0] < 0 or neighbor_point[0] >= self.image.shape[0] \
                        or neighbor_point[1] < 0 or neighbor_point[1] >= self.image.shape[1]:
                    continue
                # check if neighbor is already processed
                if region[neighbor_point] != 0:
                    continue
                if abs(int(self.image[neighbor_point]) - int(self.image[point])) > self.threshold:
                    continue
                region[neighbor_point] = 255
                queue.append(neighbor_point)

            if self.render:
                cv2.imshow("Region", region)
                cv2.waitKey(1)

        cv2.imshow("Region", region)
        print('processing finished')

    def start(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.click_event)
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    agent = RegionGrowingAgent("g.jpeg", threshold=13, render=True)
    agent.start()
