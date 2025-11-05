'''First try'''

# import cv2
# import numpy as np
# from PIL import Image
# import heapq

# # ---- Step 1: Load and preprocess the image ----
# image_path = "./jungleTest.png"
# image = Image.open(image_path).convert("RGB")
# image = image.resize((256, 256))        # Downscale for grid and speed
# np_image = np.array(image)

# # ---- Step 2: Convert to grid (tree = 1, open = 0) ----
# # Green pixel = tree (very simple demo approach)
# tree_mask = (
#     (np_image[:,:,1] > 90) &
#     (np_image[:,:,1] > np_image[:,:,0] + 30) &
#     (np_image[:,:,1] > np_image[:,:,2] + 30)
# ).astype(np.uint8)

# # ---- Step 3: Let user select 2 points interactively ----
# def get_points(img):
#     points = []
#     def click_event(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             points.append((y, x))  # y, x for row, col
#             cv2.circle(img, (x, y), 5, (255,0,0), -1)
#             cv2.imshow('image', img)
#     cv2.imshow('image', np_image)
#     cv2.setMouseCallback('image', click_event)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return points

# selected_points = get_points(np_image.copy())
# if len(selected_points) != 2:
#     print("Select exactly 2 points.")
#     exit(1)
# start, goal = selected_points

# # ---- Step 4: A* pathfinding ----
# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def astar(array, start, goal):
#     neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
#     close_set = set()
#     came_from = {}
#     gscore = {start:0}
#     fscore = {start:heuristic(start, goal)}
#     oheap = []
#     heapq.heappush(oheap, (fscore[start], start))
#     while oheap:
#         current = heapq.heappop(oheap)[1]
#         if current == goal:
#             data = []
#             while current in came_from:
#                 data.append(current)
#                 current = came_from[current]
#             data.append(start)
#             return data[::-1]
#         close_set.add(current)
#         for i, j in neighbors:
#             neighbor = current[0] + i, current[1] + j
#             if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
#                 tentative_g_score = gscore[current] + array[neighbor[0]][neighbor[1]]  # tree removal cost
#                 if neighbor in close_set: continue
#                 if tentative_g_score < gscore.get(neighbor, float('inf')):
#                     came_from[neighbor] = current
#                     gscore[neighbor] = tentative_g_score
#                     fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#                     heapq.heappush(oheap, (fscore[neighbor], neighbor))
#     return None

# path = astar(tree_mask, start, goal)
# if not path:
#     print("No path found!")
#     exit(1)

# # ---- Step 5: Visualize and summarize ----
# for y, x in path:
#     np_image[y, x] = [255, 0, 0]  # Mark path as blue

# cv2.imshow("Best Route", np_image)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# trees_removed = sum(tree_mask[y, x] for y, x in path)
# print("Start:", start)
# print("Goal:", goal)
# print("Trees Removed:", trees_removed)
# print("Path Length:", len(path))

'''Second Try'''

# import cv2
# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt

# # ---- Step 1: Load and preprocess the image ----
# imagePath = "./jungleTest.png"
# image = Image.open(imagePath).convert("RGB")
# image = image.resize((256, 256))
# npImage = np.array(image)

# # ---- Step 2: Convert to grid (tree = 1, open = 0) ----
# treeMask = (
#     (npImage[:, :, 1] > 90) &
#     (npImage[:, :, 1] > npImage[:, :, 0] + 30) &
#     (npImage[:, :, 1] > npImage[:, :, 2] + 30)
# ).astype(np.int32)  # convert to int32 to avoid uint8 overflow

# # ---- Step 3: Let user manually input 2 points ----
# def getPoints(img):
#     plt.imshow(img)
#     plt.title("Select 2 points: Start and Goal")
#     points = plt.ginput(2)  # select 2 points
#     plt.close()

#     if len(points) != 2:
#         print("Select exactly 2 points.")
#         exit(1)

#     # Convert float to integer pixel indices
#     points = [(int(y), int(x)) for x, y in points]
#     return points

# selectedPoints = getPoints(npImage.copy())
# start, goal = selectedPoints

# # ---- Step 4: A* pathfinding ----
# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def aStar(array, start, goal):
#     neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
#     closeSet = set()
#     cameFrom = {}
#     gScore = {start: 0}
#     fScore = {start: heuristic(start, goal)}
#     oHeap = []
#     heapq.heappush(oHeap, (fScore[start], start))

#     while oHeap:
#         current = heapq.heappop(oHeap)[1]
#         if current == goal:
#             path = []
#             while current in cameFrom:
#                 path.append(current)
#                 current = cameFrom[current]
#             path.append(start)
#             return path[::-1]

#         closeSet.add(current)
#         for i, j in neighbors:
#             neighbor = current[0] + i, current[1] + j
#             if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
#                 tentativeG = gScore[current] + int(array[neighbor[0]][neighbor[1]])
#                 if neighbor in closeSet:
#                     continue
#                 if tentativeG < gScore.get(neighbor, float('inf')):
#                     cameFrom[neighbor] = current
#                     gScore[neighbor] = tentativeG
#                     fScore[neighbor] = tentativeG + heuristic(neighbor, goal)
#                     heapq.heappush(oHeap, (fScore[neighbor], neighbor))
#     return None

# path = aStar(treeMask, start, goal)
# if not path:
#     print("No path found!")
#     exit(1)

# # ---- Step 5: Visualize result ----
# pathImage = npImage.copy()
# for y, x in path:
#     pathImage[y, x] = [255, 0, 0]  # blue path

# plt.figure(figsize=(6,6))
# plt.imshow(pathImage)
# plt.title("Best Route (Red = Trees, Blue = Path)")
# plt.axis('off')
# plt.show()

# treesRemoved = sum(treeMask[y, x] for y, x in path)
# print("Start:", start)
# print("Goal:", goal)
# print("Trees Removed:", treesRemoved)
# print("Path Length:", len(path))


'''third try'''


# import cv2
# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt

# # ---- Step 1: Load and preprocess the image ----
# imagePath = "./jungleTest.png"
# image = Image.open(imagePath).convert("RGB")
# image = image.resize((256, 256))
# npImage = np.array(image)

# # ---- Step 2: Convert to grid (tree = 1, open = 0) ----
# treeMask = (
#     (npImage[:, :, 1] > 90) &
#     (npImage[:, :, 1] > npImage[:, :, 0] + 30) &
#     (npImage[:, :, 1] > npImage[:, :, 2] + 30)
# ).astype(np.int32)  # Use int32 to prevent overflow

# # ---- Step 3: Select 2 points ----
# def getPoints(img):
#     plt.imshow(img)
#     plt.title("Select 2 points: Start and Goal")
#     points = plt.ginput(2)
#     plt.close()
#     if len(points) != 2:
#         print("Select exactly 2 points.")
#         exit(1)
#     points = [(int(y), int(x)) for x, y in points]
#     return points

# selectedPoints = getPoints(npImage.copy())
# start, goal = selectedPoints

# # ---- Step 4: A* pathfinding ----
# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def aStarAllPaths(array, start, goal, tolerance=10):
#     """
#     Returns best path and list of alternative near-optimal paths.
#     Tolerance = max additional cost allowed over best path.
#     """
#     neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
#     cameFrom = {}
#     gScore = {start: 0}
#     fScore = {start: heuristic(start, goal)}
#     openSet = [(fScore[start], start)]

#     allPaths = []
#     bestCost = float('inf')

#     while openSet:
#         current = heapq.heappop(openSet)[1]
#         currentCost = gScore[current]

#         # Found goal
#         if current == goal:
#             path = []
#             while current in cameFrom:
#                 path.append(current)
#                 current = cameFrom[current]
#             path.append(start)
#             path = path[::-1]
#             cost = sum(array[y, x] for y, x in path)
#             bestCost = min(bestCost, cost)
#             allPaths.append((cost, path))
#             continue  # keep exploring for near-optimal paths

#         # Explore neighbors
#         for dy, dx in neighbors:
#             ny, nx = current[0] + dy, current[1] + dx
#             if 0 <= ny < array.shape[0] and 0 <= nx < array.shape[1]:
#                 tentativeG = gScore[current] + int(array[ny][nx])
#                 if tentativeG < gScore.get((ny, nx), float('inf')):
#                     cameFrom[(ny, nx)] = current
#                     gScore[(ny, nx)] = tentativeG
#                     fScore[(ny, nx)] = tentativeG + heuristic((ny, nx), goal)
#                     heapq.heappush(openSet, (fScore[(ny, nx)], (ny, nx)))

#     # Filter near-best paths
#     nearOptimal = [p for c, p in allPaths if c <= bestCost + tolerance]
#     bestPath = min(allPaths, key=lambda x: x[0])[1] if allPaths else None
#     return bestPath, nearOptimal

# bestPath, allPaths = aStarAllPaths(treeMask, start, goal)

# if not bestPath:
#     print("No path found!")
#     exit(1)

# # ---- Step 5: Visualize ----
# pathImage = npImage.copy()

# # Draw all near-optimal paths in yellow
# for path in allPaths:
#     for y, x in path:
#         pathImage[y, x] = [255, 255, 0]  # yellow

# # Draw best path in red
# for y, x in bestPath:
#     pathImage[y, x] = [255, 0, 0]  # red

# plt.figure(figsize=(6,6))
# plt.imshow(pathImage)
# plt.title("Paths: Yellow = Alternatives, Red = Best")
# plt.axis('off')
# plt.show()

# # ---- Step 6: Stats ----
# treesRemoved = sum(treeMask[y, x] for y, x in bestPath)
# print("Start:", start)
# print("Goal:", goal)
# print("Best Path Length:", len(bestPath))
# print("Trees Removed (Best):", treesRemoved)
# print("Alternative Paths Found:", len(allPaths))


'''forth try'''

# import cv2
# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt

# # ---- Step 1: Load and preprocess the image ----
# imagePath = "./jungleShot.jpg"

# # Load image in original aspect ratio
# image = Image.open(imagePath).convert("RGB")

# # Optional downscale for performance but keep ratio
# maxDim = 512
# w, h = image.size
# scale = min(maxDim / w, maxDim / h, 1.0)
# if scale < 1.0:
#     newSize = (int(w * scale), int(h * scale))
#     image = image.resize(newSize, Image.Resampling.LANCZOS)

# npImage = np.array(image)

# # ---- Step 2: Convert to grid (tree = 1, open = 0) ----
# treeMask = (
#     (npImage[:, :, 1] > 90) &
#     (npImage[:, :, 1] > npImage[:, :, 0] + 30) &
#     (npImage[:, :, 1] > npImage[:, :, 2] + 30)
# ).astype(np.int32)

# # ---- Step 3: Select 2 points ----
# def getPoints(img):
#     plt.imshow(img)
#     plt.title("Select 2 points: Start and Goal")
#     points = plt.ginput(2)
#     plt.close()
#     if len(points) != 2:
#         print("Select exactly 2 points.")
#         exit(1)
#     # (x, y) → (row, col)
#     points = [(int(y), int(x)) for x, y in points]
#     return points

# selectedPoints = getPoints(npImage.copy())
# start, goal = selectedPoints

# # ---- Step 4: A* pathfinding ----
# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def aStarMulti(array, start, goal, tolerance=5):
#     """
#     Finds best path and collects alternative near-optimal paths.
#     """
#     neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
#     gScore = {start: 0}
#     fScore = {start: heuristic(start, goal)}
#     cameFrom = {}
#     openSet = [(fScore[start], start)]
#     allPaths = []
#     bestCost = float('inf')

#     visitedGoals = 0
#     maxPaths = 30  # limit exploration to prevent explosion

#     while openSet:
#         current = heapq.heappop(openSet)[1]
#         currentCost = gScore[current]

#         # Found a goal
#         if current == goal:
#             path = []
#             while current in cameFrom:
#                 path.append(current)
#                 current = cameFrom[current]
#             path.append(start)
#             path = path[::-1]
#             cost = sum(array[y, x] for y, x in path)
#             bestCost = min(bestCost, cost)
#             allPaths.append((cost, path))
#             visitedGoals += 1
#             if visitedGoals >= maxPaths:
#                 break
#             continue

#         # Explore neighbors
#         for dy, dx in neighbors:
#             ny, nx = current[0] + dy, current[1] + dx
#             if 0 <= ny < array.shape[0] and 0 <= nx < array.shape[1]:
#                 tentativeG = gScore[current] + int(array[ny][nx])
#                 if tentativeG < gScore.get((ny, nx), float('inf')):
#                     cameFrom[(ny, nx)] = current
#                     gScore[(ny, nx)] = tentativeG
#                     fScore[(ny, nx)] = tentativeG + heuristic((ny, nx), goal)
#                     heapq.heappush(openSet, (fScore[(ny, nx)], (ny, nx)))

#     # Filter near-optimal paths
#     nearOptimal = [p for c, p in allPaths if c <= bestCost + tolerance]
#     bestPath = min(allPaths, key=lambda x: x[0])[1] if allPaths else None
#     return bestPath, nearOptimal

# bestPath, allPaths = aStarMulti(treeMask, start, goal)

# if not bestPath:
#     print("No path found!")
#     exit(1)

# # ---- Step 5: Visualize ----
# pathImage = npImage.copy()

# # Draw all alternative (near-optimal) paths in yellow
# for path in allPaths:
#     for y, x in path:
#         pathImage[y, x] = [255, 255, 0]  # yellow

# # Draw best path in red
# for y, x in bestPath:
#     pathImage[y, x] = [255, 0, 0]  # red

# plt.figure(figsize=(10, int(10 * pathImage.shape[0] / pathImage.shape[1])))
# plt.imshow(pathImage)
# plt.title("Paths: Yellow = Alternatives, Red = Best")
# plt.axis('off')
# plt.show()

# # ---- Step 6: Stats ----
# treesRemoved = sum(treeMask[y, x] for y, x in bestPath)
# print("Start:", start)
# print("Goal:", goal)
# print("Best Path Length:", len(bestPath))
# print("Trees Removed (Best):", treesRemoved)
# print("Alternative Paths Found:", len(allPaths))


'''fifth try'''

# import cv2
# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # ---- Step 1: Load and preprocess the image ----
# imagePath = "./jungleShot.jpg"

# # Load with original aspect ratio
# image = Image.open(imagePath).convert("RGB")
# maxDim = 512
# w, h = image.size
# scale = min(maxDim / w, maxDim / h, 1.0)
# if scale < 1.0:
#     image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
# npImage = np.array(image)

# # ---- Step 2: Convert to grid ----
# treeMask = (
#     (npImage[:, :, 1] > 90) &
#     (npImage[:, :, 1] > npImage[:, :, 0] + 30) &
#     (npImage[:, :, 1] > npImage[:, :, 2] + 30)
# ).astype(np.int32)

# gridDisplay = np.zeros((treeMask.shape[0], treeMask.shape[1], 3), dtype=np.uint8)
# gridDisplay[treeMask == 1] = [0, 150, 0]  # green for trees
# gridDisplay[treeMask == 0] = [255, 255, 255]  # white for open land

# # ---- Step 3: Select start & goal ----
# def getPoints(img):
#     plt.imshow(img)
#     plt.title("Select 2 points: Start and Goal")
#     pts = plt.ginput(2)
#     plt.close()
#     if len(pts) != 2:
#         print("Select exactly 2 points.")
#         exit(1)
#     pts = [(int(y), int(x)) for x, y in pts]
#     return pts

# start, goal = getPoints(gridDisplay.copy())

# # ---- Step 4: A* with animation data collection ----
# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def aStarAnimated(array, start, goal):
#     neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
#     gScore = {start: 0}
#     fScore = {start: heuristic(start, goal)}
#     cameFrom = {}
#     openSet = [(fScore[start], start)]
#     visited = []
#     frames = []

#     while openSet:
#         current = heapq.heappop(openSet)[1]
#         visited.append(current)

#         # Save frame of exploration
#         frame = gridDisplay.copy()
#         for (y, x) in visited:
#             frame[y, x] = [100, 149, 237]  # blue = explored
#         frame[start[0], start[1]] = [255, 0, 255]  # magenta start
#         frame[goal[0], goal[1]] = [255, 255, 0]  # yellow goal
#         frames.append(frame)

#         if current == goal:
#             # Reconstruct path
#             path = []
#             while current in cameFrom:
#                 path.append(current)
#                 current = cameFrom[current]
#             path.append(start)
#             path = path[::-1]

#             # Add frames for final path drawing
#             for step in path:
#                 frame = frame.copy()
#                 frame[step[0], step[1]] = [255, 0, 0]  # red path
#                 frames.append(frame.copy())
#             return path, frames

#         for dy, dx in neighbors:
#             ny, nx = current[0] + dy, current[1] + dx
#             if 0 <= ny < array.shape[0] and 0 <= nx < array.shape[1]:
#                 tentativeG = gScore[current] + int(array[ny][nx])
#                 if tentativeG < gScore.get((ny, nx), float('inf')):
#                     cameFrom[(ny, nx)] = current
#                     gScore[(ny, nx)] = tentativeG
#                     fScore[(ny, nx)] = tentativeG + heuristic((ny, nx), goal)
#                     heapq.heappush(openSet, (fScore[(ny, nx)], (ny, nx)))

#     return None, frames

# bestPath, frames = aStarAnimated(treeMask, start, goal)
# if not bestPath:
#     print("No path found.")
#     exit(1)

# # ---- Step 5: Animation ----
# fig, ax = plt.subplots(figsize=(8, 8 * gridDisplay.shape[0] / gridDisplay.shape[1]))
# im = ax.imshow(frames[0])
# plt.axis('off')
# plt.title("A* Pathfinding: Blue=Explored, Red=Path")

# def update(frame):
#     im.set_data(frame)
#     return [im]

# ani = FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)
# plt.show()

# # ---- Step 6: Summary ----
# treesRemoved = sum(treeMask[y, x] for y, x in bestPath)
# print("Start:", start)
# print("Goal:", goal)
# print("Best Path Length:", len(bestPath))
# print("Trees Removed:", treesRemoved)


'''sixth try'''

# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # ---------- I/O ----------
# def load_image(path: str, max_dim: int = 900) -> np.ndarray:
#     img = Image.open(path).convert("RGB")
#     w, h = img.size
#     s = min(max_dim / w, max_dim / h, 1.0)
#     if s < 1.0:
#         try:
#             img = img.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)
#         except AttributeError:
#             img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
#     return np.array(img)

# # ---------- Vegetation mask and costs ----------
# def compute_tree_mask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
#     r = rgb[...,0].astype(np.float32)
#     g = rgb[...,1].astype(np.float32)
#     b = rgb[...,2].astype(np.float32)
#     max_rb = np.maximum(r, b)
#     exg = (g - max_rb) / (g + max_rb + 1e-6)
#     return (exg > threshold).astype(np.int32)

# def build_cost_map(tree_mask: np.ndarray, tree_cost: int = 20, open_cost: int = 1) -> np.ndarray:
#     return np.where(tree_mask == 1, tree_cost, open_cost).astype(np.int32)

# # ---------- Point picking ----------
# def pick_points(img: np.ndarray) -> tuple[tuple[int,int], tuple[int,int]]:
#     fig, ax = plt.subplots(figsize=(8, 8*img.shape[0]/img.shape[1]))
#     ax.imshow(img)
#     ax.set_title("Click START then GOAL. Close window if stuck.")
#     pts = plt.ginput(2, timeout=0)
#     plt.close(fig)
#     if len(pts) != 2:
#         raise RuntimeError("Exactly two points required.")
#     # (x,y) -> (row,col)
#     s = (int(round(pts[0][1])), int(round(pts[0][0])))
#     g = (int(round(pts[1][1])), int(round(pts[1][0])))
#     # clamp
#     h, w = img.shape[:2]
#     s = (min(max(0,s[0]), h-1), min(max(0,s[1]), w-1))
#     g = (min(max(0,g[0]), h-1), min(max(0,g[1]), w-1))
#     return s, g

# # ---------- A* driven by animation ----------
# class AStarAnimator:
#     def __init__(
#         self,
#         rgb: np.ndarray,
#         cost_map: np.ndarray,
#         start: tuple[int, int],
#         goal: tuple[int, int],
#         expansions_per_frame: int = 10,
#     ):
#         self.rgb = rgb
#         self.cost = cost_map
#         self.start = start
#         self.goal = goal
#         self.expansions_per_frame = max(1, expansions_per_frame)

#         self.nei = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         self.g = {start: 0}
#         self.f = {start: self.h(start, goal)}
#         self.came: dict[tuple[int, int], tuple[int, int]] = {}
#         self.open: list[tuple[float, tuple[int, int]]] = [(self.f[start], start)]
#         self.closed = set()
#         self.done = False
#         self.path: list[tuple[int, int]] = []
#         self.last_current: tuple[int, int] | None = None

#         # prebuilt “base” layer that highlights vegetation
#         self.base = rgb.copy()
#         mask = (cost_map > cost_map.min())
#         green = np.zeros_like(self.base)
#         green[:] = [0, 140, 0]
#         self.base[mask] = (
#             0.4 * self.base[mask].astype(np.float32)
#             + 0.6 * green[mask].astype(np.float32)
#         ).astype(np.uint8)

#         self.frame_buffer = np.empty_like(self.base)
#         self.visited_mask = np.zeros(cost_map.shape, dtype=bool)
#         self.frontier_mask = np.zeros(cost_map.shape, dtype=bool)

#     @staticmethod
#     def h(a, b):
#         return float(np.hypot(a[0] - b[0], a[1] - b[1]))

#     def _advance(self):
#         for _ in range(self.expansions_per_frame):
#             if self.done:
#                 return
#             if not self.open:
#                 self.done = True
#                 return

#             _, cur = heapq.heappop(self.open)
#             if cur in self.closed:
#                 continue

#             self.closed.add(cur)
#             self.visited_mask[cur] = True
#             self.last_current = cur

#             if cur == self.goal:
#                 path = [cur]
#                 while cur in self.came:
#                     cur = self.came[cur]
#                     path.append(cur)
#                 self.path = list(reversed(path))
#                 self.done = True
#                 return

#             gc = self.g[cur]
#             for dy, dx in self.nei:
#                 ny, nx = cur[0] + dy, cur[1] + dx
#                 if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
#                     tentative_g = gc + int(self.cost[ny, nx])
#                     node = (ny, nx)
#                     if tentative_g < self.g.get(node, float("inf")):
#                         self.came[node] = cur
#                         self.g[node] = tentative_g
#                         self.f[node] = tentative_g + self.h(node, self.goal)
#                         heapq.heappush(self.open, (self.f[node], node))

#     def step(self) -> np.ndarray:
#         self._advance()

#         np.copyto(self.frame_buffer, self.base)

#         if self.visited_mask.any():
#             self.frame_buffer[self.visited_mask] = [100, 149, 237]

#         self.frontier_mask.fill(False)
#         for _, node in self.open:
#             if node not in self.closed:
#                 self.frontier_mask[node] = True
#         if self.frontier_mask.any():
#             self.frame_buffer[self.frontier_mask] = [255, 165, 0]

#         if self.last_current is not None and not self.done:
#             cy, cx = self.last_current
#             self.frame_buffer[cy, cx] = [65, 105, 225]

#         sy, sx = self.start
#         gy, gx = self.goal
#         self.frame_buffer[sy, sx] = [255, 0, 255]
#         self.frame_buffer[gy, gx] = [255, 255, 0]

#         if self.path:
#             for py, px in self.path:
#                 self.frame_buffer[py, px] = [255, 0, 0]

#         return self.frame_buffer

# # ---------- Animate everything ----------
# def animate_search(anim: AStarAnimator, interval_ms: int = 20):
#     fig, ax = plt.subplots(figsize=(8, 8*anim.rgb.shape[0]/anim.rgb.shape[1]))
#     ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
#     ax.axis("off")
#     im = ax.imshow(anim.step())  # first frame

#     def update(_):
#         im.set_data(anim.step())
#         return [im]

#     # keep a strong reference to avoid GC killing the animation
#     ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, repeat=False)
#     plt.show()
#     return ani


# def fallback_straight_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
#     """Generate a straight path between start and goal using Bresenham's algorithm."""
#     y0, x0 = start
#     y1, x1 = goal
#     path: list[tuple[int, int]] = []

#     dx = abs(x1 - x0)
#     dy = -abs(y1 - y0)
#     sx = 1 if x0 < x1 else -1
#     sy = 1 if y0 < y1 else -1
#     err = dx + dy

#     while True:
#         path.append((y0, x0))
#         if x0 == x1 and y0 == y1:
#             break
#         e2 = 2 * err
#         if e2 >= dy:
#             err += dy
#             x0 += sx
#         if e2 <= dx:
#             err += dx
#             y0 += sy

#     return path

# # ---------- Main ----------
# if __name__ == "__main__":
#     image_path = "./jungleShot.jpg"   # change if needed
#     rgb = load_image(image_path, max_dim=900)

#     tree_mask = compute_tree_mask(rgb, threshold=0.15)
#     cost_map  = build_cost_map(tree_mask, tree_cost=20, open_cost=1)

#     # pick start/goal interactively
#     start, goal = pick_points(rgb)
#     print(f"Start: {start}, Goal: {goal}")

#     animator = AStarAnimator(rgb, cost_map, start, goal, expansions_per_frame=25)
#     _ani = animate_search(animator, interval_ms=12)

#     # After window closes, report path stats
#     if animator.path:
#         trees_crossed = int(np.sum(tree_mask[tuple(np.array(animator.path).T)]))
#         print(f"Path length: {len(animator.path)}")
#         print(f"Estimated trees crossed: {trees_crossed}")
#     else:
#         print("No viable path found with current costs. Falling back to straight traversal.")
#         direct_path = fallback_straight_path(start, goal)
#         trees_crossed = int(np.sum(tree_mask[tuple(np.array(direct_path).T)]))
#         print(f"Direct path length: {len(direct_path)}")
#         print(f"Estimated trees encountered on direct path: {trees_crossed}")


'''seventh try'''

# import numpy as np
# from PIL import Image
# import heapq
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # ---------- I/O ----------
# def load_image(path: str, max_dim: int = 900) -> np.ndarray:
#     img = Image.open(path).convert("RGB")
#     w, h = img.size
#     s = min(max_dim / w, max_dim / h, 1.0)
#     if s < 1.0:
#         try:
#             img = img.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)
#         except AttributeError:
#             img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
#     return np.array(img)

# # ---------- Vegetation mask and costs ----------
# def compute_tree_mask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
#     r = rgb[...,0].astype(np.float32)
#     g = rgb[...,1].astype(np.float32)
#     b = rgb[...,2].astype(np.float32)
#     max_rb = np.maximum(r, b)
#     exg = (g - max_rb) / (g + max_rb + 1e-6)
#     return (exg > threshold).astype(np.int32)

# def build_cost_map(tree_mask: np.ndarray, tree_cost: int = 20, open_cost: int = 1) -> np.ndarray:
#     return np.where(tree_mask == 1, tree_cost, open_cost).astype(np.int32)

# # ---------- Point picking ----------
# def pick_points(img: np.ndarray) -> tuple[tuple[int,int], tuple[int,int]]:
#     fig, ax = plt.subplots(figsize=(8, 8*img.shape[0]/img.shape[1]))
#     ax.imshow(img)
#     ax.set_title("Click START then GOAL. Close window if stuck.")
#     pts = plt.ginput(2, timeout=0)
#     plt.close(fig)
#     if len(pts) != 2:
#         raise RuntimeError("Exactly two points required.")
#     # (x,y) -> (row,col)
#     s = (int(round(pts[0][1])), int(round(pts[0][0])))
#     g = (int(round(pts[1][1])), int(round(pts[1][0])))
#     # clamp
#     h, w = img.shape[:2]
#     s = (min(max(0,s[0]), h-1), min(max(0,s[1]), w-1))
#     g = (min(max(0,g[0]), h-1), min(max(0,g[1]), w-1))
#     return s, g

# # ---------- A* driven by animation ----------
# class AStarAnimator:
#     def __init__(
#         self,
#         rgb: np.ndarray,
#         cost_map: np.ndarray,
#         start: tuple[int, int],
#         goal: tuple[int, int],
#         expansions_per_frame: int = 10,
#     ):
#         self.rgb = rgb
#         self.cost = cost_map
#         self.start = start
#         self.goal = goal
#         self.expansions_per_frame = max(1, expansions_per_frame)

#         self.nei = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#         self.g = {start: 0}
#         self.f = {start: self.h(start, goal)}
#         self.came: dict[tuple[int, int], tuple[int, int]] = {}
#         self.open: list[tuple[float, tuple[int, int]]] = [(self.f[start], start)]
#         self.closed = set()
#         self.done = False
#         self.path: list[tuple[int, int]] = []
#         self.last_current: tuple[int, int] | None = None

#         # prebuilt “base” layer that highlights vegetation
#         self.base = rgb.copy()
#         mask = (cost_map > cost_map.min())
#         green = np.zeros_like(self.base)
#         green[:] = [0, 140, 0]
#         self.base[mask] = (
#             0.4 * self.base[mask].astype(np.float32)
#             + 0.6 * green[mask].astype(np.float32)
#         ).astype(np.uint8)

#         self.frame_buffer = np.empty_like(self.base)
#         self.visited_mask = np.zeros(cost_map.shape, dtype=bool)
#         self.frontier_mask = np.zeros(cost_map.shape, dtype=bool)

#     @staticmethod
#     def h(a, b):
#         return float(np.hypot(a[0] - b[0], a[1] - b[1]))

#     def _advance(self):
#         for _ in range(self.expansions_per_frame):
#             if self.done:
#                 return
#             if not self.open:
#                 self.done = True
#                 return

#             _, cur = heapq.heappop(self.open)
#             if cur in self.closed:
#                 continue

#             self.closed.add(cur)
#             self.visited_mask[cur] = True
#             self.last_current = cur

#             if cur == self.goal:
#                 path = [cur]
#                 while cur in self.came:
#                     cur = self.came[cur]
#                     path.append(cur)
#                 self.path = list(reversed(path))
#                 self.done = True
#                 return

#             gc = self.g[cur]
#             for dy, dx in self.nei:
#                 ny, nx = cur[0] + dy, cur[1] + dx
#                 if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
#                     tentative_g = gc + int(self.cost[ny, nx])
#                     node = (ny, nx)
#                     if tentative_g < self.g.get(node, float("inf")):
#                         self.came[node] = cur
#                         self.g[node] = tentative_g
#                         self.f[node] = tentative_g + self.h(node, self.goal)
#                         heapq.heappush(self.open, (self.f[node], node))

#     def step(self) -> np.ndarray:
#         self._advance()

#         np.copyto(self.frame_buffer, self.base)

#         if self.visited_mask.any():
#             self.frame_buffer[self.visited_mask] = [100, 149, 237]

#         self.frontier_mask.fill(False)
#         for _, node in self.open:
#             if node not in self.closed:
#                 self.frontier_mask[node] = True
#         if self.frontier_mask.any():
#             self.frame_buffer[self.frontier_mask] = [255, 165, 0]

#         if self.last_current is not None and not self.done:
#             cy, cx = self.last_current
#             self.frame_buffer[cy, cx] = [65, 105, 225]

#         sy, sx = self.start
#         gy, gx = self.goal
#         self.frame_buffer[sy, sx] = [255, 0, 255]
#         self.frame_buffer[gy, gx] = [255, 255, 0]

#         if self.path:
#             for py, px in self.path:
#                 self.frame_buffer[py, px] = [255, 0, 0]

#         return self.frame_buffer

# # ---------- Animate everything ----------
# def animate_search(anim: AStarAnimator, interval_ms: int = 20):
#     fig, ax = plt.subplots(figsize=(8, 8*anim.rgb.shape[0]/anim.rgb.shape[1]))
#     ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
#     ax.axis("off")
#     im = ax.imshow(anim.step())  # first frame

#     def update(_):
#         im.set_data(anim.step())
#         return [im]

#     # keep a strong reference to avoid GC killing the animation
#     ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, repeat=False)
#     plt.show()
#     return ani


# def plot_paths_summary(anim: AStarAnimator, rgb: np.ndarray):
#     if not anim.path:
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))

#     rows = [pt[0] for pt in anim.path]
#     cols = [pt[1] for pt in anim.path]

#     axes[0].imshow(rgb)
#     axes[0].plot(cols, rows, color="red", linewidth=4, alpha=0.9)
#     axes[0].set_title("Best Path Overlay")
#     axes[0].axis("off")
#     axes[0].set_ylim(rgb.shape[0], 0)

#     axes[1].imshow(rgb)
#     explored_y, explored_x = np.where(anim.visited_mask)
#     if explored_y.size:
#         axes[1].scatter(explored_x, explored_y, s=2, c="gold", alpha=0.3, label="Explored")
#     axes[1].plot(cols, rows, color="red", linewidth=4, alpha=0.9, label="Best Path")
#     axes[1].set_title("Explored vs Best Path")
#     axes[1].axis("off")
#     axes[1].set_ylim(rgb.shape[0], 0)
#     axes[1].legend(loc="upper right")

#     plt.tight_layout()
#     plt.show()


# def fallback_straight_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
#     """Generate a straight path between start and goal using Bresenham's algorithm."""
#     y0, x0 = start
#     y1, x1 = goal
#     path: list[tuple[int, int]] = []

#     dx = abs(x1 - x0)
#     dy = -abs(y1 - y0)
#     sx = 1 if x0 < x1 else -1
#     sy = 1 if y0 < y1 else -1
#     err = dx + dy

#     while True:
#         path.append((y0, x0))
#         if x0 == x1 and y0 == y1:
#             break
#         e2 = 2 * err
#         if e2 >= dy:
#             err += dy
#             x0 += sx
#         if e2 <= dx:
#             err += dx
#             y0 += sy

#     return path

# # ---------- Main ----------
# if __name__ == "__main__":
#     image_path = "./jungleShot.jpg"   # change if needed
#     rgb = load_image(image_path, max_dim=900)

#     tree_mask = compute_tree_mask(rgb, threshold=0.15)
#     cost_map  = build_cost_map(tree_mask, tree_cost=20, open_cost=1)

#     # pick start/goal interactively
#     start, goal = pick_points(rgb)
#     print(f"Start: {start}, Goal: {goal}")

#     animator = AStarAnimator(rgb, cost_map, start, goal, expansions_per_frame=25)
#     _ani = animate_search(animator, interval_ms=12)

#     # After window closes, report path stats
#     if animator.path:
#         trees_crossed = int(np.sum(tree_mask[tuple(np.array(animator.path).T)]))
#         print(f"Path length: {len(animator.path)}")
#         print(f"Estimated trees crossed: {trees_crossed}")
#         plot_paths_summary(animator, rgb)
#     else:
#         print("No viable path found with current costs. Falling back to straight traversal.")
#         direct_path = fallback_straight_path(start, goal)
#         trees_crossed = int(np.sum(tree_mask[tuple(np.array(direct_path).T)]))
#         print(f"Direct path length: {len(direct_path)}")
#         print(f"Estimated trees encountered on direct path: {trees_crossed}")

'''eighth try'''

import numpy as np
from PIL import Image
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- I/O ----------
def load_image(path: str, max_dim: int = 900) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(max_dim / w, max_dim / h, 1.0)
    if s < 1.0:
        try:
            img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
        except AttributeError:
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return np.array(img)

# ---------- Vegetation mask and costs ----------
def compute_tree_mask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    max_rb = np.maximum(r, b)
    exg = (g - max_rb) / (g + max_rb + 1e-6)
    return (exg > threshold).astype(np.int32)

def build_cost_map(tree_mask: np.ndarray, tree_cost: int = 20, open_cost: int = 1) -> np.ndarray:
    return np.where(tree_mask == 1, tree_cost, open_cost).astype(np.int32)

# ---------- Point picking ----------
def pick_points(img: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.set_title("Click START then GOAL. Close window if stuck.")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Exactly two points required.")
    s = (int(round(pts[0][1])), int(round(pts[0][0])))
    g = (int(round(pts[1][1])), int(round(pts[1][0])))
    h, w = img.shape[:2]
    s = (min(max(0, s[0]), h - 1), min(max(0, s[1]), w - 1))
    g = (min(max(0, g[0]), h - 1), min(max(0, g[1]), w - 1))
    return s, g

# ---------- A* with animation ----------
class AStarAnimator:
    def __init__(self, rgb, cost_map, start, goal, expansions_per_frame=10):
        self.rgb = rgb
        self.cost = cost_map
        self.start = start
        self.goal = goal
        self.expansions_per_frame = max(1, expansions_per_frame)
        self.nei = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.g = {start: 0}
        self.f = {start: self.h(start, goal)}
        self.came = {}
        self.open = [(self.f[start], start)]
        self.closed = set()
        self.done = False
        self.path = []
        self.last_current = None

        self.base = rgb.copy()
        mask = (cost_map > cost_map.min())
        green = np.zeros_like(self.base)
        green[:] = [0, 140, 0]
        self.base[mask] = (
            0.4 * self.base[mask].astype(np.float32)
            + 0.6 * green[mask].astype(np.float32)
        ).astype(np.uint8)

        self.frame_buffer = np.empty_like(self.base)
        self.visited_mask = np.zeros(cost_map.shape, dtype=bool)
        self.frontier_mask = np.zeros(cost_map.shape, dtype=bool)

    @staticmethod
    def h(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _advance(self):
        for _ in range(self.expansions_per_frame):
            if self.done:
                return
            if not self.open:
                self.done = True
                return

            _, cur = heapq.heappop(self.open)
            if cur in self.closed:
                continue

            self.closed.add(cur)
            self.visited_mask[cur] = True
            self.last_current = cur

            if cur == self.goal:
                path = [cur]
                while cur in self.came:
                    cur = self.came[cur]
                    path.append(cur)
                self.path = list(reversed(path))
                self.done = True
                return

            gc = self.g[cur]
            for dy, dx in self.nei:
                ny, nx = cur[0] + dy, cur[1] + dx
                if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
                    tentative_g = gc + int(self.cost[ny, nx])
                    node = (ny, nx)
                    if tentative_g < self.g.get(node, float("inf")):
                        self.came[node] = cur
                        self.g[node] = tentative_g
                        self.f[node] = tentative_g + self.h(node, self.goal)
                        heapq.heappush(self.open, (self.f[node], node))

    def step(self):
        self._advance()
        np.copyto(self.frame_buffer, self.base)
        if self.visited_mask.any():
            self.frame_buffer[self.visited_mask] = [100, 149, 237]
        self.frontier_mask.fill(False)
        for _, node in self.open:
            if node not in self.closed:
                self.frontier_mask[node] = True
        if self.frontier_mask.any():
            self.frame_buffer[self.frontier_mask] = [255, 165, 0]
        if self.last_current is not None and not self.done:
            cy, cx = self.last_current
            self.frame_buffer[cy, cx] = [65, 105, 225]
        sy, sx = self.start
        gy, gx = self.goal
        self.frame_buffer[sy, sx] = [255, 0, 255]
        self.frame_buffer[gy, gx] = [255, 255, 0]
        if self.path:
            for py, px in self.path:
                self.frame_buffer[py, px] = [255, 0, 0]
        return self.frame_buffer

# ---------- Animate everything ----------
def animate_search(anim: AStarAnimator, interval_ms=20):
    fig, ax = plt.subplots(figsize=(8, 8 * anim.rgb.shape[0] / anim.rgb.shape[1]))
    ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
    ax.axis("off")
    im = ax.imshow(anim.step())

    def update(_):
        im.set_data(anim.step())
        return [im]

    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, repeat=False)
    plt.show()
    return ani

# ---------- Summary Visualization ----------
def plot_paths_summary(anim: AStarAnimator, rgb: np.ndarray, original: np.ndarray, cost_map: np.ndarray):
    if not anim.path:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows = [pt[0] for pt in anim.path]
    cols = [pt[1] for pt in anim.path]

    # (0,0) — Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # (0,1) — Cost Map visualization
    cost_vis = cost_map.astype(np.float32)
    axes[0, 1].imshow(cost_vis, cmap="viridis")
    axes[0, 1].set_title("Cost Map (Higher = Trees)")
    axes[0, 1].axis("off")

    # (1,0) — Best Path Overlay
    axes[1, 0].imshow(rgb)
    axes[1, 0].plot(cols, rows, color="red", linewidth=3, alpha=0.9)
    axes[1, 0].set_title("Best Path Overlay")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylim(rgb.shape[0], 0)

    # (1,1) — Explored vs Best Path
    axes[1, 1].imshow(rgb)
    explored_y, explored_x = np.where(anim.visited_mask)
    if explored_y.size:
        axes[1, 1].scatter(explored_x, explored_y, s=2, c="gold", alpha=0.3, label="Explored")
    axes[1, 1].plot(cols, rows, color="red", linewidth=3, alpha=0.9, label="Best Path")
    axes[1, 1].set_title("Explored vs Best Path")
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    axes[1, 1].legend(loc="upper right")

    # ---- Set subplot spacing ----
    fig.subplots_adjust(
        hspace=0.095,  # horizontal spacing
        wspace=0.062,  # vertical spacing
        top=0.945,
        bottom=0.015,
        left=0.010,
        right=0.990
    )

    plt.show()

# ---------- Fallback Path ----------
def fallback_straight_path(start, goal):
    y0, x0 = start
    y1, x1 = goal
    path = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        path.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return path

# ---------- Main ----------
if __name__ == "__main__":
    image_path = "./jungleShot.jpg"
    original = load_image(image_path, max_dim=900)
    rgb = original.copy()

    tree_mask = compute_tree_mask(rgb, threshold=0.15)
    cost_map = build_cost_map(tree_mask, tree_cost=20, open_cost=1)

    start, goal = pick_points(rgb)
    print(f"Start: {start}, Goal: {goal}")

    animator = AStarAnimator(rgb, cost_map, start, goal, expansions_per_frame=25)
    _ani = animate_search(animator, interval_ms=12)

    if animator.path:
        trees_crossed = int(np.sum(tree_mask[tuple(np.array(animator.path).T)]))
        print(f"Path length: {len(animator.path)}")
        print(f"Estimated trees crossed: {trees_crossed}")
        plot_paths_summary(animator, rgb, original, cost_map)
    else:
        print("No viable path found. Falling back to straight traversal.")
        direct_path = fallback_straight_path(start, goal)
        trees_crossed = int(np.sum(tree_mask[tuple(np.array(direct_path).T)]))
        print(f"Direct path length: {len(direct_path)}")
        print(f"Estimated trees encountered: {trees_crossed}")
