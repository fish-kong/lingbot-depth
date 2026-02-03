# -*- coding: utf-8 -*-
import cv2
import numpy as np
def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """
    Convert depth map to color visualization using OpenCV colormap.

    Args:
        depth_map (np.ndarray): Depth map (H, W)
        vmin (float): Minimum depth for colormap (auto if None)
        vmax (float): Maximum depth for colormap (auto if None)
        colormap: OpenCV colormap (TURBO, JET, VIRIDIS, etc.)

    Returns:
        np.ndarray: Colored depth map (H, W, 3) in BGR format
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Auto-range if not specified
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored
count=0
def mouse_callback(event, x, y, flags, userdata):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count+=1
        pixel_to_3d_axis = depth3d[y, x,:]
        print(f"{count}-->piont: ({x}, {y}), X Y Z: {pixel_to_3d_axis}")
        cv2.circle(mix1, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(mix1,str(count),(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
        cv2.imshow('Image', mix1)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

origdepth = np.load('result/depth_input.npy')
preddepth = np.load('result/depth_refined.npy')

rgb=cv2.imread('result/rgb.png')
depth=preddepth.copy()
matz = depth.astype('float32').copy() # unit m

para=np.loadtxt('examples/8/intrinsics.txt',dtype=np.float32)

height,width=720,1280
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y, indexing='xy')
X = X.flatten()
Y = Y.flatten()

matx2 = (X - para[0,2]) * matz[Y, X] / para[0,0]
maty2 = (Y - para[1,2]) * matz[Y, X] / para[1,1]
depth3d=np.concatenate( 
    [matx2.reshape( height ,width)[:,:,np.newaxis],
      maty2.reshape( height ,width)[:,:,np.newaxis], 
      matz[Y, X].reshape( height ,width)[:,:,np.newaxis]
      ],  2)
depth_colormap1 = depth_to_color_opencv(depth)
mix1 = cv2.addWeighted(rgb, 1, depth_colormap1, 1, 10)
cv2.imshow('Image', mix1)
key = cv2.waitKey(0)
