import cv2
import os
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT, BoTSORT, BYTETracker, HybridSORT, OCSORT, StrongSORT, ImprAssocTrack

#Dataset_PATH = '../Data/Tracking/jochiwon/jochiwon-131/jochiwon-134/'
Dataset_PATH = '../Data/Tracking/jochiwon/jochiwon-5M/jochiwon-5M/'
SAVE_IMAGES_PATH = 'results'
output_file = "../TrackEval-master/data/trackers/mot_challenge/jochiwon-5M/5M/data/5M.txt"
#output_file = "../TrackEval-master/data/trackers/mot_challenge/jochiwon-5M/2M30S_01/data/2M30S_01.txt"

#DRAW_REC = False
DRAW_REC = True

os.makedirs(f"{SAVE_IMAGES_PATH}/img", exist_ok=True)

def yolo_to_tensor(yolo_label_path, image_width, image_height, confidence_thresh = 0.0001):
#def yolo_to_tensor(yolo_label_path, image_width, image_height, confidence_thresh = 0.001):
#def yolo_to_tensor(yolo_label_path, image_width, image_height, confidence_thresh = 0.25):
    """
    Convert YOLO formatted label to tensor containing relative coordinates.

    Parameters:
    yolo_label_path (str): Path to the YOLO formatted label file.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    (class, x_center, y_center, width, height) in relative coordinates.
    """
    relative_coords = []
    if not os.path.exists(yolo_label_path):
        print("File does not exist.")
        return np.array(relative_coords)

    with open(yolo_label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:
                confidence = float(parts[5])
                if(confidence <= confidence_thresh):
                    continue
                obj_class = float(parts[0])
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                x_center = float(parts[1]) * image_width - width / 2
                y_center = float(parts[2]) * image_height - height / 2                

                #relative_coords.append([obj_class, x_center, y_center, width, height])
                relative_coords.append([x_center, y_center, x_center + width, y_center + height, confidence, obj_class])
    
    return np.array(relative_coords)


#BYTETracker, BoTSORT, OCSORT, DeepOCSORT, HybridSORT, StrongSORT, ImprAssocTrack
#tracker = BoTSORT
# tracker = OCSORT(
#     #model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     #device='cuda:0',
#     #fp16=False,
# )

# tracker = BYTETracker()
# tracker = OCSORT()

tracker = BoTSORT(
    reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_RN50_120.pth'), # which ReID model to use, CNN-CLIP-ReID    
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID-SIE-OLP
    #with_reid = True
    with_reid = False
)

img_dir = Path(f'{Dataset_PATH}img1')
det_dir = Path(f'{Dataset_PATH}dt')
tracking_results = []

image_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))

#for img_file in sorted(img_dir.glob('*.jpg')):
for img_file in sorted(image_files):
    frame_id = int(img_file.stem)
    img_path = str(img_file)
    det_path = f'{det_dir}/{frame_id:08d}.txt'
    #print("det_path :", det_path)
    im = cv2.imread(img_path)

    try:
        height, width, _ = im.shape
    except AttributeError:
        break

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = yolo_to_tensor(det_path, width, height)

    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
        
    #print("tracker :", tracker)
    tracking_results.append(tracker.save_txts(frame_id))
    if DRAW_REC:
        tracker.plot_results(im, show_trajectories=False)
        file_save = f"{SAVE_IMAGES_PATH}/img/{frame_id:08d}.png"
        cv2.imwrite(file_save, im)

# with open(output_file, 'w') as file:
#     for result in tracking_results:
#         file.write(','.join(map(str, result)) + '\n')

skip_frame = 0

with open(output_file, 'w') as file:
    for results in tracking_results:#print("tracker :", tracker)
        # skip_frame += 1
        # if skip_frame < 4:
        #     continue
        for result in results:
            file.write(','.join(map(str, result)) + '\n')