import cv2
import os
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT, BoTSORT, BYTETracker, HybridSORT, OCSORT, StrongSORT, ImprAssocTrack, FTOSORT

#Dataset_PATH = '../Data/Tracking/jochiwon/jochiwon-131/jochiwon-134/'
Dataset_PATH = '../Data/Tracking/jochiwon/jochiwon-5M/jochiwon-5M/'
SAVE_IMAGES_PATH = 'results'
output_file = "../TrackEval-master/data/trackers/mot_challenge/jochiwon-5M/5M/data/5M.txt"
#output_file = "../TrackEval-master/data/trackers/mot_challenge/jochiwon-5M/2M30S_01/data/2M30S_01.txt"

#DRAW_REC = False
DRAW_REC = True

os.makedirs(f"{SAVE_IMAGES_PATH}/img", exist_ok=True)

def yolo_to_tensor(yolo_label_path, image_width, image_height, confidence_thresh = 0.001):
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

#tracker = BYTETracker()
#tracker = OCSORT()

tracker = FTOSORT(
    #reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_RN50_120.pth'), # which ReID model to use, CNN-CLIP-ReID    
    reid_weights=Path('weights/MSMT17/MSMT17_clipreid_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID-SIE-OLP
    #with_reid = True
    with_reid = False
)

img_dir = Path(f'{Dataset_PATH}img1')
det_dir = Path(f'{Dataset_PATH}dt')
tracking_results = []
lost_tracking_results = []
unmove_tracking_results = []

image_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))

#for img_file in sorted(img_dir.glob('*.jpg')):
#for img_file in sorted(image_files):
for i, img_file in enumerate(sorted(image_files)):
    det_path = f'{det_dir}/{img_file.stem}.txt'
    img_path = str(img_file)
    im = cv2.imread(img_path)
    height, width, _ = im.shape

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = yolo_to_tensor(det_path, width, height)

    # Check if there are any detections
    if dets.size > 0:
        if i == 0:
            tracker.update(dets, im)
            tracker.update(dets, im)
            tracker.update(dets, im)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detectionssa, make prediction ahead
    else:
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)

    tracking_results.append(tracker.save_txts(i))
    lost_tracking_temp = tracker.save_txts_lost(i)
    if lost_tracking_temp:
        lost_tracking_results.append(lost_tracking_temp)

    unmove_tracking_temp = tracker.save_txts_unmove(i)
    if unmove_tracking_temp:
        unmove_tracking_results.append(unmove_tracking_temp)

    if DRAW_REC:
        #tracker.plot_results(im, show_trajectories=False)
        tracker.plot_0_results(im, show_trajectories=False)
        file_save = f"{SAVE_IMAGES_PATH}/img/{i+1:08d}.png"

        check_1 = 0
        check_2 = 0
        for item in lost_tracking_results[:]:
            if item[0][0] == (i):          
                #print("lost_tracking", frame_id)      
                check_1 = 1
                cv2.rectangle(im, (int(item[0][2]), int(item[0][3])), (int(item[0][2] + item[0][4]), int(item[0][3] + item[0][5])), (0, 0, 255), 4)
                lost_tracking_results.remove(item)
        for item in unmove_tracking_results[:]:
            if item[0][0] == (i):              
                #print("remove_tracking", frame_id)        
                check_2 = 1
                cv2.rectangle(im, (int(item[0][2]), int(item[0][3])), (int(item[0][2] + item[0][4]), int(item[0][3] + item[0][5])), (147, 43, 160), 4)
                unmove_tracking_results.remove(item)

        # if check_1 + check_2 == 2:            
        #     cv2.imwrite(file_save, im)
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