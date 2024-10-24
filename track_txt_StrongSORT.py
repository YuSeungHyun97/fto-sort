import cv2
import os
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT, BoTSORT, BYTETracker, HybridSORT, OCSORT, StrongSORT, ImprAssocTrack

Dataset_PATH = '../Data/Tracking/jochiwon/jochiwon-5M/jochiwon-5M/'
SAVE_IMAGES_PATH = 'results'
output_file = SAVE_IMAGES_PATH + "/131.txt"

os.makedirs(f"{SAVE_IMAGES_PATH}/img", exist_ok=True)

def yolo_to_tensor(yolo_label_path, image_width, image_height, frame_id):
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
                obj_class = float(parts[0])
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                x_center = float(parts[1]) * image_width - width / 2
                y_center = float(parts[2]) * image_height - height / 2
                confidence = float(parts[5])
                # relative_coords.append([obj_class, x_center, y_center, width, height])
                # print("frame_id :", frame_id)
                relative_coords.append([x_center, y_center, x_center + width, y_center + height, confidence, obj_class, frame_id])
    
    return np.array(relative_coords)
    #return torch.tensor(relative_coords)


#BYTETracker, BoTSORT, OCSORT, DeepOCSORT, HybridSORT, StrongSORT, ImprAssocTrack
#tracker = BoTSORT
# tracker = DeepOCSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     device='cuda:0',
#     fp16=False,
# )
tracker = StrongSORT(    
    #reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    reid_weights=Path('weights/MSMT17/MSMT17_clipreid_RN50_120.pth'), # which ReID model to use, CNN-CLIP-ReID
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID-SIE-OLP
)

img_dir = Path(f'{Dataset_PATH}img1')
det_dir = Path(f'{Dataset_PATH}dt')
tracking_results = []

for img_file in sorted(img_dir.glob('*.jpg')):
    frame_id = int(img_file.stem)
    img_path = str(img_file)
    det_path = f'{det_dir}/{frame_id:08d}.txt'
    im = cv2.imread(img_path)

    try:
        height, width, _ = im.shape
    except AttributeError:
        break

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = yolo_to_tensor(det_path, width, height, frame_id)    

    # Check if there are any detections
    if dets.size > 0:
        tracking_results.append(tracker.update(dets, im)) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracking_results.append(tracker.update(dets, im)) # --> M X (x, y, x, y, id, conf, cls, ind)
        

    #print("tracker :", tracker)
    #tracking_results.append(tracker.save_txts(frame_id))    
    # file_save = f"{SAVE_IMAGES_PATH}/img/{frame_id:08d}.png"
    # cv2.imwrite(file_save, im)

# with open(output_file, 'w') as file:
#     for result in tracking_results:
#         file.write(','.join(map(str, result)) + '\n')

skip_frame = 0
with open(output_file, 'w') as file:
    for results in tracking_results:#print("tracker :", tracker)
        skip_frame += 1
        if skip_frame < 4:
            continue
        for result in results:
            file.write(','.join(map(str, result)) + '\n')