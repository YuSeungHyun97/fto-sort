import argparse
import cv2
import os
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT, BoTSORT, BYTETracker, HybridSORT, OCSORT, StrongSORT, ImprAssocTrack, FTOSORT

# Argument parsing
parser = argparse.ArgumentParser(description="Select tracking model")
parser.add_argument('--data_path', type=str, default='Test_data/jochiwon/', 
                    help="Input Your data.")
parser.add_argument('--save_path', type=str, default='data/trackers/mot_challenge/jochiwon-2M30S/2M30S/data/2M30S.txt', 
                    help="Save tracking txt reuslts")
parser.add_argument('--tracking-model', type=str, default='BoTSORT', 
                    choices=['BoTSORT', 'DeepOCSORT', 'BYTETracker', 'HybridSORT', 'OCSORT', 'StrongSORT', 'ImprAssocTrack', 'FTOSORT'],
                    help="Choose the tracking model to use.")
parser.add_argument('--draw', action='store_true',
                    help="Enable drawing results on images.")
parser.add_argument('--draw_results', type=str, default='results', 
                    help="draw results path.")
args = parser.parse_args()

# Select the tracker based on the command-line argument
if args.tracking_model == 'BoTSORT':
    tracker = BoTSORT(
    reid_weights=Path('weights/MSMT17/osnet_x0_25_msmt17.pt'), # which ReID model to use
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_RN50_120.pth'), # which ReID model to use, CNN-CLIP-ReID    
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID
    #reid_weights=Path('weights/MSMT17/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'), # which ReID model to use, ViT-CLIP-ReID-SIE-OLP
    with_reid = True
    #with_reid = False
)
elif args.tracking_model == 'FTOSORT':
    tracker = FTOSORT(
    reid_weights=Path('weights/MSMT17/osnet_x0_25_msmt17.pt'), # which ReID model to use    
    with_reid = False
)
elif args.tracking_model == 'DeepOCSORT':
    tracker = DeepOCSORT()
elif args.tracking_model == 'BYTETracker':
    tracker = BYTETracker()
elif args.tracking_model == 'OCSORT':
    tracker = OCSORT()
elif args.tracking_model == 'HybridSORT':
    tracker = HybridSORT(
        reid_weights=Path('weights/MSMT17/osnet_x0_25_msmt17.pt'), # which ReID model to use
        with_reid = True
    )
elif args.tracking_model == 'StrongSORT':
    tracker = StrongSORT(
        reid_weights=Path('weights/MSMT17/osnet_x0_25_msmt17.pt'), # which ReID model to use
        with_reid = True
    )
elif args.tracking_model == 'ImprAssocTrack':
    tracker = ImprAssocTrack(
        reid_weights=Path('weights/MSMT17/osnet_x0_25_msmt17.pt'), # which ReID model to use        
    )
else:
    raise ValueError(f"Unknown tracking model: {args.tracking_model}")

os.makedirs(f"{args.draw_results}/img", exist_ok=True)

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
                
                relative_coords.append([x_center, y_center, x_center + width, y_center + height, confidence, obj_class])
    
    return np.array(relative_coords)

img_dir = Path(f'{args.data_path}img1')
det_dir = Path(f'{args.data_path}dt')
tracking_results = []

image_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))

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

    if args.draw:
        tracker.plot_results(im, show_trajectories=False)        
        file_save = f"{args.draw_results}/img/{i + 1:08d}.png"
        cv2.imwrite(file_save, im)

with open(args.save_path, 'w') as file:
    for results in tracking_results:        
        for result in results:
            file.write(','.join(map(str, result)) + '\n')