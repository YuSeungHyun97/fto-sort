# FTO-SORT
![Video Preview](data/output.gif)
The FTO-SORT system leverages the [YOLO](https://github.com/ultralytics/ultralytics) model as its primary detector for accurate object identification and incorporates a tracking mechanism based on modifications and enhancements of the [BoT-SORT](https://arxiv.org/pdf/2206.14651) and [BoxMOT](https://github.com/mikel-brostrom/boxmot/tree/master?tab=readme-ov-file). By integrating and customizing these existing tracking frameworks, the system achieves improved detection precision and robust object tracking, tailored to the specific requirements of the application.

For evaluating tracking accuracy, the [TrackEval GitHub Repository](https://github.com/JonathonLuiten/TrackEval) was utilized.

| Method                                        | HOTA (%) | MOTA (%) | IDF1 (%) | FPS (TX2) |
|-----------------------------------------------|----------|----------|----------|-----------|
| BOT-SORT + YOLOv8n-seg                        | 64.1     | 78.4     | 71.3     | 0.6       |
| BOT-SORT + YOLOv8n-seg (FBDA, OFBL)           | 72.0     | 82.1     | 82.6     | 0.6       |
| + Confidence Thresh (0.25)                    | 72.7     | 82.3     | 84.6     | 7.2       |
| + ReID remove                                 | 73.7     | 83.1     | 86.7     | 7.2       |
| + Track Thresh Modification                   | 75.5     | 83.1     | 90.1     | 6.2       |
| + FTO                                         | 75.5     | 83.1     | 90.1     | 6.2       |
| BOT-SORT + YOLOv11n-seg                       | 51.8     | 47.2     | 57.7     | 0.6       |
| BOT-SORT + YOLOv11n-seg (FBDA, OFBL)          | 68.8     | 77.7     | 79.7     | 0.6       |
| + Confidence Thresh (0.25)                    | 70.4     | 77.5     | 81.8     | 7.9       |
| + ReID remove                                 | 72.6     | 78.5     | 85.6     | 7.9       |
| + Track Thresh Modification                   | 72.3     | 77.0     | 86.1     | 6.7       |
| + FTO                                         | 72.3     | 77.0     | 86.1     | 6.7       |


## Dataset
We provide an open dataset recorded over 2 minutes and 30 seconds in a real farm environment with 23 pigs. The dataset includes ground truth annotations for tracking purposes.
""

## Installation
```bash
git clone https://github.com/YuSeungHyun97/fto-sort.git
pip install gdown
gdown ""
docker run --gpus '"device=0"' --ipc=host -v <your_path>:/ --rm -it -w /fto-sort tidlsld44/boxmot:1.1 /bin/bash
```

<details>
  <summary>Tracking_simple</summary>

   ```bash  
  python track_txt.py --tracking-model FTOSORT
  python scripts/run_mot_challenge.py --BENCHMARK jochiwon --SPLIT_TO_EVAL 2M30S
   ```

</details>
