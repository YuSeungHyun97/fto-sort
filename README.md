# FTO-SORT
![Video Preview](data/output.gif)
The FTO-SORT system leverages the [YOLO](https://github.com/ultralytics/ultralytics) model as its primary detector for accurate object identification and incorporates a tracking mechanism based on modifications and enhancements of the [BoT-SORT](https://arxiv.org/pdf/2206.14651) and [BoxMOT](https://github.com/mikel-brostrom/boxmot/tree/master?tab=readme-ov-file). By integrating and customizing these existing tracking frameworks, the system achieves improved detection precision and robust object tracking, tailored to the specific requirements of the application.

For evaluating tracking accuracy, the [TrackEval GitHub Repository](https://github.com/JonathonLuiten/TrackEval) was utilized.

## Dataset
We provide an open dataset recorded over 2 minutes and 30 seconds in a real farm environment with 23 pigs. The dataset includes ground truth annotations for tracking purposes.
""

## Installation
```bash
git clone https://github.com/YuSeungHyun97/fto-sort.git
pip install gdown
gdown "https://drive.google.com/uc?id=파일_ID"
docker run --gpus '"device=0"' --ipc=host -v <your_path>:/ --rm -it -w /fto-sort tidlsld44/boxmot:1.1 /bin/bash
```

<details>
  <summary>Tracking_simple</summary>

   ```bash  
  python track_txt.py --tracking-model FTOSORT
  python scripts/run_mot_challenge.py --BENCHMARK jochiwon --SPLIT_TO_EVAL 2M30S
   ```

</details>
