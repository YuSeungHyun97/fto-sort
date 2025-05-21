**FTO-SORT: A Fast Track-id Optimizer for Enhanced Multi-Object Tracking with SORT in Unseen Pig Farm Environments**  
Yu, S., Baek, H., Son, S., Seo, J., & Chung, Y. (2025).  
Published in *Computers and Electronics in Agriculture, 237*, 110540.  
[https://doi.org/10.1016/j.compag.2025.110540](https://doi.org/10.1016/j.compag.2025.110540)

# FTO-SORT
![Video Preview](data/output.gif)
The FTO-SORT system leverages the [YOLO](https://github.com/ultralytics/ultralytics) model as its primary detector for accurate object identification and incorporates a tracking mechanism based on modifications and enhancements of the [BoT-SORT](https://arxiv.org/pdf/2206.14651) and [BoxMOT](https://github.com/mikel-brostrom/boxmot/tree/master?tab=readme-ov-file). By integrating and customizing these existing tracking frameworks, the system achieves improved detection precision and robust object tracking, tailored to the specific requirements of the application.

For evaluating tracking accuracy, the [TrackEval GitHub Repository](https://github.com/JonathonLuiten/TrackEval) was utilized.

| Method                                                        | HOTA (%) | MOTA (%) | IDF1 (%) | FLOPs (G) | FPS (TX2) |
|---------------------------------------------------------------|:--------:|:--------:|:--------:|:---------:|:---------:|
| BOT-SORT + YOLOv8n-seg                                        |   66.3   |   78.5   |   75.1   |   14.7    |    0.6    |
| BOT-SORT + YOLOv8n-seg (FBDA, OFBL) + Confidence Thresh (0.25)|   72.0   |   82.1   |   82.6   |   14.7    |    0.6    |
| + ReID remove                                                 |   72.7   |   82.3   |   84.6   |   12.0    |    0.6    |
| + Track Thresh Modification                                   |   73.7   |   83.1   |   86.7   |   12.0    |    7.2    |
| + FTO                                                         |   75.5   |   83.1   |   90.1   |   12.0    |    6.2    |
| BOT-SORT + YOLOv11n-seg                                       |   60.7   |   65.6   |   68.7   |   12.9    |    0.6    |
| BOT-SORT + YOLOv11n-seg (FBDA, OFBL) + Confidence Thresh (0.25)|  68.8   |   77.7   |   79.7   |   12.9    |    0.6    |
| + ReID remove                                                 |   70.4   |   77.5   |   81.8   |   10.2    |    7.9    |
| + Track Thresh Modification                                   |   72.6   |   78.5   |   85.6   |   10.2    |    7.9    |
| + FTO                                                         |   72.7   |   78.2   |   86.7   |   10.2    |    6.7    |



## Dataset
We provide an open dataset recorded over 2 minutes and 30 seconds in a real farm environment with 23 pigs. The dataset includes ground truth annotations for tracking purposes.  
[Download link](https://drive.google.com/file/d/1juPjNd7YySNVHjEn-bsPKRbq9TWpzmTA/view?usp=sharing)

> If you use this dataset or the associated code, please cite the following paper:  
>  
> Yu, S., Baek, H., Son, S., Seo, J., & Chung, Y. (2025).  
> *FTO-SORT: A fast track-id optimizer for enhanced multi-object tracking with SORT in unseen pig farm environments*.  
> *Computers and Electronics in Agriculture, 237*, 110540.  
> https://doi.org/10.1016/j.compag.2025.110540


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
