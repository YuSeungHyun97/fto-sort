# FTO-SORT

The FTO-SORT system leverages the [YOLO](https://github.com/ultralytics/ultralytics) model as its primary detector for accurate object identification and incorporates a tracking mechanism based on modifications and enhancements of the [BoT-SORT](https://arxiv.org/pdf/2206.14651) and [BoxMOT](https://github.com/mikel-brostrom/boxmot/tree/master?tab=readme-ov-file). By integrating and customizing these existing tracking frameworks, the system achieves improved detection precision and robust object tracking, tailored to the specific requirements of the application.

For evaluating tracking accuracy, the [TrackEval GitHub Repository](https://github.com/JonathonLuiten/TrackEval) was utilized.

This is a brief description of my project.



<details>
  <summary>Tracking_simple</summary>

  - **Step 1:** download detection_txt ""
  - **Step 2:**
  ```bash
  python track_txt.py --tracking-model FTOSORT
  - **Step 3:** ```bash
  - python scripts/run_mot_challenge.py --BENCHMARK jochiwon --SPLIT_TO_EVAL 2M30S

  You can provide more detailed information here and even include images, links, or code examples.

</details>

<details>
  <summary>Tracking_txt</summary>

  ## Tracking Details

  Here are the details about the tracking part of the project:

  - **Step 1:** Explanation of step 1
  - **Step 2:** Explanation of step 2
  - **Step 3:** Explanation of step 3

  You can provide more detailed information here and even include images, links, or code examples.

</details>
