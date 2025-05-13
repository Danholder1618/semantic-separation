# semantic-separation
Solving the task of semantic separation of combined images using artificial neural networks

# Dataset preparation:
* python utils/yolo_to_mask.py dataset/images/train dataset/labels/train
![img_1.png](assets/img_1.png)
* python utils/yolo_to_mask.py dataset/images/val dataset/labels/val
![img.png](assets/img.png)
And we get:
![img.png](assets/img_2.png)
* python dataset_generator/generator.py --cfg dataset/watermark.yaml
![img.png](assets/img_3.png)
And we get:

![syn_00505.png](dataset/synthetic/clean/train/syn_00505.png)
![syn_00505.png](dataset/synthetic/images/train/syn_00505.png)
![syn_00505.png](dataset/synthetic/masks/train/syn_00505.png)

# Train:
* python train_base.py 
* python train_improved.py

# Statistics
* python plot_metrics.py <path_to_statistics_in_same_dir>.csv

# API:
* docker-compose up -d --build
* Swagger UI: http://127.0.0.1:8000/docs

# Python 3.11