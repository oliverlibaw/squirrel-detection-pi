# Hailo Model Zoo

DeGirum maintains a comprehensive model zoo optimized for Hailo-8 and Hailo-8L devices. The models are pre-quantized and fine-tuned for efficient inference on these hardware platforms.

## Classification Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8_1 | Gender Classification  | Hailo-8          |
| yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1 | Gender Classification  | Hailo-8L         |
| yolov8s_silu_imagenet--224x224_quant_hailort_hailo8_1    | ImageNet Classification | Hailo-8          |
| yolov8s_silu_imagenet--224x224_quant_hailort_hailo8l_1   | ImageNet Classification | Hailo-8L         |

## Regression Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_age--256x256_quant_hailort_hailo8_1        | Age Prediction         | Hailo-8          |

## Detection Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_car--640x640_quant_hailort_hailo8_1        | Car Detection          | Hailo-8          |
| yolov8n_relu6_car--640x640_quant_hailort_hailo8l_1       | Car Detection          | Hailo-8L         |
| yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1       | COCO Object Detection  | Hailo-8          |
| yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1      | COCO Object Detection  | Hailo-8L         |
| yolov8n_relu6_face--640x640_quant_hailort_hailo8_1       | Face Detection         | Hailo-8          |
| yolov8n_relu6_face--640x640_quant_hailort_hailo8l_1      | Face Detection         | Hailo-8L         |
| yolov8n_relu6_fire_smoke--640x640_quant_hailort_hailo8_1 | Fire and Smoke Detection | Hailo-8        |
| yolov8n_relu6_fire_smoke--640x640_quant_hailort_hailo8l_1 | Fire and Smoke Detection | Hailo-8L       |
| yolov8n_relu6_hand--640x640_quant_hailort_hailo8_1       | Hand Detection         | Hailo-8          |
| yolov8n_relu6_hand--640x640_quant_hailort_hailo8l_1      | Hand Detection         | Hailo-8L         |
| yolov8n_relu6_human_head--640x640_quant_hailort_hailo8_1 | Human Head Detection   | Hailo-8          |
| yolov8n_relu6_human_head--640x640_quant_hailort_hailo8l_1 | Human Head Detection  | Hailo-8L         |
| yolov8n_relu6_lp--640x640_quant_hailort_hailo8_1         | License Plate Detection | Hailo-8        |
| yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1        | License Plate Detection | Hailo-8L       |
| yolov8n_relu6_person--640x640_quant_hailort_hailo8_1     | Person Detection       | Hailo-8          |
| yolov8n_relu6_ppe--640x640_quant_hailort_hailo8_1        | PPE Detection          | Hailo-8          |
| yolov8n_relu6_ppe--640x640_quant_hailort_hailo8l_1       | PPE Detection          | Hailo-8L         |
| yolov8n_silu_coco--640x640_quant_hailort_hailo8_1        | COCO Object Detection  | Hailo-8          |
| yolov8n_silu_coco--640x640_quant_hailort_hailo8l_1       | COCO Object Detection  | Hailo-8L         |

## Keypoint Detection Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1 | WiderFace with Keypoints | Hailo-8        |
| yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8l_1 | WiderFace with Keypoints | Hailo-8L       |

---

For more information about using these models and their configuration, refer to the [DeGirum Documentation](https://docs.degirum.com).

