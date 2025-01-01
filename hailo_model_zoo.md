# Hailo Model Zoo

DeGirum maintains a comprehensive model zoo optimized for Hailo-8 and Hailo-8L devices. The models are pre-quantized and fine-tuned for efficient inference on these hardware platforms. 

DeGirum maintains a **public model zoo** at `degirum/models_hailort` on the DeGirum AI Hub. Users can access these models even without registering for the AI Hub.

DeGirum also offers the latest state-of-the-art enterprise models. You can reach out to DeGirum at this [link](https://degirum.atlassian.net/servicedesk/customer/portal/1/group/1/create/2) to request access.

Don't see the model you want in the zoo? Open a request [here](https://github.com/DeGirum/hailo_examples/issues/new?assignees=&labels=model-request&projects=&template=model_request.md&title=Model+Request%3A+%5BModel+Name%5D).

### Programmatically Listing Models

Use the following code snippet to programmatically list all models available in the Hailo Model Zoo:

```python
import degirum as dg
import degirum_tools

hailo_model_zoo = dg.connect(
    inference_host_address='@local',
    zoo_url='degirum/models_hailort'    
)

print(hailo_model_zoo.list_models())
```
The `model_name` argument in PySDK functions can take any of the values returned by the list_models function. Additionally, the `zoo_url` argument should be set to `degirum/models_hailort` to access the Hailo Model Zoo.

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
| yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1        | Age Prediction         | Hailo-8L          |

## Detection Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1       | COCO Object Detection  | Hailo-8          |
| yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1      | COCO Object Detection  | Hailo-8L         |
| yolov8n_silu_coco--640x640_quant_hailort_hailo8_1        | COCO Object Detection  | Hailo-8          |
| yolov8n_silu_coco--640x640_quant_hailort_hailo8l_1       | COCO Object Detection  | Hailo-8L         |
| yolov8n_relu6_car--640x640_quant_hailort_hailo8_1        | Car Detection          | Hailo-8          |
| yolov8n_relu6_car--640x640_quant_hailort_hailo8l_1       | Car Detection          | Hailo-8L         |
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

## Keypoint Detection Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_coco_pose--640x640_quant_hailort_hailo8_1 | Pose Detection | Hailo-8        |
| yolov8n_relu6_coco_pose--640x640_quant_hailort_hailo8l_1| Pose Detection | Hailo-8L       |
| yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1 | Face with Keypoints | Hailo-8        |
| yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8l_1 | Face with Keypoints | Hailo-8L       |

## Instance Segmentation Models

| Model Name                                     | Application              | Supported Device |
|-----------------------------------------------|--------------------------|------------------|
| yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8_1 | Instance Segmentation | Hailo-8        |
| yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8l_1 | Instance Segmentation | Hailo-8L       |

---

For more information about using these models and their configuration, refer to the [DeGirum Documentation](https://docs.degirum.com).

