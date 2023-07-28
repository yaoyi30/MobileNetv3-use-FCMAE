# MobileNetv3-use-FCMAE
Using FCMAE method to train Mobilenetv3 networks


Install Pytorch>=1.8.0, torchvision>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```


### Pretrain model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py --model mobilenet_v3_small --input_size 224 --batch_size 512 --dist_url tcp://127.0.0.1:5000 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py --model mobilenet_v3_large --input_size 224 --batch_size 512 --dist_url tcp://127.0.0.1:5000 &

```

### Finetune model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py --model mobilenet_v3_small --input_size 224 --batch_size 512 --dist_url tcp://127.0.0.1:5000 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py --model mobilenet_v3_large --input_size 224 --batch_size 512 --dist_url tcp://127.0.0.1:5000 &
```
### Datasets
Using the Mini-Imagenet dataset, there are a total of 100 categories, each with 600 images, totaling 60000 images. I divided the training set and validation set in an 8:2 ratio, with 480 training data and 120 validation data for each category, and trained 100 epochs.

### MobileNetV3
|                       | Parameters | Val Top1-acc |
| -------------------   | ---------- | --------- |
| Small (our 100 epoch) | 2950524    | 73.21%    |
| Large (our 100 epoch) | 5178732    | 78.29%    |

Reference
https://github.com/xiaolai-sqlai/mobilenetv3

https://github.com/facebookresearch/ConvNeXt-V2

