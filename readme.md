build datasets in different formats(coco/pascal voc)

docker run -it --rm -p 9999:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/cocoapi:/mnt/cocoapi -v /data/david/fai_kp/base_dataset:/mnt/data registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3 bash

docker run -it --rm -p 9999:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/cocoapi:/mnt/cocoapi -v /data/david/fai_attr:/mnt/data/fai_attr -v /data/david/fai_kp:/mnt/data/fai_kp registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3 bash

jupyter notebook --port=8888 --allow-root

visualize coco dataset:
```
docker run -it --rm -p 9999:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/cocoapi:/mnt/cocoapi registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3:latest bash
```

```
docker run -it --rm -p 6666:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/open-image-v4:/mnt/openimgv4 registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3:latest bash
```