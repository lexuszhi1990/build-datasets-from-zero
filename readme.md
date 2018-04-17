build datasets in different formats(coco/pascal voc)

docker run -it --rm -p 9999:8888 -v /home/david/jupyter-notes/build-datasets-from-zero:/notebooks -v /data/david/cocoapi:/mnt/cocoapi -v /data/david/fai_kp/base_dataset:/mnt/data registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3 bash

jupyter notebook --port=8888 --allow-root
