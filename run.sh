python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
-data /home/oester/projects/up-detr/data/ILSVRC/Data/CLS-LOC