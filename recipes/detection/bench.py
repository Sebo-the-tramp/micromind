import os
import shutil

from yolo.benchmarks import benchmark

weight = "_new_start_deeper033_3heads"

# Benchmark on GPU
benchmark(
    model="./benchmark/weights/" + weight + "/weights/best.pt",
    imgsz=320,
    half=True,
    device="cpu",
)  # microyolo with phinet

# rename benchmark.log to weight.log
os.rename("benchmarks.log", weight + ".log")

# move to benchmark folder
shutil.move(weight + ".log", "./benchmark/plots/data/optimized/" + weight + ".log")

# benchmark(model='yolov8n.pt', imgsz=320, half=False, device='cpu') # yolov8 nano
# benchmark(model="yolov8s.pt", imgsz=320, half=True, device="cpu")  # yolov8 small
