Python Version 3.11.

Install miniconda doing:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
conda create -n python311 python=3.11
```

TO BE DONE
- https://github.com/amogkam/batch-inference-benchmarks/tree/main
- https://www.anyscale.com/blog/offline-batch-inference-comparing-ray-apache-spark-and-sagemaker
- https://docs.ray.io/en/latest/data/working-with-images.html 
