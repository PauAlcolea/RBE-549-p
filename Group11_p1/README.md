## Phase 1
```
usage: Wrapper.py [-h] [--dir DIR] [-o] [-g]

options:
  -h, --help    show this help message and exit
  --dir DIR     Directory of input images
  -o, --output  Whether to save output imag∏es
  -g, --graph   Form homography graph to stitch images in optimal order. Slow, but effective for out-of-order image sets.
  ```
  ## Phase 2
  ```
usage: Wrapper.py [-h] [-p {1,2}] [--dir DIR] [-m {supervised,unsupervised}] [-g] [-w]

options:
  -h, --help            show this help message and exit
  -p, --phase {1,2}     Read Phase 1 or Phase 2 Data/ directory
  --dir DIR             directory containing images to stitch; relative to Phase#/Data, i.e. 'Train/Set1' or 'Test/unity_hall'. Assumes that
                        both phase's Data/ is in same directory as Code/
  -m, --ModelType {supervised,unsupervised}
                        Model type. Assumes supervised.pt and unsupervised.pt checkpoints are in Code/checkpoints/
  -g, --graph           Form homography graph to stitch images in optimal order. Slow, but effective for out-of-order image sets.
  -w, --warp            Apply cylindrical warp to images before stitching. Helpful for wide FOV images.
  ```