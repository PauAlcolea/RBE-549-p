## Phase 1
```
usage: Wrapper.py [-h] [-p [FLAG ...]] [-v]

options:
  -h, --help            show this help message and exit
  -p, --plot [FLAG ...]
                        Which plots to show: 
                          i=inliers, 
                          t=triangulation, 
                          r=reprojection, 
                          p=possible poses.
                          If -p is given with no flags,
                        all are shown.
  -v, --verbose         Print additional information while running
  
  ```
  ## Phase 2
  ```
usage: Wrapper.py [-h] [-d {lego,ship}] [--test] [--down DOWN] [--gif] [--frames FRAMES] [--fps FPS]

options:
  -h, --help            show this help message and exit
  -d, --dataset {lego,ship}
                        dataset to train on: lego or ship
  --test                whether to run test (default is Train)
  --down DOWN           how much you want to downscale the images so training takes less time
  --gif                 whether to render a gif
  --frames FRAMES       number of frames to render for the gif
  --fps FPS             frames per second for the output gif
  ```