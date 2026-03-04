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
usage: Train.py [-h] [-d {lego,ship}]

options:
  -h, --help            show this help message and exit
  -d, --dataset {lego,ship}
                        dataset to train on: lego or ship
  ```