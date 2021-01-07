# Mononizing Binocular Videos

***ACM Transactions on Graphics (SIGGRAPH Asia 2020 issue), Vol. 39, No. 6, December 2020, pp. 228:1--228:16.***

[ [Project Webpage](https://wbhu.github.io/projects/Mono3D) ]    [ [arXiv](https://arxiv.org/abs/2009.01424) ]    [ [Video](https://youtu.be/rbZR_sF9B5E?list=PL3zJztb9e6XVa266_SeX0Dj66Vjy5f720) ]

Mono3D is the implementation of mono-nizing binocular videos into a regular monocular video with the stereo information implicitly encoded, such that the original binocular videos can be restored with high quality.

![teaser](imgs/teaser.jpg)



## Demo

[[ Mononized view ]](https://wbhu.github.io/projects/Mono3D/demo/demo_mononize.html)  [[ Restored left view ]](https://wbhu.github.io/projects/Mono3D/demo/demo_restoreL.html)  [[ Restored right view ]](https://wbhu.github.io/projects/Mono3D/demo/demo_restoreR.html)  



## Environment

Please refer to [env.yaml](./env.yaml).



## Dataset

We cannot release the whole 3D movie dataset due to copyright issues. But the binocular image dataset and part of the binocular video dataset used in the paper are publicly available: [[ Flickr1024 ]](https://yingqianwang.github.io/Flickr1024/) and [[ Inria ]](https://www.di.ens.fr/willow/research/stereoseg/).



### Prepare Flickr1024 for training the image version model

1. Download Flickr1024 from the website: https://yingqianwang.github.io/Flickr1024/
2. Download data list from https://drive.google.com/drive/folders/14oeXizbqTCxbmkZblt7YbWjaU2IIqNJf?usp=sharing
3. Organise the dataset as following (${DATASET is the root dir for maintaining our dataset}):

  ```
${DATASET}  
|-- Flickr1024  
|   |-- Train  
|   |-- |-- 0001_L.png  
|   |   |-- 0001_R.png
|   |   |-- 0002_L.png  
|   |   |-- 0002_R.png
|   |   |-- ...
|   |-- Validation  
|   |-- |-- 0001_L.png  
|   |   |-- 0001_R.png
|   |   |-- 0002_L.png  
|   |   |-- 0002_R.png
|   |   |-- ...
|   |-- Test 
|   |-- |-- 0001_L.png  
|   |   |-- 0001_R.png
|   |   |-- 0002_L.png  
|   |   |-- 0002_R.png
|   |   |-- ...
|   |-- list  
|   |-- |-- train.txt
|   |   |-- val.txt
|   |   |-- test.txt  
  ```



## Evaluation

- Download pretrained model from https://drive.google.com/drive/folders/14oeXizbqTCxbmkZblt7YbWjaU2IIqNJf?usp=sharing, and put the 'mono3d_img.pth.tar' inside 'Exp/model_zoo'.

- Demo on a single scene

```shell
$ PYTHONPATH=. python main/demo.py --left ./imgs/demo_L.png
```

- Evaluation on the testing set of Flickr1024

```shell
$ sh scripts/test.sh mono3d_img config/Flickr1024/mono3d_img.yaml
```



## Training

On the way ...



## Copyright and License

You are granted with the [LICENSE](./LICENSE) for both academic and commercial usages.



## Acknowledgments

Thanks [Yinqian Wang](https://yingqianwang.github.io) for releasing the great dataset, Flickr1024.



## Citation

```
@article{hu-2020-mononizing,
        author   = {Wenbo Hu and Menghan Xia and Chi-Wing Fu and Tien-Tsin Wong},
        title    = {Mononizing Binocular Videos},
        journal  = {ACM Transactions on Graphics (SIGGRAPH Asia 2020 issue)},
        month    = {December},
        year     = {2020},
        volume   = {39},
        number   = {6},
        pages    = {228:1-228:16}
    }
```

  



