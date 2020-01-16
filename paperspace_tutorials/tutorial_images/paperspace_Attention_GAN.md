## Attention GAN for Image generation from Text
Pytorch implementation for reproducing AttnGAN results in the paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). 
Currently, it uses the weights trained on COCO dataset.

### Running Attention GAN
+ Upload and unzip the coco.zip file in the storage folder using Jupyter notebook
+ In the coco folder modify `captions.txt` file for the desired text input
+ Start an experiment with the following parameters:
	+ **Container:** `brannondorsey/docker-stackgan`
    + **Workspace:** `https://github.com/dysdsyd/AttnGAN.git`
    + **Command:** `bash run.sh`
+ Generated images will be dumped in the `storage/coco/coco_AttnGAN2/captions` directory. Each sentence generates 5 image files and the file with `*_g2.png` extension is the final output.



