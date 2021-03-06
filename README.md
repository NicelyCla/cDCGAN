# cDCGAN
Pytorch implementation of cDCGAN.<br>
During my studies I had a lot of trouble finding a cDCGAN architecture that worked as I expected, so I decided to write my own version, finding an alternative way to condition it.<br />
Nothing particularly new, but it can be useful for educational purposes.<br />
This was used as a comparison with our meta-DCGAN-{1,2} MLP and meta-WGAN-gp 2MLP architectures<br />
## Architecture
![](cDCGAN.png)
## Example
![](example2.png) ![](example.png)

(could you recognize that these numbers were drawn by a machine? 😁)
## Metrics
### MNIST:
FID: 11.927998813352<br>
IS: (9.74927, 0.26216) (max 10; InceptionV3 trained with MNIST)

### CIFAR10:
FID: 35.16453<br>
IS: (3.86256, 0.14062)

--------------
See also my <a href="https://github.com/NicelyCla/cWGAN-gp">cWGAN-gp</a>
