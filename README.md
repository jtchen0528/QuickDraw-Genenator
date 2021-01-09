# Doodle Classifier & Generator for QuickDraw!  
CDCGAN Generator and ResNet34 Classifier for QuickDraw! dataset from Google  

## Demo  
|airplane|bicycle|butterfly|cake|camera|
| :---: | :---: | :---: | :---: | :---: |
|![airplane](imgs/gif/airplane.gif)|![bicycle](imgs/gif/bicycle.gif)|![butterfly](imgs/gif/butterfly.gif)|![cake](imgs/gif/cake.gif)|![camera](imgs/gif/camera.gif)|

|chair|clock|diamond|The_Effiel_Tower|tree|
| :---: | :---: | :---: | :---: | :---: |
|![chair](imgs/gif/chair.gif)|![clock](imgs/gif/clock.gif)|![diamond](imgs/gif/diamond.gif)|![TheEffielTower](imgs/gif/TheEffielTower.gif)|![tree](imgs/gif/tree.gif)|

## Doodle Classifier
Model: ResNet34  
|| Train  | Test |
| :---: | :---: | :---: |
| Loss | ![Train Loss](imgs/classifier_train_loss.png) |  ![Test Loss](imgs/classifier_test_loss.png) |
| Accuracy |  ![Train Accu](imgs/classifier_train_accu.png) 99% | ![Test Accu](imgs/classifier_test_accu.png) 96% |
*   Prepare training data
    ```bash
    cd Classification
    python download_data.py -c categories.txt -r Data
    python ./DataUtils/prepare_data.py -root Data -msc 10000 -v 0.2
    ```
*   Start Training
    ```bash
    python Classifier.py -e 40 -bs 64 -lr 0.1 -m resnet34
    ```
* Evaluation
    ```
    python Evaluation.py -i ***.npy
    ```

## Generator
Model: DCGAN / DCCGAN
|| Discriminator Loss  | Generator Loss | Result |
| :---: | :---: | :---: | :---: |
| airplane | ![Discriminator Loss](imgs/D_Loss_DCGAN_airplane.png) |  ![Generator Loss](imgs/G_Loss_DCGAN_airplane.png) | ![airplane](imgs/airplane_DCGAN.png) |
| camera |  ![Discriminator Loss](imgs/D_Loss_DCGAN_camera.png) | ![Generator Loss](imgs/G_Loss_DCGAN_camera.png) | ![camera](imgs/camera_DCGAN.png)  |

*   Prepare training data
    ```bash
    cd Generation
    python download_data.py -c categories.txt -r Data
    ```
*   Start Training  
    1.  DCGAN
        ```bash
        python dcgan.py -o airplane -e 40 -log 1 -lr 5e-5
        ```
    2.  DCCGAN
        ```bash
        python dccgan.py -c 30 -s 50000 -e 4 -log 1 -bs 64
        ```
* Evaluation
    1.  DCGAN
        ```bash
        python Evaluation.py -r models/airplane -m DCGAN
        ```
    2.  DCCGAN
        ```bash
        python Evaluation.py -r Trained_models -m DCCGAN -c 30
        ```
## Credit
Thanks these guys.
*   [XJay18/QuickDraw-pytorch](https://github.com/XJay18/QuickDraw-pytorch)
*   [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
*   [togheppi/cDCGAN](https://github.com/togheppi/cDCGAN)
