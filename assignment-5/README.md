# Assignment-5 : Chunnin Exams - Whats Cookin?

[AICrowd Portal](https://www.aicrowd.com/challenges/chunin-exams-food-track-cv-2021)

Naruto preparing to clear this exams to achieve his dream of becoming the Hokage(Leader of the village) and earn everyone's respect. However, the contest requires DL and CV skills and unfortunately all he knows is to shout "dattebayo". Thats where you come in.

## Training Arc
![](https://qph.fs.quoracdn.net/main-qimg-93da9187018ebbd17d07aca8763f4283.webp)  
Help Naruto write a small image classifier in any framework of your choice (pytorch,keras,mxnet). You need to write the dataloader, the model and the training code yourself. You need to make a report with the following analysis:
- With Batch Norm
- Adding new layers
- With Dropout
- Different activation functions at the end
- Different pooling strategies
- Different optimizers
- Basic Augmentation like Rotation, Translation, Color Change

The report needs to be comprehensive and explain each design decision.Show the comparison through error plots. You may show this analysis on training on a subset of images if it is taking too long.   
**The objective of this question is to understand how to write DL code. The accuracy of the model is not important for this part.**
### Bonus
Experiment with
- Residual Blocks
- Different learning strategies
- CutMix Augmentation

## Tournament Arc
![](https://thumbs.gfycat.com/MellowWellgroomedKiskadee-size_restricted.gif)

Now Naruto is ready with his setup and is time to show who's the boss. Rip up those pretrained models and out think your competition with different training strategies and help Naruto top the leaderboard. For this you need to submit the csv file and notebook on the [AIcrowd portal]https://www.aicrowd.com/challenges/chunin-exams-food-track-cv-2021/resources) . Scoring will be based on:

+ Genin : F1 score > 0.28
+ Chunnin  : F1 score > 0.45
+ Hokage : F1 score > 0.6
+ S-Class : F1 score > 0.676

**Note: There will be strict plagirism check on the notebook. You are allowed to use any library for this part. However the notebook should have the training code.**

## Dataset
It is available [here](https://www.aicrowd.com/challenges/chunin-exams-food-track-cv-2021/dataset_files). There are approx 9233 images in train and 484 images in test set. You can use the commands given in Baseline_CV notebook to download the dataset. Using those commands you can run it on colab directly

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXHpCkoea3SmhRAjMKyHui8KC24q5T-TUQWYfnNA0qfg2Y2Tp5Lx8tU18NS7X1_0YiX5U&usqp=CAU)  
Dont Let Naruto Down. Believe it 
