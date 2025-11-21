# CSE168AI-MLDistributed-Tensorflow-Project-Student-Naturalist-
This is a project with Amy Chan, Javier Quiroz Aldana, Ishita Prakash, and Jennifer Lugo from UC Merced's CSE168 Distributed Systems class taught by Professor Xiaoyi Lu lab part two using a custom dataset and custom tensorflow model.

The goal of this project is to recreating [iNaturalist's Computer Vision Demo](https://www.inaturalist.org/pages/computer_vision_demo) using the information provided through this website by  using instance segmentation and semantic segmentation (2 approaches) to conduct object detection and image classification of wildlife. 

We are currently working on managing to reverse engineer Merlin Bird ID's as the first step by idenitfying bird species with the support of [UC Merced's Bird Seekers and Ornithological Naturalist Group](https://ucmerced.presence.io/organization/bird-seekers-and-ornithological-naturalist-group) to support their club and other campus groups such as the Merced Vernal Pools & Natural Grassland Species Reserve through developing a web application for detecting bird species at UC Merced Campus and Merced Vernal Pools & Grassland Reserve.


Merlin EBird ID Phase 1:
* 60 images of same snowy egret
* Extract Regions of Interest: Extract the snowy egret from the image using semantic segmentation using the Unet Segmentation Tensorflow Model
* Binary Classification: Identify the regions of the image if pixel is snowy egret or not through Support Vector Machine (SVN) algorithm


Bonus Feature for possible future development...
*  Segment Everything Model for Birds (Classify all birds in an image (can have multiple species in same image) with their species


Bonus Feature for possible future development...
INaturalist Phase 2:
* Only 1 cat and same cat
* Extract Regions of Interest: Extract the cat from the image using semantic segmentation using the Unet Segmentation Tensorflow Model
* Binary Classification: Identify the regions of the image if pixel is cat or not through Support Vector Machine (SVN) algorithm
