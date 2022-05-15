# BioCAS Grand Challenge

Team Name: Always at Care
Members: Yatharth Gupta and Narendra Dhakad

Model for Task 1.1

My Model is a convolution model which has 4 convolution blocks which each have a Conv2D layer, a relu activation layer, a BatchNorm2D layer which is initialized with kaiming normal weights and zero bias.After all 4 convolution blocks it has an AdaptiveAveragePolling2D layer and has a linear layer at the end. It was trained with BCEWIthLogitsLoss and Adam Optimizer. A OneCycleLR was also used to tweak the lr.
