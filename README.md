 # Sequence-Prediction-for-SiC-MOSFET-Active-Gate-Driving-with-Recurrent-Neural-Network


## 1. Introduction
In this project, an Active Gate Driver (AGD) sequence predictor is developed with recurrent neural network. Given a set of intended switching targets, such as Esw (switching loss), di/dt (for Ids) and dv/dt (for Vds), the neural network predictor is capable of generating the AGD sequence that can achieve the switching targets once it is applied to the target circuit. The AGD seuence predictor is an integral component of AGD development and is critical to take the full advantages of AGD for SiC MOSFET.

The developed neural network predictor is potentially applicable for a 'fake review generator'. Imaging a Yelp-like system, instead of giving a final score for a restaurant, the customers can give scores for #1: if the food is delicious, #2: if it is too expensive, and #3: if they will come back again. The 'fake review generator' takes the scores as teh inputs, and generates a fake review correspondingly. The difficulty of this problem is twofold:
1. review generation is the inverse problem of sentimental analysis; 
2. the three scores are not totally independent since #3 is strongly affected by both #1 and #2. 

There are evident similarities between the developed AGD sequence predictor and the 'review generator' in the sense that: 
1. the former takes switching targets as inputs and predicts AGD seuqence, while the latter takes the three review scores and generates fake reviews; 
2. the switching targets are dependent since the Esw is strongly affected by both di/dt and dv/dt, while di/dt and dv/dt are relatively independent. This dependency feature is the same for the three review scores of the review generator.

It is interesting to develop the 'fake review generator' utilizing the neural network developed for the AGD sequence predictor. The paper titled "Sequence Prediction for SiC MOSFET Active Gate Driving with Recurrent Neural Network" that describes this work in details is currently under review in IEEE Transactions on Power Electronics, it will be open sourced once it is accepted.

## 2. Neural Network Structure
Inspired by the machine translation and auto-captioning applications, the AGD sequence predictor adopts the encoder-decoder structure. The encoder converts the switching targets to a context vector C, which is then decoded by the decoder to be the AGD sequence. Since the AGD sequence is essentially a time-series, GRU is adopted for the task. The figure below demonstrates the GRU based Encoder-Decoder Recurrent Neural Network (GRU-EDRNN).

![Image1](/figures/ED_RNN.png)

### 2.1 Training Data Structure
The training data is a pair of gate current Ig sequence and the corresponding switching results. Figure below shows four example Ig sequences. Note that the Ig values are between 0.1A to 2.0A, the lengths of the generated Ig sequences are different corresponding to different switching transients, and the sequences are auto-padded to the maximum length with 0s.
![Image1](/figures/data_Ig.png)

Corresponding to the four example Ig sequences, there are four switching results as shown below. Using the MATLAB switching model developed, it is convenient to obtain the following switching results:
- Overshoot : the Ids overshoot during turn-on
- max. di/dt: the maximum di/dt value during the Sat-On and Sat-Off modes. Note that, there are negative values for max. di/dt since the absolute values are considered. And a negative value indicates that the max. di/dt occurs on the falling edge of Ids
- mean di/dt: the average value of di/dt on the rising edge of Ids
- max. dv/dt: the maximum dv/dt on the fall edge of Vds (absolute value)
- Esw: the switching loss of the turn-on process covering Sat-On and Sat-Off modes
- mean dv/dt-1: the average dv/dt during the Vds drop caused by commutation loop inductance
- mean dv/dt-2: the main Vds drop interval where Coss is discharged rapidly
![Image1](/figures/data_SW.png)

In this project, only Esw, mean di/dt and max. dv/dt are used.

### 2.2 Encoder
In the training stage, the Encoder takes switching results as input, while in the inferring stage, it takes user defined switching targets. The Encoder in this project is a simple fully connected layer to convert the switching results/targets vector (length=3) to the context vector (length=360). No activation function is used for the FC layer, since it achieves the lowest training/validation losses compared to ReLU, Sigmoid and Tanh.

### 2.3 Decoder
The Decoder translates the context vector to AGD sequence. In the training stage, the Ig sequences are the additional inputs, which are used to calculate the cross-entropy loss with the neural network predictions. In the inferring stage, the prediction at step k will be used as the input for step k+1 untill the <end> token has been predicted.
 
 ## Project Files
 1. 
