 # Sequence-Prediction-for-SiC-MOSFET-Active-Gate-Driving-with-Recurrent-Neural-Network


## 1. Introduction
In this project, an Active Gate Driver (AGD) sequence predictor is developed with recurrent neural network. Given a set of intended switching targets, such as Esw (switching loss), di/dt (for Ids) and dv/dt (for Vds), the neural network predictor is capable of generating the AGD sequence that can achieve the switching targets once it is applied to the target circuit. The AGD seuence predictor is an integral component of AGD development and is critical to take the full advantages of AGD for SiC MOSFET.

The developed neural network predictor is potentially applicable for a 'fake review generator'. Imaging a Yelp-like system, instead of giving a final score for a restaurant, the customers can give scores for #1: if the food is delicious, #2: if it is too expensive, and #3: if they will come back again. The 'fake review generator' takes the scores as teh inputs, and generates a fake review correspondingly. The difficulty of this problem is twofold:
1. review generation is the inverse problem of sentimental analysis; 
2. the three scores are not totally independent since #3 is strongly affected by both #1 and #2. 

There are evident similarities between the developed AGD sequence predictor and the 'review generator' in the sense that: 
1. the former takes switching targets as inputs and predicts AGD seuqence, while the latter takes the three review scores and generates fake reviews; 
2. the switching targets are dependent since the Esw is strongly affected by both di/dt and dv/dt, while di/dt and dv/dt are relatively independent. This dependency feature is the same for the three review scores of the review generator.

It will be interesting to develop the 'fake review generator' utilizing the neural network developed for the AGD sequence predictor.

## 2. Neural Network Structure
Inspired by the machine translation and auto-captioning applications, the AGD sequence predictor adopts the encoder-decoder structure. The encoder converts the switching targets to a context vector C, which is then decoded by the decoder to be the AGD sequence. Since the AGD sequence is essentially a time-series, GRU is adopted for the task. The figure below demonstrates the GRU based Encoder-Decoder Recurrent Neural Network (GRU-EDRNN).

![Image1](/figures/ED_RNN.png)

### 2.1 Training Data Structure
The training data is a pair of gate current Ig sequence and the corresponding switching results. Figure below shows four example Ig sequences. Note that the Ig values are between 0.1A to 2.0A, the lengths of the generated Ig sequences are different corresponding to different switching transients, and the sequences are auto-padded to the maximum length with 0s.
![Image1](/figures/data_Ig.png)


![Image1](/figures/data_SW.png)
