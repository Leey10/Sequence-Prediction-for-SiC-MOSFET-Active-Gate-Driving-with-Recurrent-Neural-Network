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
![Image1](/figures/data_SW.png)

- Overshoot : the Ids overshoot during turn-on
- max. di/dt: the maximum di/dt value during the Sat-On and Sat-Off modes. Note that, there are negative values for max. di/dt since the absolute values are considered. And a negative value indicates that the max. di/dt occurs on the falling edge of Ids
- mean di/dt: the average value of di/dt on the rising edge of Ids
- max. dv/dt: the maximum dv/dt on the fall edge of Vds (absolute value)
- Esw: the switching loss of the turn-on process covering Sat-On and Sat-Off modes
- mean dv/dt-1: the average dv/dt during the Vds drop caused by commutation loop inductance
- mean dv/dt-2: the main Vds drop interval where Coss is discharged rapidly


In this project, only Esw, mean di/dt and max. dv/dt are used.

### 2.2 Encoder
In the training stage, the Encoder takes switching results as input, while in the inferring stage, it takes user defined switching targets. The Encoder in this project is a simple fully connected layer to convert the switching results/targets vector (length=3) to the context vector (length=360). No activation function is used for the FC layer, since it achieves the lowest training/validation losses compared to ReLU, Sigmoid and Tanh.

### 2.3 Decoder
The Decoder translates the context vector to AGD sequence. In the training stage, the Ig sequences are the additional inputs, which are used to calculate the cross-entropy loss with the neural network predictions. In the inferring stage, the prediction at step k will be used as the input for step k+1 untill the <end> token has been predicted.
 
 ## 3. Project Files
 1. *combined_Ig.csv*, *combined_trans.csv* : are the training data as introduced in 2.1. Note that the order of the rows in the two files cannot be changed, otherwise, the Ig seuqence and the switching results are not matched.
 2. *dataset.py* : includes the *TrainDataSet* and *ValidDataSet* classes. The padded zeros in the Ig sequences are first removed, the seuqence values are then converted to integers, the <start> (0) and <end> (21) tokens are added last. The *collate_fn* is defined for the *data.Dataloader()* in *Train.py*. The function is used to pack the Ig seuqences as required for *TORCH.NN.UTILS.RNN.PACK_PADDED_SEQUENCE()*.
 3. *model.py* : defines the Encoder and Decoder structure as described above. The packed inputs to GRU are realized by the [*pack_padded_sequence()*](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch). The *sample()* method is defined for the inferring process, where the <start> token and the context vector are given to the neural network. The first prediction is then fed back to the neural network to make the second prediction until the <end> token is found.
 4. *train.py* : the training data in the .csv files are first concatenated and shuffled. 80% of the training data is randomly selected as the training set, and the rest 20% is the validation set. 
 5. *GRU_EDRNN.tar* is the trained neural network ready for inferring. *Infer.py* loads the model and generates sequence predictions once the switching targets are given. The prediction result is also saved in *AGD_Sequence.txt*.
 6. *Train-2set.py*, *combined_Ig_train.csv*, *combined_Ig_valid.csv*, *combined_trans_train.csv*, *combined_trans_valid.csv* are for further investigation. The training set and validation set as currently used in the project are randomly selected from the training data, this method can cause high validation loss if the distribution of the training set is not close to that of the validation set. The data set with *_train* and *_valid* suffix are generated by taking 80% of data from a particular distribution for training set, and have the rest 20% of data from the same distribution for validation set. By repeating the process for each distribution, both the training set and validation set cover the whole distribution plane. *Train-2set.py* uses the two manually seperated data set for the training. The performance is not as good as randomly seperating the training data, possible reasons are still under study.
