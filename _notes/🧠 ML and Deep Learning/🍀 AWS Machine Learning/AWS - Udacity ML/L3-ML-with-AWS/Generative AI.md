---
---

# Generative AI

##### New Terms

-   **Generator:** A neural network that learns to create new data resembling the source data on which it was trained.
-   **Discriminator**: A neural network trained to differentiate between real and synthetic data.
-   **Generator loss:** Measures how far the output data deviates from the real data present in the training dataset.
-   **Discriminator loss:** Evaluates how well the discriminator differentiates between real and fake data.

#### Generative AI vs discriminative models

##### Discriminative Model
==*discriminative models*== => **Aims to answer:**
	- *Given a set of data, how can I best classify this data or predict a value?*

##### Generative Model
- Generative AI is one of the biggest recent advancements in artificial intelligence => *It creates new things*
==*Generative models*== => **Aims to answer:**
1. Have I seen data like this before?
	- It can be still used for classification as it can identify wheather a given example is more similar to data labeled as a particular clas.

2. However, generative models can be used for generate new data.
	- The patterns learned in generative models can be used to create brand new examples of data which look similar to the data it seen before.

![[Captura de Pantalla 2021-10-04 a la(s) 21.36.19.png]]

## Generative AI Models
#GAN #AutoRegressiveModels #TransformerS

### Autoregressive Models
<mark style='background-color: #93EBFF !important'>*Autoregressive Convolutional Neural Networks*</mark>
- Used to study systems that:
	- Evolve over time
	- Assume that the likelihood of some data depends only on what has happen in the past.
- From weather prediction to stock prediction.

### Generative Adversial Networks (GANs)
<mark style='background-color: #FFA793 !important'>*Generative Adversial Networks*</mark>
- Involve pitting two networks agains each other to generate new content.
- The training algorithm swaps back and forth between training a ==generator network== (responsible for producing new data) and a ==*discriminator network*== (responsible for measuring how closely the generator network's data represents the training dataset).

### Transformer-based models
<mark style='background-color: #9CE684 !important'>*Trnasoformer-based models*</mark>
- Often used to study data with some sequential structure (such as the sequence of words in a sentence).
- Are now a common modern tool for model in natural language.

# Generative AI with AWS DeepComposer
- Consits of a USB keyoard that connects to the computer to input mlody and the AWS DeepComposer console
	- It includes Music studio to generate music
	- Learning capsules to dive deep into generative AI models
	- Chartbusters challenges to showcase the ML skills
- Developed to learn the fundamentals of Generative AI

### Summary
- To get started we need en input track and a trained model.
	- For the model there is a sample model
	- Or you can generate or custom your own model

#### Types of Generative AI models
1. ==GAN models==
	- -> **to create accompaniment tracks**
2. ==**AR-CNN:**== Autoregressive CNN models to 
	- -> **modify notes in the input track**
	- --> Enhance the melody
1. ==Transformers== to
	- -> **Extend the input track up to 30 seconds** 

#### Summary of the DEMO
- It was about the AWS DeepComposer console
	- Learn deep learning
	- input the music
	- Train a deep learning model to create new music

#### Chartbusters challenges
- Chartbusters is a global challenge
- There are two different challenges:
	- Melody-GO-Round -> Create new compositions
	- Melody Harvest -> Custom a generative AI model using Amazon SageMaker

## GANs with AWS DeepComposer

- ==GANs== are used to solve a creative task =>
	- Adding ==accompaniments that match the style== of an input track provided.


### What are GANs?
A GAN is a type of generative machine learning model which pits two neural networks against each other to generate new content
- A generator and a discriminator
	- <mark style='background-color: #FFA793 !important'>Generator</mark> is a neural network that learns to create new data resembling the source data on which it was trained.
		- Creates new data -> Same distribution as the date it was trained
		- Learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data
			- Can create data with an input or without it
	- <mark style='background-color: #9CE684 !important'>Discriminator</mark> is another neural network traned to differentiate between real and synthetic data.
- They are trained in *alternating cycles*

#### Training methodology
- During training the generator and discriminator work in a tight loop:

![[Captura de Pantalla 2021-10-05 a la(s) 9.57.43.png]]

1. Generator
	- Takes in a batch of single track melodies
	- Generates a batch of multi-track piano rolls as outputs
		- By adding accompaniments to each input
	- The discriminator takes the generated music and predicts how far they deviate from the real data present in the training dataset.
		- This deviation is the ==Generator loss==
	- This feedback from the discriminator is used by the generator to incrementally get better at creating realistic output.

2. Discriminator
	- As the generator gets better at createing music accompaniments it begins fooling the discriminator.
	- So the discriminator needs to be retrained as well.
		- The discriminator measures the ==discriminator loss== to evaluate how well it is differentiating between real and fake data.
	- The discriminator is a binary classifier:
		- It classifies inputs into `fake` and `real`

### GANs Loss Function
- The measure of the error in the prediction, given a set of weights, is called a loss function
	- Weights represent how important an associated feature is to determining the accuracy of a prediction
- Loss functions are important element of training a machine learning model because they are used to update the weights after every iteration of the model
- 
		
![[Captura de Pantalla 2021-10-05 a la(s) 10.06.07.png]]

## AR-CNN with AWS DeepComposer

- AR-CNN make iterative changes over time to create new data.

#### How music is represented?
1. **Input tracks:**
	- Represented as a piano roll:
	- In each two dimensional piano roll -> time is on the horizontal axis and pitch is on the vertical axis
		- pitch
2. **Edit event:**
	- Occurs when a note is either added or removed from the input track during inference

#### Training
- To train the AR-CNN model to predict when notes need to be added or removed from the input track.
	- The model iteratively updates the input track to sounds more like the training dataset
	- During training the the model is also challenged to detect differences between an original piano roll and a newly modified piano roll

## Quiz
1. **Which is the following statements is false in the context of AR-CNN?**
	- [ ] 2D images can be used to represent music
	- [x] AR-CNN generates output music iteratively over time
		- *Because it also can remove*
	- [ ] Edit event refers to a note added to the input track during inference.
	- [ ] Autoregressive models can be used to study weather forecasting.

2. **Please identify which of the following statements are true about a generative adversial network (GAN). There may be more than one correct answer.**
	- [ ] The generator and discriminator both use source data only.
	- [x] The generator learns to produce more realistic data and the discriminator learns to differentiate real data from the newly created data.
	- [x] The discriminator learns from both real Batch music and realistic Batch music
	- [ ] The generator is responsible for both creating new music and providing feedback

3. Which model is responsible for each of these roles in generative AI?
	- Discriminator --> Evaluating the output quality
	- Generator ---> Creating new ouput
	- Discriminator --> Providing feedback

4. True or false: *Loss functions help us determine when to stop training a model*
	- Yes -> When it stabilizes

## Demo: Create Music with AWS DeepComposer


#### Summary
- We need a music track to get started
	- Recorded using teh keyborard (virtual or real)
	- Input a MIDI file
- We can choose between three models:
	- AR-CNN
	- GAN
	- Transformers
- We can adjust the para maters used for each model
- We can used another generative model over a modeled melody