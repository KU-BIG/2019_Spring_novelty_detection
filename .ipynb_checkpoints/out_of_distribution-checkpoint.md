# Out of Distribution
---
[TOC]
---
## 1. References


- [A Baseline for Detecting Misclassified and Out-of-Distribution Example in Neural Networks (2017)](https://arxiv.org/pdf/1610.02136.pdf)
- [Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks(ODIN) (2018)](https://arxiv.org/pdf/1706.02690.pdf)
- [Learning Confidence for Out-of-Distribution Detection in Neural Networks (2018)](https://arxiv.org/pdf/1802.04865.pdf)
- [Training Confidence-Calibrated Classifiers for Detecting Out-of-Distribution Samples (2018)](https://openreview.net/pdf?id=ryiAv2xAZ)


---
## 2. What is Out of Distribution?

- No Wikipedia Result
- 위키피디아 결과 없음
- First used in [A Baseline for Detecting Misclassified and Out-of-Distribution Example in Neural Networks (2017) ](https://arxiv.org/pdf/1610.02136.pdf)
- [A Baseline for Detecting Misclassified and Out-of-Distribution Example in Neural Networks (2017) ](https://arxiv.org/pdf/1610.02136.pdf) 에서 처음으로 언급됨
- The goal wanted to be achieved in the paper
- 논문에서 이루고자 했던 목표
  - Error and Success prediction
  - 에러와 성공 예측
  > &nbsp;&nbsp;Can we we predict whether a trained classifier will make an error on a particular held-out[^1] test example; can we predict if it will correctly classify said example?  
  > &nbsp;&nbsp;훈련된 예측기가 특정한 테스트 샘플에서 오류를 일으킬지 예측; 분류가 올바르게 되었다고 얘기해줄 수 있을까?
  - In- and out-of-distribution Detection
  - 샘플이 In-distribution인지 Out-of-distribution인지 판별
  > &nbsp;&nbsp;Can we predict whether  test example is from a different distribution from the training data; can we predict if it is from within the same distribution?  
  > &nbsp;&nbsp;샘플이 트레이닝 데이터와 다른 분포에서 왔는지 예측; 샘플이 트레이닝 데이터와 같은 분포 안에서 왔는지 예측할수 있을까?
  - Seen/Unseen
  - 이미 본 데이터인지/한번도 보지 못한 데이터인지

- Out of distribution means unseen data
- Out of distribution이란 트레이닝 셋에서 한번도 보지 못한 데이터를 의미함

---
## 3. Out of Distribution in Other Papers

&nbsp;&nbsp;In order to clarify the meaning of out-of-distribution, let's see how it was used in other papers.    
&nbsp;&nbsp;out-of-distribution의 의미를 더 명확하게 하기 위해, 다른 논문에서는 어떻게 사용되었는지 알아보자

>  &nbsp;&nbsp;We consider the two related problems of detecting if an example is misclassified or out-of-Distribution  
> &nbsp;&nbsp;우리는 샘플이 잘못 분류되었거나, Out-of-Distribution인 연관된 두 가지 문제를 고려합니다.  
> &nbsp;&nbsp;'A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks (2016)'


> &nbsp;&nbsp;The problem of detecting whether a test sample is from in-distribution (i.e., training distribution by a classifier) or out-of-distribution sufficiently different from it arises in many real-world machine learning applications  
> &nbsp;&nbsp;테스트 샘플이 in-distribution(예. 트레이닝 데이터의 분포)에서 왔는지 혹은 그것과 충분히 다른 out-of-distribution에서 왔는지를 판별하는 문제는 실제로 머신러닝을 적용할때 빈번하게 발생한다  
> 'Training Confidence-Calibrated Classifiers for Detecting Out-of-Distribution Samples (2018)'

> &nbsp;&nbsp;Closely related to this is the task of out-of-distribution detection, where a network must determine whether or not an input is outside of the set on which it is expected to safely performance  
> &nbsp;&nbsp;Out-of-distribution 판별과 긴밀하게 연결되어 있는 이 작업은, 네트워크가 인풋이 안전하게 예측을 수행할수 있는 셋 밖에서 왔는지 판단하는 것이다.  
> 'Learning Confidence for Out-of-Distribution Detection in Neural Networks (2018)'

> &nbsp;&nbsp;When neural networks process images which do not resemble the distribution seen during training, so called out-of-distribution images  
> &nbsp;&nbsp;신경망이 트레이닝 과정동안 본 분포를 닮지 않는 이미지, out-of-distribution 이미지  
> 'Metric Learning for Novelty and Anomaly Detection (2018)'

---

## 4. Evaluation Metric for Out of Distribution Detection

### 1) Metrics for Out-of-Distribution Detection

<center><img src='image_3.png'></center>

- ROC (Receiver Operating Characteristics)
  - **False positive rate(FPR)** versus the **true positive rate(TPR)(=Recall)** for a number of different candidate threshold values between 0.0 and 1.0
  - 0~1사이의 threshold를 변경해가면서, 그 때의 **False positive rate(FPR)** 대 **True positive rate(TPR)(=Recall)**
  - In other word, it plots the false alarm rate versus the hit rate
  - False alarm rate와 hit rate를 나타내는 것으로도 볼 수 있다
- AUC (Area Under the Curve)
  - Literally area under the ROC curve
  - ROC curve 아래의 면적
  - AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
  - AUC는 0부터 1까지의 범위를 가진다. 100% 틀린 예측을 하는 모델의 AUC는 0이고, 100% 맞는 예측을 하는 모델의 AUC는 1이다
- PRC (Precision-Recall Curve)
  - **Precision** versus the **Recall**
  - Precision 대 Recall
  - To know how good a model is at predicting the positive class
  - 모델이 positive class를 얼마나 잘 예측하는지를 보기 위함

### 2) Better metric for class-imbalanced data

- Precision captures false positive more sensitively than FPR, **thus PRC is more appropriate than ROC when it comes to class-imbalanced problem**
- Precision은 false positive를 FPR에 비해 훨씬 더 민감하게 잡아낸다. **따라서 클래스의 불균형이 있는 문제에서는 ROC보다 PRC를 보는 것이 더 낫다.**

> e.g.) 1 million samples, 100 positive and others are all negative

> case1) 100 predicted positive, 90 true positive  
> case2) 2000 predicted positive, 90 true positive  

> case1) 0.9 TPR, 0.00001 FPR  
> case2) 0.9 TPR, 0.00191 FPR
> FPR difference = 0.00190

> case1) 0.9 Recall, 0.9 Precision  
> case2) 0.9 Recall, 0.045 Precision
> Precision difference = 0.855

> Upon same false positive difference, precision shows bigger difference than FPR. In other words, precision is more sensitive to false positives  
> 같은 수준의 false positive의 차이에 대해서, precision이 FPR보다 더 큰 차이를 보인다. 다시 말하면, precision이 false positive에 대해 더 민감하게 반응한다

### 3) In papers

- AUROC
  - Area under the Receiver Operation Characteristic curve
  - ROC 아래의 면적

- AUPR In
  - Area under the Precision Recall curve
  - In-distribution examples are used as the positive class
  - PRC 아래의 면적
  - In-distribution 샘플이 positive class로 사용됨

- AUPR Out
  - Area under the precision recall curve
  - Out-of-distribution examples are used as the positive class
  - PRC 아래의 면적
  - Out-of-distribution 샘플이 positive class로 사용됨

### 4) Goals of the papers

#### 1)  Confidence Estimation

&nbsp;&nbsp;Due to the algorithm of machine learning classifier, even though model is less sure about the prediction, it anyway outputs the classification result. This could be a serious problem when it's about medical application or human safety. Thus researches are trying to give the confidence level of the machine learning prediction

&nbsp;&nbsp;머신러닝 분류기는 모델의 예측이 부정확하더라도 어떻게든 분류 결과를 내놓는다. 이것은 의료 분야에의 응용이나 인간의 안전과 관련된 분야에서 심각한 문제가 될 수 있다. 연구자들은 머신러닝 에측이 내놓는 결과에 대한 정확성을 제공하고자 한다.


#### 2)  Out-of-Distribution Detection

&nbsp;&nbsp;Machine learning classifiers have tendency to incorrectly classify test data when the training and test distributions differ. This could be one of the factors that hinders development of machine learning technology in several of fields and is of great concern to AI Safety.

&nbsp;&nbsp;머신러닝 분류기들은 트레이닝 셋과 테스트 셋의 분포가 다르면 잘못된 분류를 하는 경향이 있다. 이것은 여러 분야에서 머신러닝 기술의 발전을 저해하는 요인 중 하나이며, AI Safety의 큰 관심사중 하나이다.

#### 3) Relationship between confidence estimation and out-of-distribution detection

- Terrance DeVries and Graham W.Taylor think those two goals are closely related  
- Terrance DeVries and Graham W.Taylor는 두 목표가 밀접하게 연관되어 있다고 생각함
> &nbsp;&nbsp;'As would we expected given that we generally have less confidence in our decision  when in foreign situations'  
> &nbsp;&nbsp;'일반적으로 낯선 상황에서 내리는 선택은 낮은 신뢰성을 가지고 있을 것으로 기대된다'


---
## Recent Researches

### 1. A Baseline for Detecting Misclassified and Out-of-Distribution Example in Neural Networks

#### Baseline Model for Prediction Confidence

<center><img src='image_5.png'></center>

&nbsp;&nbsp; At the final layer of network, softmax probability is used to determine to which class the sample belongs. As a baseline model, they use maximum softmax probability as the confidence of the prediction

&nbsp;&nbsp; 네트워크의 마지막 레이어에서 가장 큰 softmax probability를 가지는 클래스가 모델이 예측한 클래스로 선택된다. 예측의 정확도를 출력하는 Baseline model으로 마지막 레이어에서의 softmax probability를 사용한다.

#### Abnormality Module

<center><img src='image_4.png'></center>

&nbsp;&nbsp; Image shows an example of abnormality module. This stems from the idea that sometimes there is information other than softmax prediction probabilities that is more useful for detection. the blue layers are auxiliary decoders. Auxiliary decoders, which are sometimes known to increase classification performance, are jointly trained on in-distribution examples with the scorer. Then the abnormality module, the red layers, are trained on clean and noised training example

&nbsp;&nbsp; 위 이미지는 Abnormality module의 예시이다. 이것은 softmax prediction probability보다 더 유용한 정보가 있을 것이라는 아이디어에서 비롯되었다. 파란색(보라색) 레이어는 auxiliary decoder이다. Auxiliary decoder는 때때로 분류 정확도를 높이는 데 도움을 주는 것으로 알려져 있다. 이 Auxiliary decoder를 in-distribution example을 이용하여 scorer와 함께 훈련시킨다. 그 후에 abnormality module(빨간색 레이어)를 clean and nosied training example로 훈련시킨다.

### 2. Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks (ODIN)

#### Key Idea

&nbsp;&nbsp;We present our method, ODIN, for detecting out-of-distribution samples. The detector is built on two components : temperature scaling and input preprocessing.

&nbsp;&nbsp;우리는 out-of-distribution 샘플들을 탐지하는데 ODIN이라는 방법을 제시한다. 그 탐지기는 두가지 요소로 구성되는데 온도 스케일링과 입력 전처리이다.

#### Temperature Scaling

Assume that the neural network $f = (f_1,...,f_N)$ is trained to classify N classes. For each input x, the neural network assigns a label $\hat{y}(x) = arg\:max_iS_i(x;T)$ by computing the softmax output for each class.  

신경망 $f = (f_1,...,f_N)$ 이 N개의 클래스들을 분류하기 위해 학습된것이라 가정하자. 각 입력값 x에 대해, 신경망은 각 클래스에 대한 소프트맥스 출력값을 계산하여 레이블 $\hat{y}(x) = arg\:max_iS_i(x;T)$  를 할당한다.  

$$ S_i (x; T) = {e^{f_i(x)/T} \over \sum_{j=1}^{N}e^{f_j(x)/T}}$$


- T is the temperature scaling parameter and set to 1 during the training.

  T는 온도 스케일링 매개변수이고 훈련중 1로 설정한다.

- For a given input x, we call the maximum softmax probability, i.e.,  $S_\hat{y}(x;T)=max_iS_i(x;T)$ the **softmax score**.

  주어진 입력값 x에 대해, 우리는 소프트맥스 확률값의 최댓값을 $S_\hat{y}(x;T)=max_iS_i(x;T)$ , 즉 **소프트맥스 점수** 라고 부른다.

- A good manipulation of temperature T can push the softmax scores of in- and out-of-distribution images further apart from each other, making the out-of-distribution images distinguishable.

  온도 T에 대한 적절한 조작은 in- , out-of-distribution 이미지들의 소프트맥스 점수들을 서로 더 차이가 나도록 할 수 있으며, 이는 out-of-distribution 이미지들을 더 분별력있게 해준다.

#### Input Preprocessing

Before feeding the image x into the neural network, we preprocess the input by adding small perturbations to it.

이미지 x를 신경망에 넣기 전에, x에 작은 섭동을 추가함으로써 인풋값을 전처리해준다.

The preprocessed image is given by

$$ \tilde{x} = x - \epsilon sign(-\nabla_xlogS_\hat{y}(x;T))$$

- where the parameter epsilon can be interpreted as the perturbation magnitude.

  전처리된 이미지는 위 식에 의해 주어지며 파라미터 epsilon은 섭동 크기로 해석할 수 있다.

- we aim to increase the softmax score of any given input, without the need for a class label at all.

  우리는 클래스 정답값을 전혀 필요로하지 않고 주어진 입력값의 소프트맥스 점수를 증가시키는것이 목표이다.

- the perturbation can have stronger effect on the in-distribution images

  그 섭동은 in-distribution 이미지들에 강한 영향을 줄 수 있다.

- the perturbations can be easily computed by back-prop the gradient of the cross-entropy loss w.r.t the input.

  그 섭동들은 인풋에 대해 크로스엔트로피 로스의 그레디언트를 역전파 계산에 의해 쉽게 구할 수 있다.

#### Out-of-distribution Detector

- For each image x, we first calculate the preprocessed image $\tilde{x}$ according to the equation.  
- 각 이미지 x에 대해, 우리는 우선 방정식에 따라 전처리된 이미지를 계산할 수 있다.  

- Next, we feed the preprocessed image x~ into the neural network, calculate its softmax score S(x~;T) and compare the score to the threshold delta.
- 다음으로, 우리는 전처리된 이미지 x~를 신경망에 넣고 소프트맥스 점수인 S(x~;T)를 계산하고 그 점수와 임계값 델타를 비교한다.

$$ g(x;\delta,T,\epsilon) = \begin{cases}
1 \quad if \: max_i p (\tilde{x};T) \leq \delta
\\
0 \quad if \: max_i p (\tilde{x};T) > \delta
\end{cases} $$

- We say that the image x is an in-distribution example if the softmax score is above the threshold and that the image x is an out-of-distribution example, otherwise.
- 우리는 소프트맥스 점수가 임계값보다 크면 이미지 x를 in-distribution 사례라하고 그렇지 않으면 out-of-distribution 사례라 한다.


- The parameters T, eps and delta are chosen so that the true positive rate (i.e., the fraction of in-distribution images correctly classified as in-distribution images) under some out-of-distribution image data set is 95%.

- T, eps, delta는 어떤 Out-of-distribution image data에서 True positive rate가 95%이도록 정해진다.

### 3. Training Confidence-Calibrated Classifiers for Detecting Out-of-Distribution Samples

#### Key Idea

 &nbsp;&nbsp;Use Original Loss + KL Divergence + GAN Loss in order to develop a novel training method for classifier so that such inference algorithms can work better.
 &nbsp;&nbsp;Original Loss에 더해 추가로 KL Divergence + GAN Loss를 이용해 inference algorithm이 더 잘 작동하게 하는 training method를 개발한다.

#### Confidence Loss

&nbsp;&nbsp; By minimizing KL Divergence additionally from the predictive distribution on out-of-distribution samples to the uniform one to give less confidence on predictions of them, in- and out-of-distributions are more separable. X from out-of-distribution will be trained by uniform distribution(zero confidence), X from in-distribution will be trained by Label-dependent probability of Y
&nbsp;&nbsp;Uniform Distribution (zero confidence)을 목표로하는 KL Divergence를 최소화하여 OOD 샘플의 Distribution 에 대해서는 Uniform한 확률로 근사시켜 ID랑 OOD랑 더 분류가 가능해지게 만든다. Out of Distribution 의 확률분포에서 나온 X들은 uniform distribution (zero confidence)으로 학습하고 In-distribution의 확률 분포에 나온 X들은 Y의 Label-dependent probability로 학습한다.

#### GAN Loss
&nbsp;&nbsp;A priori knowledge on out-of-distribution is needed but hard to know. To estimate $P_{out}$ effectively, use GAN to get sample x of $P_{out}$ as a generator model. $x$ is sample in in-distribution which is close to out-of-distribution(= Decision Boundary Sample = low density $x$ sample in $P_{in}$), in order to distinguish in-distribution and out-of-distribution. Pre-train classifier was trained only with in-distribution, hence pick $x$ sample which are close to out-of-distribution and put it in discriminator of GAN, and then train Generator to cheat it.
&nbsp;&nbsp;여기서 Confidence Loss 를 최소화 하려면 OOD 에 대한 사전 정보가 필요하지만 알기 힘들다. 따라서  $P_{out}$  에서 효과적으로 근사하기 위해,  $P_{out}$  의 샘플  $x$ 를 얻기 위한 생성모델로 GAN을 쓴다는 것이다. 이 때  $x$  는 In-distribution 중에서 OOD랑 가까운 (= Decision Boundary Sample =  $P_{in}$  에서 밀도가 낮은  $x$  샘플)  $x$  샘플로서 ID와 OOD 를 구별하기 위한 것이고, 갖고 있는 pre-trained classifier 는 In-distribution으로만 학습했기 때문에 그 중에서 OOD랑 가까운 샘플들을  $x$로 뽑아서 GAN의 Discriminator에 넣고, 또 그것을 속이기 위한 Generator 를 학습시킨다.

#### Joint Training Method
&nbsp;&nbsp; Train both alternatively, classifier and GAN improve the performance each other, GAN learns in more reliable way even if the sample is not generated explicitly.
&nbsp;&nbsp; 두 가지를 번갈아가면서 학습하면 , classifier와 GAN이 서로 성능을 향상시키고, GAN은 Explicit 하게 샘플을 생성하지 않아도 Classifier가 더 신뢰할 수 있는 방향으로 학습한다.

#### Confidence Loss Term

$$\min_{\theta}  \:\mathbb{E}_{P_{in}(\hat{x}, \hat{y})}[-logP_{\theta}(y=\hat{y}|\hat{x})]  + \:\beta\mathbb{E}_{P_{out}(x)}[KL(\mathcal{U}(y)||P_\theta(y|x))] $$


#### GAN Loss Term

$$\min_{G} \max_{D} \:\beta\mathbb{E}_{P_{G}(x)}[KL(\mathcal{U}(y)||P_\theta(y|x))] $$

$$+ \:\mathbb{E}_{P_{in}(x)}[logD(x)] + \mathbb{E}_{P_{G}(x)}[log(1-D(x))] $$


Generator: makes $G(x)$  
Discriminator : makes  $D : X\rightarrow [0,1]$ ​ which makes the probability of a target distribution

&nbsp;&nbsp;General GAN Loss term would be $P_G => P_{in}$  in order to force generator to make in-distribution sample,  
&nbsp;&nbsp;기존의 GAN Loss term 이라면 생성자가 in-distribution의 샘플을 만들게 하기 위해  $P_G => P_{in}$  과 같이 설계하겠지만,

<center><img src='image_6.png'></center>

&nbsp;&nbsp;In (a) and (c), GAN is just used to classify in-distribution, generated samples(red stars) are high density of in-distribution, thus images are clear. However, in (b) and (d), followed proposed method in the paper in order to distinguish out-of-distribution, those are low density of in-distribution and thus images are unclear
&nbsp;&nbsp;위 그림에서 (a)와 (c) 는 기존의 GAN 방법으로 단순히 In-distribution을 분류하기 위한 모델로 GAN 을 썼기 때문에, 생성된 샘플(빨간 별)들은 ID 확률 분포에서 밀도가 높은 부분에 해당하고 그에 해당하는 그림들 역시 뚜렷하다. 하지만 (b) 와 (d)는 논문에서 제안하는 방법으로 GAN을 사용하는 것이 OOD를 구별하기 위한 것이므로 샘플들이 ID의 낮은 밀도 부분에 해당하고, 해당 그림 역시 모호하게 보인다.

#### Joint Objective Loss Function

$$\min_{G} \max_{D} \min_{\theta}  \:\mathbb{E}_{P_{in}(\hat{x}, \hat{y})}[-logP_{\theta}(y=\hat{y}|\hat{x})] $$
$$ + \:\beta\mathbb{E}_{P_{G}(x)}[KL(\mathcal{U}(y)||P_\theta(y|x))] $$
$$ + \:\mathbb{E}_{P_{in}(\hat{x})}[logD(\hat{x})] + \mathbb{E}_{P_{G}(x)}[log(1-D(x))] $$

### 4. Learning Confidence for Out-of-Distribution Detection in Neural Networks

#### Key Idea

&nbsp;&nbsp;The Network needs to get a good score, however, it has chances to ask for a hint while it gets penalty for every hint. The best strategy for network is to solve problems without hint for problem network is sure, ask for hints when it's not confident rather than just failing to solve it.     
&nbsp;&nbsp;네트워크는 문제를 잘 풀어서 좋은 스코어를 받아야 하고, 확실하지 않은 문제에 대해서는 페널티를 얻고 힌트를 요청할 수 있다. 네트워크에게 있어 최선의 전략은 자신 있는 문제는 힌트 없이 풀고, 확실하지 않은 문제는 페널티를 감안하더라도 힌트를 받아서 문제를 푸는 것이다.


#### Confidence Branch
<center><img src='image_1.png'></center>

&nbsp;&nbsp; Add Confidence Branch just after the penultimate layer of any conventional feedforward architecture in parallel with the original class prediction branch.  
&nbsp;&nbsp; Feedforward 구조의 끝에서 두번째 레이어 뒤에 Prediction Branch와 평행하게 Confidence Branch를 추가한다.

<center><img src='image_2.png'></center>

$$p\prime = c \cdot p + (1-c)y$$

&nbsp;&nbsp; Use $p\prime$ for training in order to give an opportunity to ask for hints to the network. From this process, the network will learn $p$ and $c$ at the same time, which makes possible to predict and give the confidence of the prediction at the same time.  
&nbsp;&nbsp; 모델에게 힌트를 얻을 기회를 주기 위해, 학습을 위해서는 $p\prime$을 사용한다. 이 과정을 통해 네트워크는 $p\prime$과 $c$를 동시에 학습하게 된다. 이를 통해 예측을 함과 동시에 예측의 정확도를 제공하는 것이 가능하다

&nbsp;&nbsp; For inference, we use $p$ only, however, also gives $c$ in order to give the confidence of the prediction.  
&nbsp;&nbsp; 추론을 위해서는 $p$만을 사용하지만, 이때 $c$를 함께 출력해 분류에 대한 정확도를 함께 제공한다.

$$\mathcal{L}_t = -\sum\limits_{i=1}^M log(p_i\prime)y_i$$

&nbsp;&nbsp; Use negative log likelihood, but alternatives can also be used.  
&nbsp;&nbsp; 여기서는 Negative log likelihood를 사용했으나, 다른 Loss Function도 사용 가능하다.

$$\mathcal{L}_c = -log(c)$$


&nbsp;&nbsp; Without any restriction to $c$, the network will always receive the entire ground truth by choosing $c=0$. Thus we add $c$ term in the lost function in order to maximize it.  
&nbsp;&nbsp; $c$에 아무런 제한을 걸지 않으면 네트워크는 $c=0$을 내놓음으로써 Ground Truth만을 활용하게 된다. 따라서 $c$를 최대화하도록 Loss Function에 항을 넣는다.

$$\mathcal{L} = \mathcal{L}_t + \lambda\mathcal{L}_c$$

&nbsp;&nbsp; Final Loss  
&nbsp;&nbsp; 최종적인 Loss Function






[^1] : Holdout set sometimes referred to as “testing” data, the holdout subset provides a final estimate of the machine learning model’s performance after it has been trained and validated. Holdout sets should never be used to make decisions about which algorithms to use or for improving or tuning algorithms.
