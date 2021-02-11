## Deepfake detection 보안성 향상을 위한 dataset 구축

탐지기술의 강인성과 보안성 확인을 위한 고난도 변형 데이터베이스 구축을 위해 탐지 오류를 유발하는 adversarial data를 생성한다.

### Step1. 동영상을 이미지로 나타내기
_video_2image.py_

### Step2. 얼굴 crop하기
_extract_face.py_

### Step3. Adversarial attack적용하기
_AE_method_attack_img.py_

##### Adversarial Networks
- 적대적으로 경쟁하는 생성기와 판별기를 통해 진본데이터와 매우 유사한 위조데이터를 생성하여 현실에 없는 새로운 데이터 생성, 새로운 형태로 데이터 변환, 데이터 품질 향상 등 새로운 기회 가능성 제시할 수 있게 한다.

##### Adversarial Attack
- 적대적 공격은 머신러닝이 스스로 잘못된 판단을 하도록 유도하는 방식의 공격이다. 입력 데이터를 잘못 예측하도록 데이터를 조작하여 조금씩 이미지의 픽셀을 수정한다.
- 가장 대표적으로 사용되는 공격(attack) FGSM과 이를 기반으로 파생된 공격들을 적용한다.


참고: [Adversarial-Attacks-PyTorch](https://github.com/Harry24k/adversarial-attacks-pytorch)

###### FGSM(Fast Gradient Sign Method)
> 신경망의 그래디언트(gradient)를 이용해 적대적 샘플을 생성하는 기법이다. 입력 이미지에 대한 손실 함수의 그래디언트를 계산하여 그 손실을 최대화하는 이미지를 생성한다.


###### PGD(Projected gradient descent)
> 하나의 단계로 이루어지는 FGSM과 달리 여러개의 스텝으로 쪼개져 적은 변형으로 더 강력한 효과를 내는 기법이다.


###### BIM(Basic Iteration Method)
> FGSM을 반복적으로 수행하는 기법이다.


```python
method = {
	'0_FGSM':[2,5,8],
	'1_PGD':[20,50,80,2],
	'2_BIM':[4,7,10,1]
                }

for eps in range(3):
	globals()['atk{}'.format(0)] = torchattacks.FGSM(model, eps=method['0_FGSM'][eps] / 255)
	globals()['atk{}'.format(1)] = torchattacks.PGD(model, eps=method['1_PGD'][eps] / 255, alpha=method['1_PGD'][-1] / 255, steps=4)
	globals()['atk{}'.format(2)] = torchattacks.BIM(model, eps=method['2_BIM'][eps] / 255, alpha=method['2_BIM'][-1] / 255)

	for count in range(3):
		globals()['data_atk{}'.format(count)] = globals()['atk{}'.format(count)](data, (target + 1) % 2)
        globals()['data_atk{}'.format(count)], target = (globals()['data_atk{}'.format(count)]).to(device), target.to(device)
        logits = model(globals()['data_atk{}'.format(count)])
        globals()['data_atk{}'.format(count)] = (globals()['data_atk{}'.format(count)]).cpu().numpy()
```
