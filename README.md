
## 🏆 **최종 성적**

- **Public Top10% (44/426) Macro F1 Score 0.971** (1등 0.983)
- **Private Top17% (66/385) Macro F1 Score 0.969** (1등 0.985)

---

## 🎯 내가 세운 전략

- **가설**: 저해상도 이미지만 사용하는 것보다 고해상도 이미지를 함께 쓰는 것이 성능 향상에 도움이 될 것이다
- **Strategy 1**
    - 저해상도 이미지 + 고해상도 이미지를 모두 사용하여 이미지 분류 학습
    
    Strategy 1-a. Pretrain-Finetune의 전이 학습 프레임워크를 이용
    
    Strategy 1-b. Weight freezing 없이 단순 학습
    
- **Strategy 2**
    
    1a. 기존에 존재하던 pre-trained image super-resolution (SR) 모델을 불러와 
    **”저해상도 이미지 → 고해상도 이미지”**로 upscaling하도록 fine-tuning
    
    1b. 고해상도 이미지만 이용해 분류 모델을 학습
    
    1. 1a의 Fine-tuned SR 모델을 이용해 test set에 있는 저해상도 이미지를 고해상도 이미지로 upscale
    2. upscale된 test set의 저해상도 이미지를 1b에서 학습된 분류모델로 분류
