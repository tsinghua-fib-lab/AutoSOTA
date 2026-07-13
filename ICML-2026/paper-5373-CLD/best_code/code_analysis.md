# Code Analysis for Paper 5373 - CLD Optimization

## Evaluation Path
- benchmark_cld.py loads test split, Whisper model, CLD head, runs asr_model.predict()
- Metrics: WER%, CER%, sklearn classification_report with accuracy
- Detection Accuracy = classification_report accuracy field

## Training Path
- train_cvxnn.py: extracts mean-pooled Whisper encoder features (768-dim)
- Creates CVX_ReLU_MLP, runs ADMM, saves .pkl
- Default params: neuron=64, beta=1e-3, rho=0.1, admm_iters=6, pcg_iters=32, rank=20
- Baseline model trained with neuron=32 (confirmed: P_S=32 in pickle)
- Feature extraction: ~25 min. ADMM: ~5 min.

## Inference Path
- CVXNNLangDetectHead.predict(): mean pool over time, call stacked_predict, argmax
- stacked_predict uses jax.nn.relu(X@W1)@W2 per class
- Language tokens fed to Whisper for transcription

## Key Files
- asr_model.py: Whisper wrapper, feature extraction (mean pooling), predict pipeline
- cvx_relu_mlp.py: CVX_ReLU_MLP convex 2-layer network
- cvx_grelu_mlp.py: CVX_GReLU_MLP (simpler ADMM, no G operator)
- admm.py: ADMM solver for CVX_ReLU_MLP
- nystrom.py: Nystrom preconditioner
- lang_detect_head.py: CLD head loader + predictor

## Safe Modifications
- train_cvxnn.py: training hyperparameters
- admm.py: ADMM iterations, convergence monitoring
- asr_model.py: feature extraction methods
- cvx_relu_mlp.py: neuron count
- nystrom.py: Nystrom rank

## Baseline Model
- neuron=32, beta=1e-3, rho=0.1
- Training samples: 12,368 (768-dim)
- Validation samples: 1,546 (768-dim)
- Features cached in pickle - can reuse for fast retraining
