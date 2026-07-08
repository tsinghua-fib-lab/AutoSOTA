import logging
import math
import sys
from collections import defaultdict
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from hflayers import Hopfield
from omegaconf import OmegaConf

from models.base_model import BaseModel
from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.components.alpha_sampler import AlphaSampler
from models.uncertainty.components.cache import InferenceCache
from models.uncertainty.components.data_mixing import MixDataService
from models.uncertainty.components.eps_ctx_encode import FcModel, ContextEncodeModule, PositionEncoding
from models.uncertainty.components.eps_ctx_gen import EpsilonContextGen
from models.uncertainty.ml_base import CalibTrainerMixin, EpsCtxMemoryMixin, LOSS_MODE_MIX, LOSS_MODE_RES, \
    LOSS_MODE_ABS, LOSS_MODE_MSE, LOSS_MODE_EPS_CDF, BATCH_MODE_ONE_TS, PadCollate
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts
from models.uncertainty.pi_eps_sel_base import ConformalSelectionMixin
from models.uncertainty.score_service import score
from utils.calc_torch import unfold_window, calc_stats
from utils.utils import get_device

LOGGER = logging.getLogger(__name__)


class EpsPredictionHopfield(BaseModel, PIModel, CalibTrainerMixin, EpsCtxMemoryMixin, ConformalSelectionMixin):

    def __init__(self, **kwargs):
        BaseModel.__init__(self)
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        score.set_mode(kwargs.get('score_transform', None))
        # Train/Predict Hopfield Mode
        self._alphas = list(kwargs['alpha'])
        self._train_alphas = kwargs['train_alphas']
        if self._train_alphas is None:
            self._train_alphas = self._alphas
        else:
            self._train_alphas = list(self._train_alphas)
        self._train_with_past_window = kwargs['use_memory_window']
        self._train_only_past = kwargs['train_only_past']
        self._mem_past_window = kwargs['eps_mem_size']   # Only train with direct past
        self._pred_with_past_window = kwargs['use_memory_window']
        self._limit_seq_to_mem_size = kwargs['limit_train_seq_to_mem_size']
        self._absolute_epsilons = kwargs['predict_abs_eps']
        self._hopfield_beta_factor = float(kwargs['hopfield_beta'])
        self._sample_alpha_inference = kwargs['sample_alpha_inference']
        if kwargs['sample_alpha_train'] is not None:
            self._train_alpha_sampler = AlphaSampler(mode=kwargs['sample_alpha_train'],
                                                     pred_alphas=self._alphas)
        else:
            self._train_alpha_sampler = None
            assert self._sample_alpha_inference is False  # Only possible when sampled in training

        if self._train_with_past_window:
            sub_sequence_size = self._mem_past_window + 1
            sub_sequence_stride = 1
        elif self._limit_seq_to_mem_size is not None:
            sub_sequence_size = kwargs['eps_mem_size']
            sub_sequence_stride = kwargs['limit_train_seq_stride']
        else:
            sub_sequence_size = None
            sub_sequence_stride = None

        CalibTrainerMixin.__init__(self,
                                   batch_mode=kwargs['batch_mode'],
                                   batch_mix_count=kwargs['batch_mix_count'],
                                   with_loss_weight=kwargs['with_loss_weight'],
                                   coverage_loss_weight=kwargs['coverage_loss_weight'],
                                   chung_loss_weight=kwargs['chung_loss_weight'],
                                   split_in_subsequence_of_size=sub_sequence_size,
                                   subsequence_stride=sub_sequence_stride,
                                   batch_size=10 if self._train_with_past_window else kwargs['batch_size'],
                                   all_alpha_in_one_batch=True,
                                   loss_mode=kwargs['loss_mode'],
                                   pub_training=kwargs.get('pub_training', False))
        pub_inference = kwargs.get('pub_inference', False)
        self._mix_data_service = MixDataService(
            inference_mix_mode=kwargs['mix_data_inference_mode'],
            mix_inference_count=kwargs['mix_data_inference_count'],
            mix_inference_draws=kwargs['mix_inference_draws'],
            mix_inference_sample_mem_size=kwargs.get("mix_inference_sample_mem_size", None),
            ts_ids=kwargs["ts_ids"],
            pub_inference=pub_inference
        )
        EpsCtxMemoryMixin.__init__(self, mem_size=kwargs['eps_mem_size'], keep_calib_eps=kwargs['keep_calib_eps'],
                                   store_step_no=True,
                                   memory_in_ram=kwargs.get('memory_in_ram', False),
                                   mix_data_count=self._mix_data_service.get_mix_inference_count(),
                                   pub_inference=pub_inference)
        ConformalSelectionMixin.__init__(self)
        self._save_attention_record = kwargs['record_attention']
        # Context
        self._online_memory = kwargs['online_memory']
        self._online_mem_delay = kwargs.get('online_mem_delay', None)
        self._current_mem_delay = 0
        self._mem_update_buffer = defaultdict(list)
        self._ctx_past_window = kwargs['ctx_past_window']
        self._use_pre_calib_eps_for_calib = False
        self._ctx_gen = EpsilonContextGen(mode=kwargs['ctx_mode'], ts_ids=kwargs['ts_ids'])
        # ContextEncode Param
        self._pos_encode_params = kwargs['pos_encode']
        if self._pos_encode_params is not None and self._pos_encode_params['mode'] is None:
            self._pos_encode_params = None      # Set to None if mode is None
        self._history_comp_params = kwargs['history_compress']
        if self._history_comp_params is not None and self._history_comp_params['mode'] is None:
            self._history_comp_params = None     # Set to None if mode is None
        # Network
        self._pre_encode_context = True
        self._eps_to_stored_pattern = kwargs['eps_to_stored_pattern']
        self._alpha_in_final_layer = False
        self._head_per_alpha = kwargs['head_per_alpha']
        self._predict_beta = False
        # Prediction (Inference) Mode
        self._conformal_selection = kwargs['conf_selection']
        self._conformal_sel_abs = kwargs['conf_eps_abs']
        self._conformal_sel_beta = kwargs['conf_sel_beta']
        self._sum_assoc_for_qc = kwargs['mix_head_dist']
        self._conf_quantile_mode = kwargs['conf_quantile_mode']
        assert self._conf_quantile_mode in ['sample', 'cdf', 'debug']
        if self._loss_mode in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF]:
            self._hopfield_heads = 1
        elif self._loss_mode == LOSS_MODE_MSE:
            self._hopfield_heads = 1
        elif self._head_per_alpha:
            self._hopfield_heads = len(self._train_alphas) * (1 if self._absolute_epsilons else 2)
        else:
            self._hopfield_heads = 1

        self._use_adaptiveci = kwargs.get('use_adaptiveci', False)
        if self._use_adaptiveci:
            self._gamma = kwargs['gamma']
        self._alpha_t = None
        self._ctx_enc_hiddenL = kwargs.get('ctx_encode_hiddenL', 0)
        self._ctx_enc_dropout = kwargs.get('ctx_encode_dropout', None)
        self._ctx_encode_relu_before_hf = kwargs.get('ctx_encode_relu_before_hf', False)

        # Assertions
        if self._use_adaptiveci:
            assert self._conformal_selection
        if self._head_per_alpha:
            assert not self._alpha_in_final_layer
            assert self._train_alpha_sampler is None
        if self._train_alphas != self._alphas:
            assert self._head_per_alpha and self._conformal_selection
        if self._loss_mode in [LOSS_MODE_ABS, LOSS_MODE_MIX, LOSS_MODE_MSE, LOSS_MODE_EPS_CDF]:
            assert self._absolute_epsilons
        if self._no_train_alpha_needed():
            assert self._conformal_selection
            assert self._train_alpha_sampler is None
            assert not self._head_per_alpha
            assert not self._alpha_in_final_layer
        if self._mix_data_service.mode is not None:
            assert self._batch_mode != BATCH_MODE_ONE_TS
            assert not self._pred_with_past_window
            assert not self._train_with_past_window
        assert (self._hopfield_heads % 2 == 0 or self._absolute_epsilons)
        assert (self._hopfield_heads > 2 or not self._alpha_in_final_layer)

        self._inference_cache = InferenceCache()
        self._use_cached_inference = kwargs.get('cached_inference', False)
        #
        # Init Network
        #
        # Input Encoding
        self._ctx_history_state = None
        self._mix_hist_states = None
        if self._pre_encode_context:
            ctx_size = self._ctx_gen.context_size(kwargs['no_x_features'], self._ctx_past_window,
                                                  fc_state_dim=kwargs['fc_state_dim'])
            ctx_enc_hidden = tuple([ctx_size] * self._ctx_enc_hiddenL)
            self._ctx_encoding = ContextEncodeModule(
                ctx_input_dim=ctx_size, ctx_enc_hidden=ctx_enc_hidden, ctx_out_dim=ctx_size,
                history_compression=OmegaConf.to_container(self._history_comp_params)
                if self._history_comp_params is not None else None,
                dropout=self._ctx_enc_dropout,
                relu_after_last=self._ctx_encode_relu_before_hf
            )
            ctx_encoded_size = self._ctx_encoding.output_dim
            if self._pos_encode_params is not None:
                self._extra_ctx_pos_encoding =\
                    PositionEncoding(mem_size=self.max_mem_size, **self._pos_encode_params)
                ctx_encoded_size = ctx_encoded_size + self._extra_ctx_pos_encoding.additional_dim
            else:
                self._extra_ctx_pos_encoding = None

        else:
            ctx_encoded_size = self._ctx_gen.context_size(kwargs['no_x_features'], self._ctx_past_window,
                                                          fc_state_dim=kwargs['fc_state_dim'])
            self._ctx_encoding = lambda **enc_args: (enc_args['context'], None)
            self._ctx_encoding.has_history_comp = False
            self._extra_ctx_pos_encoding = None

        # Beta (NOT hopfield beta) Prediction - (Not used)
        if self._predict_beta:
            # Predict beta share of alpha
            self._beta_network = nn.Sequential(FcModel(input_dim=ctx_encoded_size + 1, out_dim=1, hidden=()),
                                               nn.Sigmoid())

        # Hopfield
        hopfield_hidden = ctx_encoded_size  # // 2 #TODO
        self._hopfield = Hopfield(
            batch_first=True,
            input_size=ctx_encoded_size + (0 if self._head_per_alpha or self._no_train_alpha_needed() else 2),   # R (Query) size + Alpha/Beta
            hidden_size=hopfield_hidden,
            pattern_size=1,                                                     # Epsilon size (Values)
            output_size=None,                                                   # Not used because no output projection
            num_heads=self._hopfield_heads,                                     # k per interval bound
            stored_pattern_size=ctx_encoded_size + (1 if self._eps_to_stored_pattern else 0),  # Stored ctx size (+ eps) (Keys)
            pattern_projection_size=1,
            scaling=self._hopfield_beta(ctx_encoded_size),
            # do not pre-process layer input
            #normalize_stored_pattern=False,
            #normalize_stored_pattern_affine=False,
            #normalize_state_pattern=False,
            #normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            # do not post-process layer output
            disable_out_projection=True                 # To get Heads - one head per epsilon
        )

        # Finals Layers
        if not self._head_per_alpha and (self._alpha_in_final_layer or self._hopfield_heads > 2):
            self._final_layer = nn.ModuleList()
            for _ in range(1):
                dim = self._hopfield_heads + (1 if self._alpha_in_final_layer else 0)
                self._final_layer.append(FcModel(input_dim=dim, hidden=(dim,),
                                                 out_dim=(1 if self._absolute_epsilons else 2)))

    @property
    def has_mix_service(self) -> bool:
        return True

    def _hopfield_beta(self, memory_dim):
        # Default beta = Self attention beta = 1 / sqrt(key_dim)
        return self._hopfield_beta_factor / math.sqrt(memory_dim)

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        Y_hat = []
        fc_state_step = []
        calib_artifacts = []
        for c_data in calib_data:
            c_result = self._forcast_service.predict(
                FCPredictionData(ts_id=c_data.ts_id, X_past=c_data.X_pre_calib, Y_past=c_data.Y_pre_calib,
                                 X_step=c_data.X_calib, step_offset=c_data.step_offset))
            Y_hat.append(c_result.point)
            fc_state_step.append(c_result.state)
            calib_artifacts.append(PICalibArtifacts(fc_Y_hat=Y_hat[-1], fc_state_step=fc_state_step[-1]))

        # Init Variance for Mixing
        if self._loss_mode == LOSS_MODE_MIX:
            self._mix_variance = nn.Parameter(torch.std(torch.stack([score.get(Y_hat=Y_hat[i], Y=c_data.Y_calib)
                                                                      for i, c_data in enumerate(calib_data)])),
                                              requires_grad=True)
        else:
            self._mix_variance = None
        trainer_config = kwargs['trainer_config']
        experiment_config = kwargs['experiment_config']
        if trainer_config.trainer_config.init_model is not None:
            EVALUATE_LOADED_MODEL = False
            LOGGER.warning(f"Model is not trained but loaded from {trainer_config.trainer_config.init_model}!!")
            if isinstance(trainer_config.trainer_config.init_model, dict):
                int_model_path = trainer_config.trainer_config.init_model[experiment_config.seed]
            else:
                int_model_path = trainer_config.trainer_config.init_model
            self.load_state(int_model_path, self.device)
            self.to(device=get_device(experiment_config.gpu_id))
            if EVALUATE_LOADED_MODEL:
                trainer_config.trainer_config.n_epochs = 0  # No Training only eval
                self._train_model(calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step, alphas=self._train_alphas,
                                  experiment_config=experiment_config, trainer_config=trainer_config,
                                  history_size=self.max_mem_size if self._history_comp_params else 0)
        else:
            self._train_model(calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step, alphas=self._train_alphas,
                              experiment_config=experiment_config, trainer_config=trainer_config,
                              history_size=self.max_mem_size if self._history_comp_params else 0)
        return calib_artifacts

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        # Fill Memory
        self._ctx_history_state, self._mix_hist_states =\
            self._fill_memory(calib_data, calib_artifact, mix_calib_data=mix_calib_data,
                              mix_calib_artifacts=mix_calib_artifact)
        if self._use_cached_inference and alpha == self._alphas[0]:  # is the first alpha -> clear cache
            self._inference_cache.clear()
        #calib_artifact.fc_state_step = None  # Save Memory
        return calib_artifact

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset
        self._current_mem_delay = 0  # Update Memory Delay
        self._mem_update_buffer.clear()  # Clear Memory Delay Buffer

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        if self._use_cached_inference:
            return self._predict_step_cached(pred_data)
        # Feed through model and retrieve eps_lower, eps_higher
        alpha, X_step, X_past, Y_past, eps_past, step_abs =\
            pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past, pred_data.eps_past,\
            pred_data.step_offset_overall
        # Get FC Prediction and encoded context
        Y_hat, ctx_encoded, self._ctx_history_state, step_abs = self._encode_and_fc(
            ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step, step_abs=step_abs, eps_past=eps_past,
            ctx_past=self._memory.ctx_chronological, ctx_history_state=self._ctx_history_state
        )
        # Sample alpha (optional)
        if not self._head_per_alpha and self._train_alpha_sampler and self._conformal_selection\
                and self._sample_alpha_inference:
            alpha_retrieve = self._train_alpha_sampler.sample_inference(alpha=alpha, device=self._current_device)
            reduce_alpha = True
        else:
            alpha_retrieve = alpha
            reduce_alpha = False
        # Retrieve From Model
        if self._pred_with_past_window:
            assert self._mix_data_service.mode is None
            eps_q_low, eps_q_high, _, _, assoc_matrix, proj_pattern_matrix = self._forward(
                current_ctx=ctx_encoded, alpha=alpha_retrieve,
                memory_ctx=self._memory.ctx_chronological[-self._mem_past_window:],
                memory_eps=self._memory.eps_chronological[-self._mem_past_window:],
                current_ctx_step=step_abs, memory_ctx_step=self._memory.step_no_chronological[-self._mem_past_window:],
                retrieve_model_prediction=not self._conformal_selection,
                retrieve_association_matrix=self._save_attention_record or self._conformal_selection)
            uc_attention = self._record_attention(
                assoc_matrix, step_abs,
                self._memory.step_no_chronological[-self._mem_past_window:])
            if self._conformal_selection:
                assoc_matrix = self._reduce_assoc_matrix(association_matrix=assoc_matrix, alpha=alpha,
                                                         reduce_alpha=reduce_alpha)
                eps_q_low, eps_q_high, _, add_quantile_info = self._get_quantile_conformal(
                    association_matrix=assoc_matrix, alpha=self._alpha_t, inference=True,
                    eps=self._memory.eps_chronological[-self._mem_past_window:])
        else:
            selected_mix_ts = self._mix_data_service.select_mix_inference_step_ids(pred_data.ts_id)
            selected_ts_subsets = self._mix_data_service.select_mix_inference_subsets()
            selected_mem_samples = self._mix_data_service.select_mix_inference_mem_samples(
                len(self._memory), None if len(self._mix_data_memory) < 1 else len(self._mix_data_memory[0]))
            eps_q_low, eps_q_high, _, _, assoc_matrix, proj_pattern_matrix = self._forward(
                current_ctx=ctx_encoded, alpha=alpha_retrieve,
                memory_ctx=self._memory.ctx if self._mix_data_service.mode is None
                else self._get_data_with_mix_mem_ctx(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                memory_eps=self._memory.eps if self._mix_data_service.mode is None
                else self._get_data_with_mix_mem_eps(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                memory_ctx_step=self._memory.step_no if self._mix_data_service.mode is None
                else self._get_data_with_mix_mem_step(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                current_ctx_step=step_abs,
                retrieve_model_prediction=not self._conformal_selection,
                retrieve_association_matrix=self._save_attention_record or self._conformal_selection)
            uc_attention = self._record_attention(
                assoc_matrix, step_abs, self._memory.step_no if self._mix_data_service.mode is None
                else self._get_data_with_mix_mem_step(selected_mix_ts, selected_ts_subsets, selected_mem_samples))
            if self._conformal_selection:
                assoc_matrix = self._reduce_assoc_matrix(association_matrix=assoc_matrix, alpha=alpha,
                                                         reduce_alpha=reduce_alpha)
                eps_q_low, eps_q_high, _, add_quantile_info = self._get_quantile_conformal(
                    association_matrix=assoc_matrix, alpha=self._alpha_t, inference=True,
                    eps=self._memory.eps if self._mix_data_service.mode is None
                    else self._get_data_with_mix_mem_eps(selected_mix_ts, selected_ts_subsets, selected_mem_samples))
        # Generate Interval
        pred_int = Y_hat + score.resolve(eps_q_low, **pred_data.score_param).to(torch.device('cpu')),\
            Y_hat + score.resolve(eps_q_high, **pred_data.score_param).to(torch.device('cpu'))
        prediction_result = PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)
        prediction_result.eps_ctx = ctx_encoded.detach()
        prediction_result.uc_attention = uc_attention
        prediction_result.quantile_info = add_quantile_info
        return prediction_result

    def _predict_step_cached(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        assert not self._pred_with_past_window
        assert not self._use_adaptiveci
        assert self._conformal_selection
        assert not self._sample_alpha_inference
        assert not self._save_attention_record
        alpha, X_step, X_past, Y_past, eps_past, step_abs =\
            pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past, pred_data.eps_past,\
            pred_data.step_offset_overall

        if self._inference_cache.is_cached(pred_data.ts_id, pred_data.step_offset_overall):
            # Use from Cache
            prediction_result = self._inference_cache.get_cached(pred_data.ts_id, pred_data.step_offset_overall, alpha)
            prediction_result.from_cache = True
        else:
            Y_hat, ctx_encoded, self._ctx_history_state, step_abs = self._encode_and_fc(
                ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step, step_abs=step_abs, eps_past=eps_past,
                step_rel=pred_data.step_offset_prediction,
                ctx_past=self._memory.ctx_chronological if not self._memory.empty else None,
                ctx_history_state=self._ctx_history_state
            )
            n_draws = self._mix_data_service.mix_inference_draws()
            eps_q_low_l, eps_q_high_l = defaultdict(list), defaultdict(list)
            for subset_draw in range(n_draws):
                selected_mix_ts = self._mix_data_service.select_mix_inference_step_ids(pred_data.ts_id)
                selected_ts_subsets = self._mix_data_service.select_mix_inference_subsets()
                selected_mem_samples = self._mix_data_service.select_mix_inference_mem_samples(
                    len(self._memory), None if len(self._mix_data_memory) < 1 else len(self._mix_data_memory[0]))
                eps_q_low, eps_q_high, _, _, assoc_matrix, proj_pattern_matrix = self._forward(
                    current_ctx=ctx_encoded, alpha=alpha,
                    memory_ctx=self._memory.ctx if self._mix_data_service.mode is None
                    else self._get_data_with_mix_mem_ctx(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                    memory_eps=self._memory.eps if self._mix_data_service.mode is None
                    else self._get_data_with_mix_mem_eps(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                    memory_ctx_step=self._memory.step_no if self._mix_data_service.mode is None
                    else self._get_data_with_mix_mem_step(selected_mix_ts, selected_ts_subsets, selected_mem_samples),
                    current_ctx_step=step_abs,
                    retrieve_model_prediction=not self._conformal_selection,
                    retrieve_association_matrix=self._save_attention_record or self._conformal_selection)
                assoc_matrix = self._reduce_assoc_matrix(association_matrix=assoc_matrix, alpha=alpha, reduce_alpha=None)
                for a in self._alphas:
                    eps_q_low, eps_q_high, _, add_quantile_info = self._get_quantile_conformal(
                        association_matrix=assoc_matrix, alpha=a, inference=True,
                        eps=self._memory.eps if self._mix_data_service.mode is None
                        else self._get_data_with_mix_mem_eps(selected_mix_ts, selected_ts_subsets, selected_mem_samples))
                    eps_q_low_l[a].append(eps_q_low)
                    eps_q_high_l[a].append(eps_q_high)

            for a in self._alphas:
                if n_draws > 1:
                    eps_q_low_l[a] = self._mix_data_service.merge_mix_draws(eps_q_low_l[a])
                    eps_q_high_l[a] = self._mix_data_service.merge_mix_draws(eps_q_high_l[a])
                else:
                    eps_q_low_l[a] = eps_q_low_l[a][0]
                    eps_q_high_l[a] = eps_q_high_l[a][0]
                # Generate Interval
                pred_int = Y_hat + score.resolve(eps_q_low_l[a], **pred_data.score_param).to(torch.device('cpu')),\
                    Y_hat + score.resolve(eps_q_high_l[a], **pred_data.score_param).to(torch.device('cpu'))
                prediction_result = PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)
                prediction_result.eps_ctx = ctx_encoded.detach()
                prediction_result.uc_attention = None
                prediction_result.quantile_info = None
                self._inference_cache.cache_result(pred_data.ts_id, pred_data.step_offset_overall, a,
                                                   prediction_result)

            # Use from Cache
            prediction_result = self._inference_cache.get_cached(pred_data.ts_id, pred_data.step_offset_overall, alpha)

        return prediction_result

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # Update memory
        if self._online_memory and (not hasattr(pred_result, "from_cache") or not pred_result.from_cache):
            encoded_eps_ctx = pred_result.eps_ctx
            pred_result.eps_ctx = None
            eps = score.get(Y=Y_step, Y_hat=pred_result.fc_Y_hat, **pred_data.score_param)
            step = torch.arange(start=pred_data.step_offset_overall,
                                end=pred_data.step_offset_overall + eps.shape[0],
                                dtype=torch.long, device=Y_step.device)
            if self._online_mem_delay is not None:
                self._current_mem_delay += 1
                self._mem_update_buffer[pred_data.ts_id].append((encoded_eps_ctx, eps, step))
            else:
                self._add_step_to_mem(ctx=encoded_eps_ctx, eps=eps, step=step)
            if self._mix_data_service.mode is not None:  # Multi TS
                step_abs = pred_data.step_offset_overall
                for idx, mix_ts_data in enumerate(pred_data.mix_ts):
                    Y_hat, ctx_encoded, self._mix_hist_states[idx], _ = self._encode_and_fc(
                        ts_id=mix_ts_data.ts_id, X_past=mix_ts_data.X_past, Y_past=mix_ts_data.Y_past,
                        X_step=mix_ts_data.X_step, eps_past=mix_ts_data.eps_past, step_abs=step_abs,
                        ctx_past=self._get_mix_mem(ts_id=mix_ts_data.ts_id).ctx_chronological
                        if self._ctx_encoding.has_history_comp else None,
                        ctx_history_state=self._mix_hist_states[idx]
                        if self._ctx_encoding.has_history_comp else None,
                    )
                    mix_eps = score.get(Y=mix_ts_data.Y_step, Y_hat=Y_hat, **mix_ts_data.score_param)
                    if self._online_mem_delay is not None:
                        self._mem_update_buffer[mix_ts_data.ts_id].append((encoded_eps_ctx, eps, step))
                    else:
                        self._add_step_to_mix_mem(ts_id=mix_ts_data.ts_id, ctx=ctx_encoded, eps=mix_eps, step=step)
            # Fill buffered memory in case of delay
            if self._online_mem_delay is not None and self._current_mem_delay > self._online_mem_delay:
                for ts_id, buffer in self._mem_update_buffer.items():
                    for buffer_values in buffer:
                        if ts_id == pred_data.ts_id:
                            self._add_step_to_mem(ctx=buffer_values[0], eps=buffer_values[1], step=buffer_values[2])
                        else:
                            self._add_step_to_mix_mem(ts_id=ts_id, ctx=buffer_values[0], eps=buffer_values[1],
                                                      step=buffer_values[2])
                self._current_mem_delay = 0
                self._mem_update_buffer.clear()
        #If Adaptive:
        if self._use_adaptiveci:
            alpha = pred_data.alpha
            pred_int = pred_result.pred_interval
            err_step = 0 if pred_int[0] <= Y_step <= pred_int[1] else 1
            # Simple Mode
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
            self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    def _encode_and_fc(self, ts_id, X_past, Y_past, X_step, step_abs, eps_past, ctx_past, ctx_history_state, step_rel=None):
        fc_result = self._forcast_service.predict(
            FCPredictionData(ts_id=ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=step_abs, step_rel=step_rel))
        Y_hat = fc_result.point
        fc_state_step = fc_result.state
        ts_id_enc = self._ctx_gen.get_ts_id_enc(ts_id)
        ctx = self._ctx_gen.calc_single(
            X_past=X_past[-self._ctx_past_window:] if X_past is not None and self._ctx_gen.use_past else None,
            Y_past=Y_past[-self._ctx_past_window:] if Y_past is not None and self._ctx_gen.use_past else None,
            eps_past=eps_past[-self._ctx_past_window:] if eps_past is not None and self._ctx_gen.use_past else None,
            X_step=X_step.squeeze(dim=0) if X_step is not None else None,
            Y_hat_step=Y_hat.squeeze(dim=0),
            fc_state_step=fc_state_step.squeeze(dim=0) if fc_state_step is not None else None,
            ts_id_enc=torch.tensor([ts_id_enc], dtype=torch.long, device=self._current_device) if ts_id_enc is not None else None
        )
        # Encode Context
        ctx = ctx.to(self._current_device)
        step_abs = torch.tensor([step_abs], dtype=torch.long, device=self._current_device)
        ctx_encoded, cts_history_state = self._ctx_encoding(
            context=ctx, step_no=step_abs, context_past=ctx_past,
            context_past_state=ctx_history_state, past_pre_encoded=True, past_has_history=True)
        return Y_hat, ctx_encoded, cts_history_state, step_abs

    def model_ready(self):
        if not self._pub_inference:
            return not self._memory.empty
        else:
            return not self._mix_data_memory[0].empty

    @property
    def can_handle_different_alpha(self):
        return True

    def required_past_len(self) -> Tuple[int, int]:
        if self._ctx_gen.use_eps_past:
            return self._ctx_past_window, sys.maxsize
        elif self._ctx_gen.use_past:
            return 0, sys.maxsize if self._history_comp_params is not None else self._ctx_past_window
        else:
            return 0, sys.maxsize if self._history_comp_params is not None else 0

    def _encode_ctx(self, context, step_no) -> Tuple[torch.tensor, torch.tensor]:
        return self._ctx_encoding(context=context, step_no=step_no, context_past=None, context_past_state=None)

    def _get_calib_ctx(self, calib_data, Y_hat, fc_state_step=None) -> Tuple[torch.tensor, int, int]:
        return self._ctx_gen.calib_data_to_ctx(calib_data, Y_hat=Y_hat, past_window=self._ctx_past_window,
                                               fc_state_step=fc_state_step,
                                               use_pre_calib_eps_for_calib=self._use_pre_calib_eps_for_calib)

    #
    # Network Module methods
    #

    def _forward(self, current_ctx, memory_ctx, memory_eps, alpha, mask=None,
                 current_ctx_step=None, memory_ctx_step=None,
                 retrieve_model_prediction=True, retrieve_association_matrix=False,
                 retrieve_projected_pattern_matrix=False):
        squeeze = False
        if self._memory_in_ram:
            memory_ctx = memory_ctx.to(current_ctx.device)
            memory_eps = memory_eps.to(current_ctx.device)

        if len(current_ctx.shape) == 2:
            squeeze = True
            if self._sample_alpha_inference:
                assert alpha.shape[0] > 1
                alpha = alpha.unsqueeze(0)
            else:
                assert isinstance(alpha, float)
                alpha = torch.tensor([alpha], dtype=torch.float, device=current_ctx.device).unsqueeze(0)
            current_ctx = current_ctx.unsqueeze(0)
            memory_ctx = memory_ctx.unsqueeze(0)
            memory_eps = memory_eps.unsqueeze(0)
            current_ctx_step = current_ctx_step.unsqueeze(0).unsqueeze(2)
            memory_ctx_step = memory_ctx_step.unsqueeze(0).unsqueeze(2)

        # Optional Positional Encoding
        if self._extra_ctx_pos_encoding is not None:
            max_current_step = torch.max(current_ctx_step)
            min_memory_step = torch.min(memory_ctx_step)
            current_ctx = self._extra_ctx_pos_encoding(context_enc=current_ctx, step_no=current_ctx_step,
                                                       max_step=max_current_step, min_step=min_memory_step,
                                                       ref_step=max_current_step)
            memory_ctx = self._extra_ctx_pos_encoding(context_enc=memory_ctx, step_no=memory_ctx_step,
                                                      max_step=max_current_step, min_step=min_memory_step,
                                                      ref_step=max_current_step)

        no_batches = current_ctx.shape[0]
        ctx_features = current_ctx.shape[2]
        no_alphas = alpha.shape[1]
        seq_len = current_ctx.shape[1]
        alphas = alpha.unsqueeze(2)
        alphas = alphas.permute(1, 0, 2).reshape(-1, 1, 1)
        alphas = alphas.repeat_interleave(seq_len, dim=1)
        if self._head_per_alpha or self._no_train_alpha_needed():
            state_pattern = current_ctx
            if self._no_train_alpha_needed():
                beta = None
            elif self._loss_mode == LOSS_MODE_ABS:
                beta = torch.zeros_like(alphas)
            else:
                beta = torch.div(alphas, 2)
        else:
            current_ctx = current_ctx.repeat(no_alphas, 1, 1)
            if self._loss_mode != LOSS_MODE_RES:
                beta = torch.zeros_like(alphas)
            elif self._predict_beta:
                beta = alphas * self._beta_network(torch.cat((current_ctx, alphas), dim=2).reshape(-1, ctx_features + 1))\
                    .reshape(no_batches, seq_len, -1)
            else:
                beta = torch.div(alphas, 2)
            state_pattern = torch.cat((current_ctx, beta - alphas + 1, beta), dim=2)
        del current_ctx     # now in state_pattern
        if self._eps_to_stored_pattern:
            stored_pattern = torch.cat((memory_ctx, memory_eps), dim=2)
        else:
            stored_pattern = memory_ctx
        del memory_ctx       # now in stored_pattern
        if self._absolute_epsilons:
            memory_eps = torch.abs(memory_eps)

        if not self._head_per_alpha:
            stored_pattern = stored_pattern.repeat(no_alphas, 1, 1)
            memory_eps = memory_eps.repeat(no_alphas, 1, 1)

        if retrieve_association_matrix:
            association_matrix = self._hopfield.get_association_matrix(
                (stored_pattern, state_pattern, memory_eps), association_mask=mask)
        else:
            association_matrix = None
        if retrieve_projected_pattern_matrix:
            projected_pattern_matrix = self._hopfield.get_projected_pattern_matrix(
                (stored_pattern, state_pattern, memory_eps), association_mask=mask)
        else:
            projected_pattern_matrix = None

        if retrieve_model_prediction:
            hopfield_result = self._hopfield((stored_pattern, state_pattern, memory_eps), association_mask=mask)
            if self._head_per_alpha:
                if no_alphas == 1:
                    idx = self._train_alphas.index(alphas[0, 0, 0])
                    eps_q_high = hopfield_result[:, :, idx].unsqueeze(2)
                    if self._absolute_epsilons:
                        eps_q_low = torch.neg(eps_q_high)
                    else:
                        eps_q_low = hopfield_result[:, :, self._hopfield_heads // 2 + idx].unsqueeze(2)
                else:
                    assert len(self._train_alphas) == no_alphas
                    assert hopfield_result.shape[2] == no_alphas * (1 if self._absolute_epsilons else 2)
                    head_split = self._hopfield_heads if self._absolute_epsilons else self._hopfield_heads // 2
                    eps_q_high = hopfield_result[:, :, :head_split]\
                        .permute(2, 0, 1)\
                        .reshape(-1, seq_len, 1)
                    if self._absolute_epsilons:
                        eps_q_low = torch.neg(eps_q_high)
                    else:
                        eps_q_low = hopfield_result[:, :, head_split:]\
                            .permute(2, 0, 1)\
                            .reshape(-1, seq_len, 1)

            elif self._alpha_in_final_layer or self._hopfield_heads > 2:
                if self._alpha_in_final_layer:
                    feat_len = hopfield_result.shape[2] + 1
                    input_final = torch.cat((hopfield_result, alphas), dim=2).reshape(-1, feat_len)
                else:
                    feat_len = hopfield_result.shape[2]
                    input_final = hopfield_result.reshape(-1, feat_len)
                final_out = self._final_layer[0](input_final).reshape(no_batches * no_alphas, seq_len, -1)
                eps_q_high = final_out[:, :, 0].unsqueeze(2)
                if self._absolute_epsilons:
                    eps_q_low = torch.neg(eps_q_high)
                else:
                    eps_q_low = final_out[:, :, 1].unsqueeze(2)
            else:
                # TODO Improve memory efficency by using.tensor_split()
                eps_q_high = hopfield_result[:, :, 0].unsqueeze(2)
                if self._absolute_epsilons:
                    eps_q_low = torch.neg(eps_q_high)
                else:
                    eps_q_low = hopfield_result[:, :, 1].unsqueeze(2)
        else:
            eps_q_low = None
            eps_q_high = None

        if squeeze:
            return eps_q_low.squeeze(0) if eps_q_low is not None else None,\
                   eps_q_high.squeeze(0) if eps_q_high is not None else None,\
                   beta.squeeze(0) if beta is not None else None,\
                   alphas.squeeze(0),\
                   association_matrix.squeeze(0) if association_matrix is not None else None,\
                   projected_pattern_matrix.squeeze(0) if projected_pattern_matrix is not None else None
        else:
            return eps_q_low, eps_q_high, beta, alphas, association_matrix, projected_pattern_matrix

    def forward(self, val, *args, **kwargs):
        """
        Forward method used in !training!
        """
        Y, Y_hat, alpha, step_no = kwargs['Y'].detach(), kwargs['Y_hat'].detach(), kwargs['alpha'],\
                                   kwargs['step_no'].detach()

        if "score_param" in kwargs:
            score_param = dict(ts_sd=kwargs['score_param'].detach())  # TODO: Make this more general
        else:
            score_param = {}

        # Hack: Padding Value is -1 which is no possible step_no value -> One can get PaddingMask
        padding_mask = step_no == PadCollate.PAD_VALUE
        has_padding = torch.any(padding_mask)
        if not has_padding:
            del padding_mask

        eps = score.get(Y=Y, Y_hat=Y_hat, **score_param).detach()  # [batch, batch_size, *]
        y_features = Y_hat.shape[2]
        no_alphas = alpha.shape[1]
        if self._batch_mode == BATCH_MODE_ONE_TS:
            batch_encoded, reshape_func, batches, seq_len, ctx_features = self._forward_encode(**kwargs)
            interleave_score_param = True
            mask = None
        else:
            batch_encoded, reshape_func, batches, seq_len, ctx_features = self._forward_encode_naive_mix(**kwargs)
            alpha = alpha[0:batch_encoded.shape[0], :]
            interleave_score_param = False
            if self._pub_training:
                mask = torch.meshgrid(kwargs['ts_id'].squeeze(), kwargs['ts_id'].squeeze())
                mask = mask[0] == mask[1]
            else:
                mask = None
        Y = reshape_func(Y)
        Y_hat = reshape_func(Y_hat)
        step_no = reshape_func(step_no)
        eps = reshape_func(eps)
        # if False: In case we want it sorted by epsilon (BUT no history compression possible)
        #    eps.requires_grad = True
        #    eps, sort_idx = torch.sort(eps, dim=1)
        #    ctx = ctx.view(-1, ctx_features)[sort_idx.view(-1, 1), :].view(batches, seq_len, ctx_features)
        #    Y = Y.view(-1, 1)[sort_idx.view(-1, 1), 0:1].view(batches, seq_len, 1)
        #    Y_hat = Y_hat.view(-1, 1)[sort_idx.view(-1, 1), 0:1].view(batches, seq_len, 1)
        #    step_no = step_no.view(-1, 1)[sort_idx.view(-1, 1), 0:1].view(batches, seq_len, 1)

        if self._train_alpha_sampler is not None:
            used_alpha = self._train_alpha_sampler.sample(alpha)
        else:
            used_alpha = alpha

        if self._train_with_past_window:
            assert self._batch_mode == BATCH_MODE_ONE_TS
            assert not has_padding  # Not supported ATM
            current_ctx = batch_encoded[:, -1, :].unsqueeze(1)
            current_ctx_step = step_no[:, -1, :].unsqueeze(1)
            memory_ctx = batch_encoded[:, :-1, :]
            memory_ctx_step = step_no[:, :-1, :]
            memory_eps = eps[:, :-1, :]
            eps_q_low, eps_q_high, beta, alphas, assoc_matrix, _ = self._forward(
                current_ctx=current_ctx, memory_ctx=memory_ctx, memory_eps=memory_eps, alpha=used_alpha,
                current_ctx_step=current_ctx_step, memory_ctx_step=memory_ctx_step,
                retrieve_model_prediction=self._loss_mode not in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF],
                retrieve_association_matrix=self._loss_mode in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF]
                                            or (self._loss_mode == LOSS_MODE_MSE and val)
            )
            min_window = 0
            res = lambda t: t[:, -1, :]
        else:
            if self._train_only_past:
                min_window = 40
                assert mask is None  # PUB Mask not allowed here
                mask = torch.cat(
                    (torch.full((seq_len - min_window, min_window), fill_value=True, dtype=torch.bool),
                     torch.tril(torch.full((seq_len - min_window, seq_len - min_window), fill_value=True, dtype=torch.bool), diagonal=-1))
                    , dim=1)
            else:
                min_window = 0
                if mask is None:
                    mask = torch.diag(torch.full((seq_len, ), fill_value=True, dtype=torch.bool))
            if has_padding:
                mask = torch.logical_or(mask.unsqueeze(0).repeat(batches, 1, 1), padding_mask.repeat(1, 1, seq_len))
            eps_q_low, eps_q_high, beta, alphas, assoc_matrix, _ = self._forward(
                current_ctx=batch_encoded[:, min_window:, :], memory_ctx=batch_encoded, memory_eps=eps, alpha=used_alpha,
                current_ctx_step=step_no[:, min_window:, :], memory_ctx_step=step_no,
                retrieve_model_prediction=self._loss_mode not in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF],
                retrieve_association_matrix=self._loss_mode in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF]
                                            or (self._loss_mode == LOSS_MODE_MSE and val),
                mask=mask.to(device=batch_encoded.device)
            )
            res = lambda t: t.reshape(-1, 1)

        if not self._no_train_alpha_needed():   # We directly derive quantiles from training
            Y_hat = res(Y_hat[:, min_window:, :]).repeat(no_alphas, 1)  # repeat for each alpha
            eps_q_low, eps_q_high, beta, alphas = res(eps_q_low), res(eps_q_high), res(beta), res(alphas)
            q_low = Y_hat + score.resolve(eps_q_low, **score_param, repeat=no_alphas,
                                          interleave=seq_len if interleave_score_param else 1)
            q_high = Y_hat + score.resolve(eps_q_high, **score_param, repeat=no_alphas,
                                           interleave=seq_len if interleave_score_param else 1)
            if self._batch_mode == BATCH_MODE_ONE_TS:
                return dict(q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(beta - alphas + 1), Y_hat=Y_hat,
                            loss_mask=padding_mask if has_padding else None)
            else:
                return dict(Y=Y, base_alphas=alpha, alpha=alpha,
                            q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(beta - alphas + 1), Y_hat=Y_hat,
                            loss_mask=padding_mask if has_padding else None)
        else:  # We need to get quantile by inference setting for metrics
            assert not self._train_with_past_window and not self._train_only_past
            # Get Intervals by inference Mode
            if val:
                if has_padding:
                    assoc_matrix = assoc_matrix.masked_fill(padding_mask.unsqueeze(1).repeat(1, 1, 1, seq_len), 0.1)
                Y_hat = res(Y_hat[:, min_window:, :]).repeat(len(self._train_alphas), 1)

                val_q_low, val_q_high, val_alpha, val_beta = self._train_metric_interval(
                    eps=eps, assoc_matrix=assoc_matrix, batches=batches, seq_len=seq_len)
                q_low = Y_hat + score.resolve(val_q_low.unsqueeze(-1), **score_param, repeat=len(self._train_alphas),
                                              interleave=seq_len if interleave_score_param else 1)
                q_high = Y_hat + score.resolve(val_q_high.unsqueeze(-1), **score_param, repeat=len(self._train_alphas),
                                               interleave=seq_len if interleave_score_param else 1)
            else:
                q_low, q_high, val_beta, val_alpha = None, None, None, None
            eps = eps.squeeze(-1)
            if self._loss_mode in [LOSS_MODE_MIX, LOSS_MODE_EPS_CDF]:
                assert not has_padding  # Not sure if working
                if self._absolute_epsilons:
                    eps = torch.abs(eps)
                assert eps_q_low is eps_q_high is None
                assert assoc_matrix.shape[1] == 1  # 1 Head for now
                assoc_matrix = assoc_matrix.reshape(-1, seq_len)
                eps_reference = eps.unsqueeze(1)
                eps_reference = eps_reference.expand(batches, seq_len, seq_len)
                eps_reference = eps_reference.reshape(-1, seq_len)
                eps_reference.requires_grad = True
                eps = eps.view(-1, 1)
                eps_pred = None
            else:  # MSE
                assert eps_q_high is not None
                eps_pred = eps_q_high.reshape(-1, 1)
                eps = eps.view(-1, 1)
                eps = torch.abs(eps)
                eps_reference = None
            if self._batch_mode == BATCH_MODE_ONE_TS:
                return dict(q_low=q_low, q_high=q_high, low_alpha=val_beta, high_alpha=(val_beta - val_alpha + 1) if val_beta is not None else None,   # For Metrics
                            alpha=torch.tensor(self._train_alphas, device=eps.device).unsqueeze(0).repeat(batches, 1),  # Overwrite batch alpha
                            eps=eps,
                            eps_predicted=eps_pred,     # Only MSE
                            eps_reference=eps_reference, weights=assoc_matrix, variance=self._mix_variance if self._mix_variance is not None else 0,  # Only Mix
                            loss_mask=padding_mask.view(-1, 1) if has_padding else None)
            else:
                return dict(Y=Y,
                            q_low=q_low, q_high=q_high, low_alpha=val_beta, high_alpha=(val_beta - val_alpha + 1) if val_beta is not None else None,   # For Metrics
                            alpha=torch.tensor(self._train_alphas, device=eps.device).unsqueeze(0).repeat(batches, 1),  # Overwrite batch alpha
                            eps=eps,
                            eps_predicted=eps_pred,     # Only MSE
                            eps_reference=eps_reference, weights=assoc_matrix, variance=self._mix_variance if self._mix_variance is not None else 0,  # Only Mix
                            loss_mask=padding_mask.view(-1, 1) if has_padding else None)

    def forward_pred_dist(self, ctx, Y, Y_hat, alpha, step_no, batches, seq_len, val):
        assert not self._eps_to_stored_pattern
        assert not self._head_per_alpha
        assert not self._alpha_in_final_layer
        assert not self._train_only_past
        assert not self._train_with_past_window
        assert not self._absolute_epsilons
        # TODO Work In Progress

        eps = score.get(Y=Y, Y_hat=Y_hat).detach()  # [batch, batch_size, *]
        eps_window_len = None
        # TODO Padding?
        eps_windowed = unfold_window(eps, eps_window_len)
        # Calc Stats for eps
        used_stats = ('moment1', 'moment2')
        eps_stats = calc_stats(eps_windowed, stats=used_stats)
        batch_encoded = self._encode_ctx(context=ctx, step_no=step_no)[0]
        mask = torch.diag(torch.full((seq_len,), fill_value=True, dtype=torch.bool, device=batch_encoded.device))
        _, eps_q_high, beta, alphas, assoc_matrix, _ = self._forward(
            current_ctx=batch_encoded[:, eps_window_len:, :], memory_ctx=batch_encoded, memory_eps=eps_stats, alpha=alpha,
            current_ctx_step=step_no[:, eps_window_len:, :], memory_ctx_step=step_no,
            retrieve_model_prediction=True, retrieve_association_matrix=val, mask=mask
        )
        # Extract stats
        stats_predicted = None
        # Get Intervals by inference Mode
        if val:
            Y_hat = Y_hat[:, eps_window_len:, :].reshape(-1, 1).repeat(len(self._train_alphas), 1)
            val_q_low, val_q_high, val_alpha, val_beta = self._train_metric_interval(
                eps=eps, assoc_matrix=assoc_matrix, batches=batches, seq_len=seq_len)
            q_low = Y_hat + score.resolve(val_q_low.unsqueeze(-1), **score_param, interleave=1, repeat=len(self._train_alphas))
            q_high = Y_hat + score.resolve(val_q_high.unsqueeze(-1), **score_param, interleave=1, repeat=len(self._train_alphas))
        else:
            q_low, q_high, val_beta, val_alpha = None, None, None, None

        return dict(q_low=q_low, q_high=q_high, low_alpha=val_beta,
                    high_alpha=(val_beta - val_alpha + 1) if val_beta is not None else None,  # For Metrics
                    alpha=torch.tensor(self._train_alphas, device=eps.device).unsqueeze(0).repeat(batches, 1), # Overwrite batch alpha
                    # For Stat loss
                    eps_stats=eps_stats, eps_stats_predicted=stats_predicted)

    def _forward_encode(self, **kwargs):
        ctx, step_no = kwargs['ctx_data'].detach(), kwargs['step_no'].detach()

        batches = ctx.shape[0]
        seq_len = ctx.shape[1]
        ctx_features = ctx.shape[2]

        batch_encoded = self._encode_ctx(context=ctx, step_no=step_no)[0]
        return batch_encoded, lambda x: x, batches, seq_len, ctx_features

    def _forward_encode_naive_mix(self, **kwargs):
        if self._history_comp_params is not None:
            ctx, step_no, ctx_hist, hist_size =\
                kwargs['ctx_data'].detach(), kwargs['step_no'].detach(), \
                kwargs['ctx_hist'].detach(), kwargs['real_hist_size']
        else:
            ctx, step_no, ctx_hist, hist_size = \
                kwargs['ctx_data'].detach(), kwargs['step_no'].detach(), None, None
        assert not self._train_with_past_window
        assert not self._train_only_past

        batches = 1
        seq_len = ctx.shape[0]
        ctx_features = ctx.shape[2]

        # ToDo If Memory problems -> This is probably the bootleneck (embedding of all histories at once) - One could loop
        batch_encoded = self._ctx_encoding(context=ctx, step_no=step_no, context_past=ctx_hist,
                                           context_past_state=None, past_real_len=hist_size)[0]
        #del ctx
        #del ctx_hist
        #del hist_size
        # ToDo If very big split in batches again
        # batch_encoded = batch_encoded.transpose(0, 1)
        # if self._limit_seq_to_mem_size and batch_encoded.shape[1] > self.max_mem_size:
        #   split_subsequence = lambda split: tuple(
        #       map(lambda x: self._split_in_subsequences(x, self.max_mem_size, self._sub_sequence_stride), split))
        # WIP
        return batch_encoded.transpose(0, 1), lambda x: x.transpose(0, 1), batches, seq_len, ctx_features

    def _record_attention(self, association_matrix, query_steps, key_steps):
        """
        :param association_matrix:  [
        :param query_steps:         [query_steps]
        :param key_steps:           [memory_steps]
        :return:
        """
        if self._save_attention_record:
            association_matrix = association_matrix.squeeze(1)  # Squeeze out sample dim because its only one
            return association_matrix.detach().to(torch.device('cpu')),\
                   query_steps.clone().detach().to(torch.device('cpu')),\
                   key_steps.clone().detach().to(torch.device('cpu'))
        else:
            return None

    #
    # Mode: Conformal Selection
    #
    def _train_metric_interval(self, eps, assoc_matrix, batches, seq_len):
        eps_q_low = torch.empty(batches * seq_len * len(self._train_alphas), dtype=torch.float,
                                device=assoc_matrix.device)
        eps_q_high = torch.empty_like(eps_q_low)
        val_beta = torch.empty_like(eps_q_low)
        val_alpha = torch.empty_like(eps_q_low)
        start = 0
        for a_idx, alpha in enumerate(self._train_alphas):
            val_alpha[a_idx * seq_len * batches:(a_idx + 1) * seq_len * batches] =\
                torch.full((seq_len * batches,), fill_value=alpha, device=assoc_matrix.device)
            for b_idx in range(batches):
                # TODO current quantile method does not work with batches!
                end = start + seq_len
                eps_q_low[start:end], eps_q_high[start:end], val_beta[start:end], _ = self._get_quantile_conformal(
                    assoc_matrix[b_idx].squeeze(0), eps=eps[b_idx], alpha=alpha, detach=True
                )
                start = end
        return eps_q_low, eps_q_high, val_alpha, val_beta

    def _reduce_assoc_matrix(self, association_matrix, alpha, reduce_alpha):
        association_matrix = association_matrix.detach()  # Hopfield is only trained by direct prediciton task
        if reduce_alpha:
            association_matrix = torch.sum(association_matrix, dim=0)
        if self._sum_assoc_for_qc:
            association_matrix = torch.sum(association_matrix, dim=0)
        else:
            assert isinstance(alpha, float)
            idx = self._train_alphas.index(alpha)
            association_matrix = association_matrix[idx]
        return association_matrix

    def _get_quantile_conformal(self, association_matrix, eps, alpha, detach=False, inference=False) -> torch.tensor:
        assert eps.shape[1] == 1
        debug = self._conf_quantile_mode == 'debug'
        add_info = {} if debug else None
        if detach:
            eps = eps.detach()
            association_matrix = association_matrix.detach()
        if self._conf_quantile_mode in ['sample', 'debug']:  # Sample
            # ToDo Maybe Scale association matrix (More or less sharp)
            sampled_selection = torch.multinomial(association_matrix, num_samples=1000, replacement=True)
            selected_eps = torch.cat([torch.index_select(eps.squeeze(1).unsqueeze(0), index=idx_, dim=1)
                                      for idx_ in sampled_selection], dim=0)
            if inference and debug:
                assert sampled_selection.shape[0] == 1
                add_info['alpha'] = alpha
                add_info['sample_base'] = association_matrix.shape[1]
                add_info['sample_count'] = torch.unique(sampled_selection, dim=1).shape[1]
                add_info['sample_sum'] = torch.index_select(association_matrix, index=sampled_selection[0].unique(), dim=1).sum().item()
        if self._conf_quantile_mode in ['cdf', 'debug']:  # CDF Approach
            if self._conformal_sel_abs:
                raise NotImplemented("Not Implemented")
                #return -quantile_value, quantile_value, alpha / 2
            else:
                device = association_matrix.device
                sorted_eps, sort_idx = torch.sort(eps.T, dim=1)
                del eps
                tmp = association_matrix[:, sort_idx.squeeze()]
                del sort_idx
                del association_matrix
                c_sum = torch.cumsum(tmp, dim=1)
                idx = torch.arange(c_sum.shape[1], 0, -1, device=device)
                tmp = torch.where(c_sum >= alpha / 2, 1, 0) * idx
                quantile_low = torch.argmax(tmp, 1, keepdim=True)  # Index
                quantile_low = sorted_eps[0, quantile_low]  # Value
                tmp = torch.where(c_sum >= 1 - (alpha / 2), 1, 0) * idx
                del c_sum
                quantile_high = torch.argmax(tmp, 1, keepdim=True)  # Index
                quantile_high = sorted_eps[0, quantile_high]  # Value
                if inference and debug:
                    add_info['cdf_low'] = quantile_low.item()
                    add_info['cdf_high'] = quantile_high.item()
                elif not debug:
                    return quantile_low.squeeze(), quantile_high.squeeze(), alpha / 2, add_info
        if False:  # Top K
            _, selected_idx = torch.topk(association_matrix, 50, dim=1)
            eps = eps.squeeze(1).unsqueeze(0)
            selected_eps = torch.cat([torch.index_select(eps, index=idx_, dim=1) for idx_ in selected_idx], dim=0)
        q_conformal_low, q_conformal_high, beta = self._calc_conformal_quantiles(
            selected_eps, alpha, no_beta_bins=self._conformal_sel_beta, use_absolute_eps=self._conformal_sel_abs
        )
        if inference and debug:
            add_info['sample_low'] = q_conformal_low.item()
            add_info['sample_high'] = q_conformal_high.item()
        return q_conformal_low, q_conformal_high, beta, add_info

    #
    # Persistence
    #

    def get_train_fingerprint(self) -> dict:
        return {
            "train_past_window": self._train_with_past_window,
            "train_only_past": self._train_only_past,
            "train_abs_epsilon": self._absolute_epsilons,
            "eps_mem_size": self.max_mem_size,
            "use_pre_calib_eps": self._use_pre_calib_eps_for_calib,
            "with_loss_weight": self._width_loss_weight,
            "coverage_loss_weight": self._coverage_loss_weight,
            "chung_loss_weight": self._chung_loss_weight,
            "loss_mode": self._loss_mode,
            "batch_size": self._batch_size,
            "ctx_mode": self._ctx_gen._full_mode,
            "ctx_past_window": self._ctx_past_window,
            "pre_encode_ctx": self._pre_encode_context,
            "history_compress": (('_'.join([f'{key}-{item}' for key, item in self._history_comp_params.items()]))
            if self._history_comp_params is not None else "None"),
            "pos_encode": (('_'.join([f'{key}-{item}' for key, item in self._pos_encode_params.items()]))
            if self._pos_encode_params is not None else "None"),
            "eps_to_stored_pattern": self._eps_to_stored_pattern,
            "alpha_in_final_layer": self._alpha_in_final_layer,
            "head_per_alpha": self._head_per_alpha,
            "predict_beta": self._predict_beta,
            "hopfield_heads": self._hopfield_heads,
            "alphas": self._train_alphas
        }

    def _get_constructor_parameters(self) -> dict:
        pass

    def to(self, *args, **kwargs):
        self.to_device(**kwargs)
        return super().to(*args, **kwargs)
