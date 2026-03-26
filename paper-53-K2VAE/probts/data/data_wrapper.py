import torch

class ProbTSBatchData:
    input_names_ = [
        'target_dimension_indicator',
        'past_time_feat',
        'past_target_cdf',
        'past_observed_values',
        'past_is_pad',
        'future_time_feat',
        'future_target_cdf',
        'future_observed_values',
    ]
    
    def __init__(self, data_dict, device):
        # Initialize attributes from the provided data dictionary
        self.__dict__.update(data_dict)
        self.__dict__['context_length'] = data_dict.get('context_length', None)
        self.__dict__['prediction_length'] = data_dict.get('prediction_length', None)
        self.__dict__['max_context_length'] = data_dict.get('max_context_length', None)
        self.__dict__['max_prediction_length'] = data_dict.get('max_prediction_length', None)
        
        # Expand dimensions for univariate data
        if len(self.__dict__['past_target_cdf'].shape) == 2:
            self._expand_dimensions()
        
        # Set tensors to the specified device
        self._set_device(device)
        # Fill missing inputs with None
        self._ensure_all_inputs_present()
        # Process padding for observed values
        self._process_padding()

    def _ensure_all_inputs_present(self):
        """Ensure all expected inputs are present in the data."""
        for input in self.input_names_:
            if input not in self.__dict__:
                self.__dict__[input] = None

    def _set_device(self, device):
        """Move all tensors to the specified device."""
        for k, v in self.__dict__.items():
            if v is not None and torch.is_tensor(v):
                v.to(device)
        self.device = device

    def _expand_dimensions(self):
        """Expand dimensions for target-related tensors if necessary."""
        self.__dict__["target_dimension_indicator"] = self.__dict__["target_dimension_indicator"][:, :1]
        for input in ['past_target_cdf','past_observed_values','future_target_cdf','future_observed_values']:
            self.__dict__[input] = self.__dict__[input].unsqueeze(-1)

    def _process_padding(self):
        """Adjust observed values based on the padding indicator."""
        if self.__dict__['past_is_pad'] is not None:
            self.__dict__["past_observed_values"] = torch.min(
                self.__dict__["past_observed_values"],
                1 - self.__dict__["past_is_pad"].unsqueeze(-1)
            )

