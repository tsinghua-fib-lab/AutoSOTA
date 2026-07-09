# Backward Compatibility Guide

## Overview

The model has been renamed from `Mantis8M` to `MantisV1` and the internal transformer module has been renamed from `ViTUnit` to `TransformerUnit`. Full backward compatibility is maintained for existing code and checkpoints.

## Usage Examples

### New Code (Recommended)
```python
from mantis.architecture import MantisV1

# Create new model
network = MantisV1(seq_len=512, hidden_dim=256)

# Load from HuggingFace (even old checkpoints with ViTUnit inside)
network = MantisV1.from_pretrained("paris-noah/Mantis-8M")

# Access the transformer unit with new name
type(network.transf_unit).__name__ == 'TransformerUnit' # True
type(network.transf_unit).__name__ == 'ViTUnit' # False
network.vit_unit # returns `network.transf_unit`
```

### Old Code (Still Works)
```python
from mantis.architecture import Mantis8M

# Create old model
network = Mantis8M(seq_len=512, hidden_dim=256)

# Load from HuggingFace
network = Mantis8M.from_pretrained("paris-noah/Mantis-8M")

# Access the transformer unit with old name
type(network.vit_unit).__name__ == 'ViTUnit' # True
type(network.vit_unit).__name__ == 'TransformerUnit' # False
network.transf_unit # AttributeError: type object 'Mantis8M' has no attribute 'transf_unit'
```

## How It Works

### Class Aliases
- `Mantis8M` is kept for legacy
- `ViTUnit` is a copy of `TransformerUnit`

### Property Aliases
- `MantisV1` (formerly `Mantis8M`) has a `vit_unit` property that returns `self.transf_unit`
- The property is read/write, so calling `model.vit_unit = ...` for MantisV1 still works

### Checkpoint Loading
The `MantisV1.from_pretrained()` method automatically handles old checkpoints:
1. Loads the checkpoint from HuggingFace
2. Renames state dict keys from `vit_unit.*` to `transf_unit.*`
3. Handles module references for seamless loading

## Migration Path

You don't need to update your code immediately, except you used to directly access `vit_utils` subpackage, which does not exist anymore. Otherwise, both old and new names work interchangeably. Consider using `MantisV1` instead of `Mantis8M` to ensure your code is future-proof as the library evolves.
