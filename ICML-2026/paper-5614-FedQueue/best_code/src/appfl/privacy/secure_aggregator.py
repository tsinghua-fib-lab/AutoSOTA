import hmac
import hashlib
import struct
import math
from typing import Dict, Tuple, Iterable, List
import torch


class SecureAggregator:
    """
    Pairwise-masking secure aggregation for privacy-preserving federated learning.

    This implementation uses pairwise masking to enable secure aggregation where
    the server can compute the sum of client updates without seeing individual
    client contributions. Each client masks their model update with random masks
    that are derived deterministically from shared secrets. When all masked updates
    are summed, the masks cancel out, leaving only the true aggregate.

    **How It Works:**
        1. Each client computes their model update (delta = local - global)
        2. For each pair of clients (i, j), both derive the same random mask from
           a shared secret using a PRG (Pseudo-Random Generator)
        3. Client i adds the mask, client j subtracts it
        4. When the server sums all masked updates, pairwise masks cancel:
           (delta_A + mask_AB) + (delta_B - mask_AB) = delta_A + delta_B

    **Security Guarantees:**
        - Server cannot see individual client updates (only the aggregate)
        - Assumes honest-but-curious server (follows protocol but tries to learn data)
        - Protects against server-only attacks
        - Uses cryptographic PRG (HMAC-SHA256) for unpredictable masks

    **Critical Limitations:**
        1. **No Dropout Tolerance**: ALL clients that start a round MUST complete it.
           If any client drops out after the client list is finalized, their masks
           will not cancel, corrupting the aggregated result.

        2. **Collusion Vulnerability**: If the server colludes with one or more clients,
           they can reconstruct other clients' updates by subtracting known values.

        3. **No Malicious Client Protection**: A malicious client can send arbitrary
           values to disrupt aggregation. This protocol does not verify correctness.

        4. **Synchronous Rounds**: All clients must participate in each round with
           the exact same client list and round ID.

    **When to Use This Protocol:**
        ✓ Trusted, reliable clients (low dropout rate)
        ✓ Honest-but-curious server threat model
        ✓ Need to hide individual updates from server
        ✓ Can tolerate occasional round failures from dropouts

    **When NOT to Use:**
        ✗ High client dropout rates (use threshold-based schemes instead)
        ✗ Malicious clients possible (need verification mechanisms)
        ✗ Server-client collusion is a concern (need multi-party computation)
        ✗ Asynchronous federated learning

    **Security Assumptions:**
        - Shared secret is distributed securely to all clients beforehand
        - Clients correctly implement the protocol (not Byzantine)
        - Communication channels are authenticated (prevent MITM attacks)
        - Server is honest-but-curious (not malicious)
        - No collusion between server and clients

    Args:
        client_id: Unique identifier for this client
        all_client_ids: Complete list of all participating client IDs for this round
                       (must be identical across all clients)
        secret: Shared secret bytes used to derive pairwise masks (must be
                identical across all clients)
        device: Torch device for computation (cpu or cuda)

    Example:
        >>> # Setup (all clients must have same secret and client list)
        >>> secret = b"shared_secret_key"
        >>> all_clients = ["client_A", "client_B", "client_C"]
        >>> device = torch.device("cpu")
        >>>
        >>> # Each client creates their aggregator
        >>> sa = SecureAggregator("client_A", all_clients, secret, device)
        >>>
        >>> # Client computes delta and masks it
        >>> delta = {"layer1": local_state["layer1"] - global_state["layer1"]}
        >>> masked_flat, shapes = sa.mask_update(delta, round_id=0)
        >>>
        >>> # Server aggregates all masked updates
        >>> aggregated = sum(all_masked_flats)
        >>> result = SecureAggregator.unflatten_to_state_dict(aggregated, shapes, device)
        >>> # result now contains sum of all deltas, masks cancelled out
    """

    def __init__(
        self,
        client_id: str,
        all_client_ids: Iterable[str],
        secret: bytes,
        device: torch.device,
    ):
        self.client_id = str(client_id)
        self.all_client_ids = sorted([str(x) for x in all_client_ids])
        self.secret = secret
        self.device = device

        # Validation
        if not self.all_client_ids:
            raise ValueError("all_client_ids cannot be empty")
        if len(set(self.all_client_ids)) != len(self.all_client_ids):
            raise ValueError("all_client_ids contains duplicate IDs")
        if self.client_id not in self.all_client_ids:
            raise ValueError(f"client_id '{self.client_id}' not in all_client_ids")
        if not secret:
            raise ValueError("secret cannot be empty")

    # ---- helpers ----
    @staticmethod
    def flatten_state_dict(
        sd: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
        """
        Flatten a state dictionary into a single 1D tensor.

        Converts all tensors in the state dict to float32 and concatenates them
        into a single flat tensor, while preserving metadata needed for unflattening.

        Args:
            sd: State dictionary mapping parameter names to tensors

        Returns:
            Tuple of (flat_tensor, shapes) where:
                - flat_tensor: 1D torch.Tensor containing all parameters
                - shapes: List of (name, shape_tuple) for reconstruction

        Note:
            All tensors are converted to float32, which may lose precision for
            models using other dtypes (e.g., float64).
        """
        parts = []
        shapes = []
        for k, v in sd.items():
            t = v.detach().to(torch.float32).reshape(-1)
            parts.append(t)
            shapes.append((k, tuple(v.shape)))
        flat = (
            torch.cat(parts)
            if len(parts) > 0
            else torch.tensor([], dtype=torch.float32)
        )
        return flat, shapes

    @staticmethod
    def unflatten_to_state_dict(
        flat: torch.Tensor,
        shapes: List[Tuple[str, Tuple[int, ...]]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct a state dictionary from a flattened tensor.

        Reverses the operation of flatten_state_dict by reshaping the flat tensor
        back into the original state dict structure.

        Args:
            flat: 1D tensor containing flattened parameters
            shapes: List of (name, shape_tuple) from flatten_state_dict
            device: Device to place the reconstructed tensors on

        Returns:
            Dictionary mapping parameter names to tensors with original shapes
        """
        out = {}
        idx = 0
        for name, shp in shapes:
            num = 1
            for d in shp:
                num *= int(d)
            seg = flat[idx : idx + num].reshape(shp).to(device)
            out[name] = seg
            idx += num
        return out

    # deterministic seed derivation (symmetric)
    def _derive_seed(self, round_id: int, a: str, b: str) -> bytes:
        """
        Derive a deterministic, symmetric seed for a client pair.

        Generates a cryptographically secure seed that is identical for both
        clients in a pair (seed(A,B) == seed(B,A)), enabling them to generate
        the same pairwise mask without communication.

        Args:
            round_id: Training round number (prevents mask reuse across rounds)
            a: First client ID
            b: Second client ID

        Returns:
            32-byte seed derived via HMAC-SHA256

        Note:
            Symmetry is achieved by sorting client IDs lexicographically,
            ensuring both clients compute the same seed regardless of order.
        """
        if a > b:
            a, b = b, a
        msg = f"rnd={round_id}|a={a}|b={b}".encode()
        return hmac.new(self.secret, msg, hashlib.sha256).digest()

    # PRG -> float32 normals (Box-Muller)
    def _seed_to_normals(self, numel: int, seed: bytes) -> torch.Tensor:
        """
        Generate `numel` standard normal random values deterministically from `seed`.
        Uses Box-Muller transform with full 32-bit precision (no bias).

        Box-Muller transform: Given two uniform random values u1, u2 in (0,1),
        produces two independent standard normal values z1, z2.
        """
        vals = []
        ctr = 0
        while len(vals) < numel:
            block = hmac.new(seed, struct.pack(">Q", ctr), hashlib.sha256).digest()
            for off in range(0, len(block), 8):
                if len(vals) >= numel:
                    break
                chunk = block[off : off + 8]
                if len(chunk) < 8:
                    break
                u = struct.unpack(">Q", chunk)[0]  # 64-bit int
                # u1 = (((u >> 32) & 0xFFFFFFFF) % (10**9) + 0.5) / (10**9 + 1.0)
                # u2 = ((u & 0xFFFFFFFF) % (10**9) + 0.5) / (10**9 + 1.0)
                # z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                # vals.append(z)

                # Alternative approach using full 32-bit precision without modulo bias:
                # Extract two 32-bit integers from the 64-bit value
                u1_int = (u >> 32) & 0xFFFFFFFF  # Upper 32 bits
                u2_int = u & 0xFFFFFFFF  # Lower 32 bits

                # Convert to (0, 1) range with full precision and no bias
                # Using (x + 1) / (2^32 + 1) ensures strictly in (0, 1)
                # and uses all 32 bits without modulo bias
                u1 = (u1_int + 1.0) / (2**32 + 1.0)
                u2 = (u2_int + 1.0) / (2**32 + 1.0)

                # Box-Muller transform: produces two independent normals
                r = math.sqrt(-2.0 * math.log(u1))
                z1 = r * math.cos(2.0 * math.pi * u2)
                z2 = r * math.sin(2.0 * math.pi * u2)

                # Use both values for efficiency
                vals.append(z1)
                if len(vals) < numel:
                    vals.append(z2)
            ctr += 1
        return torch.tensor(vals[:numel], dtype=torch.float32, device=self.device)

    def make_pairwise_mask(self, flat_len: int, round_id: int) -> torch.Tensor:
        """
        Generate the combined pairwise mask for this client.

        Creates a mask by summing (or subtracting) pairwise random masks with
        all other clients. The sign depends on lexicographic ordering to ensure
        cancellation when all client masks are summed.

        Algorithm:
            For each other client j:
                - Derive shared seed for pair (self, j)
                - Generate random mask from seed
                - If self_id < j_id: add mask
                - If self_id > j_id: subtract mask

        When all clients sum their masked updates, each pairwise mask appears
        once as +mask and once as -mask, causing cancellation.

        Args:
            flat_len: Length of the flattened model parameter vector
            round_id: Current training round (incorporated into seed)

        Returns:
            1D tensor of length flat_len containing the combined mask

        Note:
            Computational cost is O(num_clients × flat_len), which can be
            expensive for many clients or large models.
        """
        mask = torch.zeros(flat_len, dtype=torch.float32, device=self.device)
        for other in self.all_client_ids:
            if other == self.client_id:
                continue
            seed = self._derive_seed(round_id, self.client_id, other)
            r = self._seed_to_normals(flat_len, seed)
            if self.client_id < other:
                mask += r
            else:
                mask -= r
        return mask

    def mask_update(
        self, delta_state: Dict[str, torch.Tensor], round_id: int
    ) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
        """
        Mask a model update (delta) for secure aggregation.

        This is the main method clients call to prepare their model updates for
        secure aggregation. It flattens the delta, adds pairwise masks, and
        returns the masked values ready for transmission to the server.

        Workflow:
            1. Flatten delta_state into 1D tensor
            2. Generate pairwise mask for this client
            3. Add mask to flattened delta
            4. Return masked tensor and shape metadata

        The server can then sum all masked updates from all clients, and the
        pairwise masks will cancel out, leaving only the true aggregate.

        Args:
            delta_state: Model update as state_dict (typically local - global)
            round_id: Current training round number

        Returns:
            Tuple of (masked_flat_tensor, shapes) where:
                - masked_flat_tensor: 1D tensor with pairwise masks added
                - shapes: List of (name, shape) tuples for reconstruction

        Example:
            >>> # Client computes model update
            >>> delta = {k: local[k] - global_model[k] for k in local}
            >>> # Mask it for secure aggregation
            >>> masked_flat, shapes = sa.mask_update(delta, round_id=5)
            >>> # Send (masked_flat, shapes) to server

        Warning:
            All clients must use the SAME round_id, or masks will not cancel!
        """
        flat, shapes = self.flatten_state_dict(delta_state)
        mask = self.make_pairwise_mask(flat.numel(), round_id)
        return (flat + mask), shapes
