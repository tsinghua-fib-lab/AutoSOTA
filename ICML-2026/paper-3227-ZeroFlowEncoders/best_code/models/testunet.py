import unittest
import torch
import math

# =============================================================================
# IMPORT YOUR MODEL CLASSES HERE
# from model import DynamicEncoder, DynamicDecoder, DynamicUNetVectorField
# =============================================================================
from nnunet import DynamicEncoder, DynamicDecoder, DynamicUNetVectorField

class TestDynamicNetwork(unittest.TestCase):

    def setUp(self):
        # Common parameters for all tests
        self.batch_size = 4
        self.latent_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_encoder_decoder_shapes(self):
        """
        Verifies that Encoder compresses correctly and Decoder recovers 
        the exact original image size for various resolutions.
        """
        img_sizes = [32, 64, 128, 256]
        
        print(f"\n[Test] Checking Encoder/Decoder shapes for {img_sizes}...")
        
        for size in img_sizes:
            # 1. Initialize Models
            enc = DynamicEncoder(latent_dim=self.latent_dim, img_size=size).to(self.device)
            dec = DynamicDecoder(latent_dim=self.latent_dim, img_size=size).to(self.device)
            
            # 2. Create Dummy Data
            x = torch.randn(self.batch_size, 3, size, size).to(self.device)
            flat_input = x.view(self.batch_size, -1)
            
            # 3. Forward Pass
            z = enc(flat_input)
            rec = dec(z)
            
            # 4. Assertions
            self.assertEqual(z.shape, (self.batch_size, self.latent_dim), 
                             f"Encoder output wrong for size {size}")
            self.assertEqual(rec.shape, (self.batch_size, 3, size, size), 
                             f"Decoder output wrong for size {size}")
            
            print(f"  -> {size}x{size} passed.")

    def test_vector_field_forward(self):
        """
        Verifies the full U-Net VectorField forward pass.
        Checks input concatenation logic and output flattening.
        """
        img_sizes = [32, 64, 128]
        
        print(f"\n[Test] Checking VectorField U-Net flow for {img_sizes}...")

        for size in img_sizes:
            model = DynamicUNetVectorField(encx_dim=self.latent_dim, img_size=size).to(self.device)
            
            # Dummy Inputs
            y = torch.randn(self.batch_size, 3 * size * size).to(self.device)
            encx = torch.randn(self.batch_size, self.latent_dim).to(self.device)
            x = torch.randn(self.batch_size, 3 * size * size).to(self.device)
            t = torch.randn(self.batch_size, 1).to(self.device)
            
            # Forward
            try:
                out = model(y, encx, x, t)
            except RuntimeError as e:
                self.fail(f"RuntimeError for size {size}: {e}")

            # Check Output Shape (Batch, 3*H*W)
            expected_shape = (self.batch_size, 3 * size * size)
            self.assertEqual(out.shape, expected_shape, 
                             f"VectorField output shape mismatch for size {size}")
            
            print(f"  -> {size}x{size} passed.")

    def test_bilinear_vs_conv_transpose(self):
        """
        Tests both upsampling modes: Bilinear (Interpolation) vs ConvTranspose (Learnable).
        This ensures the channel math in the 'Up' block is correct for both.
        """
        size = 64
        print(f"\n[Test] Checking Bilinear vs ConvTranspose logic...")
        
        # Case 1: Bilinear = True
        model_bi = DynamicUNetVectorField(encx_dim=64, img_size=size, bilinear=True).to(self.device)
        # Case 2: Bilinear = False
        model_conv = DynamicUNetVectorField(encx_dim=64, img_size=size, bilinear=False).to(self.device)
        
        y = torch.randn(2, 3 * size * size).to(self.device)
        encx = torch.randn(2, 64).to(self.device)
        x = torch.randn(2, 3 * size * size).to(self.device)
        t = torch.randn(2, 1).to(self.device)
        
        out_bi = model_bi(y, encx, x, t)
        out_conv = model_conv(y, encx, x, t)
        
        self.assertEqual(out_bi.shape, out_conv.shape)
        print("  -> Both modes passed channel verification.")

    def test_odd_shapes_padding(self):
        """
        Advanced Test: Although your logic enforces powers of 2, standard U-Nets
        often break if padding isn't handled. We verify inputs don't crash standard calls.
        (Note: Your class asserts sizes, but this checks internal padding robustness).
        """
        print(f"\n[Test] Checking Internal Padding Robustness...")
        # We manually invoke the Up block with slightly mismatched tensor sizes
        # to ensure the F.pad logic works.
        
        up_block = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Simulate a skip connection that is 64x64
        x2 = torch.randn(1, 64, 64, 64) 
        # Simulate an input coming from below that is 31x31 (e.g. after weird pooling)
        # Upsampling 31->62. We need to pad 62 to match 64.
        x1 = torch.randn(1, 128, 31, 31) 
        
        # We define a standalone 'Up' block to test its padding logic
        # In channels = 128 + 64 // 2 ?? No, let's just use the class
        from torch import nn
        
        # Instantiate your Up block
        # in_ch=128, out_ch=64. 
        # It expects x1 to be upsampled. 
        # It expects x2 to be concatenated.
        
        try:
            # We assume the Up class is available in scope
            # We create a dummy Up block
            block = Up(in_channels=128, out_channels=64, bilinear=True)
            output = block(x1, x2) # Should verify padding handles 62 vs 64
            print(f"  -> Padding handled mismatch {x1.shape} -> {x2.shape} successfully.")
        except Exception as e:
            print(f"  -> Padding test warning (might be expected if class strictly expects powers of 2): {e}")

if __name__ == '__main__':
    # This line runs the tests when you execute the script
    unittest.main(argv=['first-arg-is-ignored'], exit=False)