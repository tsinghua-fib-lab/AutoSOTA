import re

with open("/repo/simple_snn.py", "r") as f:
    content = f.read()

# Add FragNorm between fc1->LIF1, fc2->LIF2, fc3->LIF3
# Original forward:
#   x = self.fc1(x)
#   x = self.lif1(x)
#   x = self.fc2(x)
#   x = self.lif2(x)
#   x = self.fc3(x)
#   x = self.lif3(x)

old_forward = """    def forward(self, x):
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn(x)
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        x = self.lif3(x)
        x = self.fc4(x)
        x = self.lif4(x)

        return x"""

new_forward = """    def forward(self, x):
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn(x)
        x = self.fc1(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn2(x)
        x = self.lif2(x)
        x = self.fc3(x)
        if Frag_on or Learnable_on or Dynamic_on:
            x = self.fn3(x)
        x = self.lif3(x)
        x = self.fc4(x)
        x = self.lif4(x)

        return x"""
content = content.replace(old_forward, new_forward)

# Add self.fn1, self.fn2, self.fn3 to __init__
# Current init has: self.fn = FragNorm(...) then fc1
old_init_fn = """            self.fn = FragNorm(num_features=input_dim, time_aggregate=False, affine=True,
                               track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, 1024, bias=bias)"""
new_init_fn = """            self.fn = FragNorm(num_features=input_dim, time_aggregate=False, affine=True,
                               track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fn1 = FragNorm(num_features=1024, time_aggregate=False, affine=True,
                            track_running_stats=False, momentum=0.1, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, 1024, bias=bias)"""
content = content.replace(old_init_fn, new_init_fn)

# Add fn2 after fc2/LIF2
old_fc2 = "        self.fc2 = nn.Linear(1024, 512, bias=bias)\n        self.lif2"
new_fc2 = "        self.fn2 = FragNorm(num_features=512, time_aggregate=False, affine=True,\n                            track_running_stats=False, momentum=0.1, eps=1e-5)\n        self.fc2 = nn.Linear(1024, 512, bias=bias)\n        self.lif2"
content = content.replace(old_fc2, new_fc2)

# Add fn3 after fc3/LIF3
old_fc3 = "        self.fc3 = nn.Linear(512, 128, bias=bias)\n        self.lif3"
new_fc3 = "        self.fn3 = FragNorm(num_features=128, time_aggregate=False, affine=True,\n                            track_running_stats=False, momentum=0.1, eps=1e-5)\n        self.fc3 = nn.Linear(512, 128, bias=bias)\n        self.lif3"
content = content.replace(old_fc3, new_fc3)

with open("/repo/simple_snn.py", "w") as f:
    f.write(content)

print("IDEA-02 patch applied")
