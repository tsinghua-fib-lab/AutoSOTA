conda create -n savvy python=3.12 -y
conda activate savvy
pip install -r requirements.txt 


# seg script env
mkdir third_party
cd third_party
git clone https://github.com/isl-org/ZoeDepth.git
git clone https://github.com/mit-han-lab/efficientvit.git
cd ..