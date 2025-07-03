echo "This script installs dependencies on: amazon/Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20250602"

source /opt/pytorch/bin/activate

pip install --upgrade pip

pip install numpy
pip install matplotlib
pip install pandas
pip install scipy
pip install scikit-learn
pip install jupyter
pip install pympler