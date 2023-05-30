rm_cache:
	find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" -o -name ".vscode" \) -exec rm -rf {} \;
make_dirs:
	mkdir -p data ckpts reports figs
requirements:
	#pip install -r requirements.txt
download_data:
	mkdir -p data/CIFAR10
	git clone https://github.com/YoongiKim/CIFAR-10-images data/CIFAR10/

